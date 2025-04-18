from typing import Tuple, Iterable, Optional
from enum import Enum
from collections import Counter

import math
import logging
import itertools
from copy import deepcopy
from scipy.special import digamma

from ._common import EPS, _CommonMorfessorBase
from .baseline import DataPoint
from morfessor.loss.cost import FrequencyDistributionMode, EmCost
from morfessor.util.emprune.criteria import PruneStats, PruneDecision, PruningCriterion, prune_cost_at_alpha, AutotunePruningCriterion
from ..util.misc import tail, logsumexp

_logger = logging.getLogger(__name__)


class LateenMode(Enum):
    NONE = 1
    FULL = 2
    PRUNE = 3


class MorfessorEMPrune(_CommonMorfessorBase):
    def __init__(self,
                 corpusweight, skip_frequent_reanalysis, constr_methods,
                 seed_strings: Iterable[Tuple[int,str]]=None,
                 nolexcost: bool=False,
                 freq_distr: FrequencyDistributionMode=FrequencyDistributionMode.BASELINE):
        """
        :param nolexcost: ignore lexicon cost with EM+prune
        :param seed_strings: substring set to prune from
        :param freq_distr: ?
        """
        super().__init__(corpusweight=corpusweight, skip_frequent_reanalysis=skip_frequent_reanalysis, constr_methods=constr_methods)
        self._stored_corpus = Counter()

        self.cost = EmCost(self.cc, corpusweight, nolexcost, freq_distr)
        self.cost.load_lexicon(seed_strings)

    def e_step_soft(self, raw_words: Counter[str], maxlen: int):
        expected = Counter()
        tot_cost = 0
        for compound, freq in raw_words.items():
            w_expected, cost = self._forward_backward(compound, freq, maxlen)
            expected.update(w_expected)
            tot_cost += cost
        return expected, tot_cost

    def e_step_hard(self, raw_words: Counter[str], maxlen: int) -> Tuple[Counter[str],float]:
        expected = Counter()
        tot_cost = 0
        for compound, freq in raw_words.items():
            constructions, cost = self.viterbi_segment(compound, addcount=0.0, maxlen=maxlen)
            for cons in constructions:
                expected[cons] += freq
            tot_cost += cost
        return expected, tot_cost

    def m_step(self, expected: Counter[str], expected_freq_threshold: float, noexpdigamma: bool=False):
        # prune out infrequent
        # FIXME: is protecting length 1 useful? max(c, 1e-6))?
        expected = Counter(
            dict((w, c) for (w, c) in expected.items()
                 if c > expected_freq_threshold or len(w) == 1))

        if not noexpdigamma:
            # apply exp digamma for Bayesianified/DPified EM
            # acts as a sparse prior
            # https://cs.stanford.edu/~pliang/papers/tutorial-acl2007-talk.pdf
            tot = sum(expected.values())
            multiplier = tot / math.exp(digamma(tot))
            for construction in expected.keys():
                expected[construction] = math.exp(digamma(expected[construction])) * multiplier

        # set model parameters
        self.cost.counts = expected

    def prune_lexicon(self, prune_criterion: PruningCriterion, lateen: LateenMode, maxlen: int, expected_freq_threshold: float):
        self.cost.reset()
        if lateen == LateenMode.PRUNE:
            em_params = deepcopy(self.cost)
            _logger.info("Lateen Prune: using Viterbi counts for pruning")
            expected, cost = self.e_step_hard(maxlen=maxlen)
            self.m_step(expected, expected_freq_threshold=expected_freq_threshold)

            prune_stats = list(self.compute_prune_stats())
            pruned, done = prune_criterion.prune(prune_stats)
            n_pruned = len(pruned)

            if isinstance(prune_criterion, AutotunePruningCriterion):
                self.set_corpus_coding_weight(prune_criterion.optimal_alpha)

            _logger.info("Lateen Prune: restoring soft EM counts")
            del self.cost  # TODO [Bauwens]: What is the point of the criterion setting alpha in self.cost above, if you're going to replace it by an earlier deepcopy?
            self.cost = em_params
        else:
            prune_stats = list(self.compute_prune_stats())
            pruned, done = prune_criterion.prune(prune_stats)
            n_pruned = len(pruned)

            if isinstance(prune_criterion, AutotunePruningCriterion):
                self.set_corpus_coding_weight(prune_criterion.optimal_alpha)

        for construction in pruned:
            # prune out selected constructions
            count = self.cost.counts[construction]
            self.cost.update(construction, -count)
            del self.cost.counts[construction]

        _logger.info(f"Pruned {n_pruned} constructions.")

        return self.get_cost(), done

    def compute_prune_stats(self) -> Iterable[PruneStats]:
        orig_lc, orig_cc = self.cost.cost_before_tuning()
        constructions = list(w for w, c in self.cost.counts.most_common())
        current_alpha = self.get_corpus_coding_weight()
        for construction in constructions:
            if len(construction) == 1:
                yield PruneStats(construction, -math.inf, 0, 0, 0, PruneDecision.NEVER_CHAR)
                continue

            # assume all probability mass goes to viterbi segmentation
            replacement, _ = self.viterbi_segment(construction, taboo=[construction], addcount=0)
            if replacement == construction:
                yield PruneStats(construction, -math.inf, 0, 0, 0, PruneDecision.NEVER_NO_ALT)
                continue

            count = self.cost.counts[construction]

            # apply change
            self.cost.update(construction, -count)
            for replcons in replacement:
                self.cost.update(replcons, count)
            lc, cc = self.cost.cost_before_tuning()
            # revert change
            self.cost.update(construction, count)
            for replcons in replacement:
                self.cost.update(replcons, -count)

            delta_lc = lc - orig_lc
            delta_cc = cc - orig_cc
            threshold_alpha, delta_cost, decision = prune_cost_at_alpha(current_alpha, delta_lc, delta_cc)
            if self._is_semisupervised():
                # this only protects currently active annotations
                if self.cost._annot_coding.constructions.get(construction, 0) > 0:
                    decision = PruneDecision.NEVER_SUPERVISED
            yield PruneStats(construction,
                             threshold_alpha,
                             delta_lc, delta_cc,
                             delta_cost, decision)

    def train_em_prune(self,
                       corpus: Optional[Iterable[Tuple[str,int]]],
                       prune_criterion: PruningCriterion,
                       max_epochs: int=5, sub_epochs: int=3,
                       expected_freq_threshold: float=0.5,
                       maxlen: int=30, lateen: LateenMode=LateenMode.NONE, noexpdigamma: bool=False):
        """
        Run Morfessor EM+Prune training on the given corpus.

        TODO: The result of this process should be a set of unigrams with probabilities to be used for Viterbi segmentation.
              As far as I can see, self.cost.counts is what you actually want.
        """
        corpus = Counter(dict(corpus)) if corpus is not None else self._stored_corpus
        done = False
        epoch = 0
        while epoch < max_epochs:
            epoch += 1
            for sub_epoch in range(sub_epochs):
                # E-step
                if lateen == LateenMode.FULL and sub_epoch == sub_epochs - 1:
                    _logger.info("Lateen EM: using Viterbi e-step")
                    expected, cost = self.e_step_hard(corpus, maxlen=maxlen)
                else:
                    expected, cost = self.e_step_soft(corpus, maxlen=maxlen)
                _logger.info("E-step cost: %s tokens: %s" % (cost, self.cost.all_tokens()))

                # Optionally add semi-supervision to the results of the E-step
                if self._is_semisupervised():
                    self._update_annotation_choices()
                    self.cost._annot_coding.update_weight()
                    for constr, count in self.cost._annot_coding.constructions.items():
                        expected[constr] += self.cost._annot_coding.weight * count

                # M-step
                self.m_step(expected, expected_freq_threshold=expected_freq_threshold, noexpdigamma=noexpdigamma)

            if done:
                break

            # Cost-based pruning of lexicon
            cost, done = self.prune_lexicon(prune_criterion, lateen,
                                            maxlen=maxlen, expected_freq_threshold=expected_freq_threshold)
            lc, cc = self.cost.cost_before_tuning()
            _logger.info("Cost after pruning: %s types: %s tokens: %s" % (cost, self.cost.types(), self.cost.all_tokens()))
            _logger.info("Unweighted corpus cost: %s lexicon cost: %s" % (cc, lc))
            if done:  # TODO: No break here?
                _logger.info("Reached pruning goal")

        return epoch, self.get_cost()

    def _forward_backward(self, compound: str, freq: int, maxlen: int=30):
        grid_alpha = {'start': (0.0, None)}
        grid_beta  = {'stop' : (0.0, None)}
        tokens = self.cost.all_tokens()
        logtokens = math.log(tokens) if tokens > 0 else 0

        local_morph_costs = {}

        badlikelihood = self.cost.bad_likelihood(compound, 0)

        ## Forward pass
        for t in itertools.chain(self.cc.split_locations(compound), ['stop']):
            # logsum of all paths to current node.
            # Note that we can come from any node in history.
            negcosts = []

            for pt in tail(maxlen, itertools.chain(['start'], self.cc.split_locations(compound, stop=t))):
                if grid_alpha[pt][0] is None:
                    continue
                construction = self.cc.slice(compound, pt, t)
                if construction not in local_morph_costs:
                    count = self.get_construction_count(construction)
                    if count > 0:
                        cost = (logtokens - math.log(count))
                    elif self.cc.is_atom(construction):
                        cost = badlikelihood
                    else:
                        local_morph_costs[construction] = None
                        continue
                    assert cost >= 0
                    local_morph_costs[construction] = cost
                cost = local_morph_costs[construction]
                if cost is None:
                    continue
                cost += grid_alpha[pt][0]
                #_logger.debug("cost(%s)=%.2f", construction, cost)
                negcosts.append(-cost)
            totcost = -logsumexp(negcosts)
            grid_alpha[t] = (totcost, None)

        ## Backward pass
        for t in itertools.chain(reversed(list(self.cc.split_locations(compound))), ['start']):
            negcosts = []
            for pt in itertools.islice(
                    itertools.chain(self.cc.split_locations(compound, start=t), ['stop']), maxlen):
                if grid_beta[pt][0] is None:
                    continue
                construction = self.cc.slice(compound, t, pt)
                cost = local_morph_costs[construction]
                if cost is None:
                    continue
                cost += grid_beta[pt][0]
                negcosts.append(-cost)
            totcost = -logsumexp(negcosts)
            grid_beta[t] = (totcost, None)

        ## Merge pass
        w_expected = Counter()
        totcost = grid_alpha['stop'][0]
        # grid_alpha['stop'][0], grid_beta['start'][0] are approx equal
        for t in itertools.chain(self.cc.split_locations(compound), ['stop']):
            for pt in tail(maxlen, itertools.chain(['start'], self.cc.split_locations(compound, stop=t))):
                # grid_alpha[pt][0] is the total probability of all paths ending at pt
                # grid_beta[t][0] is the total probability of all paths starting at t
                # the compound pt:t probability is the same as cached previously
                if grid_alpha[pt][0] is None:
                    continue
                if grid_beta[t][0] is None:
                    continue
                construction = self.cc.slice(compound, pt, t)
                cost = local_morph_costs[construction]
                if cost is None:
                    continue
                expect = math.exp(-grid_alpha[pt][0] -grid_beta[t][0] -cost + totcost)
                if expect > 1:
                    occurs = compound.count(construction)
                    assert expect <= occurs + EPS, f'"{construction}" has expect {expect} occurs {occurs}'
                w_expected[construction] += freq * expect

        return w_expected, freq * totcost

    def _getViterbiBoundaryCost(self) -> float:
        return 0.0

    def _add_compound(self, compound: str, count: int):
        self.cost.update_boundaries(compound, count)
        self._stored_corpus[compound] += count

    def _load_compound(self, dp: DataPoint):
        self._add_compound(dp.compound, dp.count)

    def _get_corpus_frequency(self, compound: str) -> int:
        return self._stored_corpus[compound]

    def get_compounds(self) -> Iterable[str]:
        return self._stored_corpus.keys()

    def get_compound_counts(self) -> Iterable[Tuple[str,int]]:
        return self._stored_corpus.items()
