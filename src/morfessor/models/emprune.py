from typing import Tuple
from enum import Enum
from collections import Counter

import math
import logging
from copy import deepcopy
from scipy.special import digamma

from .baseline import BaselineModel, ConstructionNode, DataPoint
from ..util.cost import FrequencyDistributionMode, EmCost
from ..util.criteria import PruneStats, PruneDecision, PruningCriterion, prune_cost_at_alpha, AutotunePruningCriterion

_logger = logging.getLogger(__name__)


class LateenMode(Enum):
    NONE = 1
    FULL = 2
    PRUNE = 3


class MorfessorEMPrune(BaselineModel):
    def __init__(self,
                 corpusweight, use_skips, force_splits, nosplit_re,
                 em_substr=None,
                 nolexcost: bool=False,
                 freq_distr: FrequencyDistributionMode=FrequencyDistributionMode.BASELINE):
        """
        :param nolexcost: ignore lexicon cost with EM+prune
        :param em_substr: substring lexicon
        :param freq_distr: ?
        """
        super().__init__(corpusweight, use_skips, force_splits, nosplit_re)

        self.cost = EmCost(self.cc, corpusweight, nolexcost, freq_distr)
        self.cost.load_lexicon(em_substr)

    def e_step(self, maxlen: int):
        expected = Counter()
        compounds = list(self.get_compound_counts())
        tot_cost = 0
        for compound, freq in compounds:
            w_expected, cost = self._forward_backward(compound, freq, maxlen)
            expected.update(w_expected)
            tot_cost += cost
        return expected, tot_cost

    def e_step_hard(self, maxlen: int) -> Tuple[Counter[str],float]:
        expected = Counter()
        compounds = list(self.get_compound_counts())
        tot_cost = 0
        for compound, freq in compounds:
            constructions, cost = self.viterbi_segment(compound, addcount=0.0, maxlen=maxlen)
            for cons in constructions:
                expected[cons] += freq
            tot_cost += cost
        return expected, tot_cost

    def m_step(self, expected: Counter[str], expected_freq_threshold: int, noexpdigamma: bool=False):
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

    def prune_lexicon(self, prune_criterion: PruningCriterion, lateen: LateenMode, maxlen: int, expected_freq_threshold: int):
        self.cost.reset()
        if lateen == LateenMode.PRUNE:
            em_params = deepcopy(self.cost)
            _logger.info('Lateen Prune: using Viterbi counts for pruning')
            expected, cost = self.e_step_hard(maxlen=maxlen)
            self.m_step(expected, expected_freq_threshold=expected_freq_threshold)

            prune_stats = list(self.compute_prune_stats())
            pruned, done = prune_criterion.prune(prune_stats)
            n_pruned = len(pruned)

            if isinstance(prune_criterion, AutotunePruningCriterion):
                self.set_corpus_coding_weight(prune_criterion.optimal_alpha)

            _logger.info('Lateen Prune: restoring soft EM counts')
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

    def compute_prune_stats(self):
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
            if self._supervised:
                # this only protects currently active annotations
                if self.cost._annot_coding.constructions.get(construction, 0) > 0:
                    decision = PruneDecision.NEVER_SUPERVISED
            yield PruneStats(construction,
                             threshold_alpha,
                             delta_lc, delta_cc,
                             delta_cost, decision)

    def train_em_prune(self, prune_criterion: PruningCriterion,
                       max_epochs: int=5, sub_epochs: int=3,
                       expected_freq_threshold: float=0.5,
                       maxlen: int=30, lateen: LateenMode=LateenMode.NONE, noexpdigamma: bool=False):
        done = False
        for epoch in range(max_epochs):
            for sub_epoch in range(sub_epochs):
                # E-step
                if lateen == LateenMode.FULL and sub_epoch == sub_epochs - 1:
                    _logger.info('Lateen EM: using Viterbi e-step')
                    expected, cost = self.e_step_hard(maxlen=maxlen)
                else:
                    expected, cost = self.e_step(maxlen=maxlen)
                _logger.info("E-step cost: %s tokens: %s" % (cost, self.cost.all_tokens()))
                if self._supervised:
                    self._update_annotation_choices()
                    self.cost._annot_coding.update_weight()
                    for constr, count in self.cost._annot_coding.constructions.items():
                        expected[constr] += self.cost._annot_coding.weight * count
                # M-step
                self.m_step(
                    expected,
                    expected_freq_threshold=expected_freq_threshold,
                    noexpdigamma=noexpdigamma
                )
            if done:
                break
            # cost-based pruning of lexicon
            cost, done = self.prune_lexicon(prune_criterion, lateen,
                                            maxlen=maxlen, expected_freq_threshold=expected_freq_threshold)
            lc, cc = self.cost.cost_before_tuning()
            _logger.info("Cost after pruning: %s types: %s tokens: %s" %
                (cost, self.cost.types(), self.cost.all_tokens()))
            _logger.info("Unweighted corpus cost: %s lexicon cost: %s" % (cc, lc))
            if done:
                _logger.info('Reached pruning goal')
        return epoch, self.get_cost()

    def _getViterbiBoundaryCost(self) -> float:
        return 0.0

    def _add_compound(self, compound: str, c: int):
        self.cost.update_boundaries(compound, c)
        self._tree[compound] = ConstructionNode(c, c, [])

    def _load_compound(self, dp: DataPoint):
        self._add_compound(dp.compound, dp.count)

    def _ensure_baseline(self):
        raise Exception("Tokeniser is not Morfessor Baseline.")
