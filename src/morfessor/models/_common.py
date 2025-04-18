from abc import abstractmethod, ABC
from typing import Iterable, List, Dict, Union, Tuple
from collections import Counter
from dataclasses import dataclass

import heapq
import itertools
import logging
import math
import numbers
import random

from morfessor.loss.cost import Cost
from ..util.constructions.base import _ConstructionMethods, BaseConstructionMethods
from morfessor.loss.corpus import FixedCorpusWeight
from ..util.misc import tail, logsumexp, categorical
from ..util.exception import SegmentOnlyModelException
from ..util.data.counting_data import DataPoint

_logger = logging.getLogger(__name__)
EPS = 1e-8


@dataclass
class SimpleConstrNode:
    count: int
    splitloc: Union[int, Tuple[int,...]]


class _CommonMorfessorBase(ABC):
    """
    Methods needed by all Morfessor tokenisers.

    The model is complete agnostic to whether it is used with lists of strings (finding
    phrases in sentences) or strings of characters (finding morphs in words).
    """

    penalty = -9999.9

    def __init__(
        self,
        corpusweight=None,
        skip_frequent_reanalysis: bool=False,
        constr_methods: _ConstructionMethods=BaseConstructionMethods(None,None)
    ):
        """Initialize a new model instance.

        :param corpusweight: weight for the corpus cost
        :param skip_frequent_reanalysis: randomly skip frequently occurring constructions to speed up training.
                According to the Morfessor 2.0 paper:
                >    As frequent compounds are encountered many times in
                >    running text, Morfessor 2.0 includes an option for
                >    randomly skipping compounds and constructions
                >    that have been recently analyzed.
        :param constr_methods: Object that knows how to handle your particular construction type. For example,
                               you could just have strings and then splitting them is easy. But you could also
                               have string tuples as constructions, and that requires special splitting methods.
        """
        self.cc = constr_methods

        # Flag to indicate the mode in which the model is operating
        self._segment_only = False

        self._skip_frequent_reanalysis = skip_frequent_reanalysis
        self._analysis_counter = Counter()

        # Semi-supervised data... stored inside the model...
        self._annotations: Dict[str,List[List[str]]] = dict()

        # Cost variables
        self.cost = Cost(self.cc, corpusweight)  # This field is overridden in EM.

        # Set corpus weight updater
        self._corpus_weight_updater = None
        self.set_corpus_weight_updater(corpusweight)

    def set_corpus_weight_updater(self, corpus_weight):
        if corpus_weight is None:
            self._corpus_weight_updater = FixedCorpusWeight(1.0)
        elif isinstance(corpus_weight, numbers.Number):
            self._corpus_weight_updater = FixedCorpusWeight(corpus_weight)
        else:
            self._corpus_weight_updater = corpus_weight

    ######### This is all data-loading-related and should probably be removed eventually. Makes little sense to store raw data inside any model ever...

    def load_data(self, data: Iterable[DataPoint]):
        """
        Loads a corpus of word-count pairs into the model.
        Returns the total cost.
        """
        self._assert_not_restricted()
        for dp in data:
            self._load_compound(dp)
        return self.get_cost()

    @abstractmethod
    def _load_compound(self, dp: DataPoint):
        pass

    def load_annotations(self, annotations: Dict[str,List[List[str]]], annotationweight: float):
        self._annotations = annotations
        self.cost.set_annot_coding_weight(annotationweight)
        self._update_annotation_choices()
        self.cost._annot_coding.update_weight()

    @abstractmethod
    def _get_corpus_frequency(self, compound: str) -> int:
        pass

    @abstractmethod
    def get_compounds(self) -> Iterable[str]:
        """Recall the compound types (i.e. words) the user loaded into the model."""
        pass

    @abstractmethod
    def get_compound_counts(self) -> Iterable[Tuple[str,int]]:
        """Recall the compound types (i.e. words) and frequencies the user loaded into the model."""
        pass

    #######################################################

    @property
    def tokens(self):
        """Return the number of construction tokens."""
        return self.cost.tokens()

    @property
    def types(self):
        """Return the number of construction types."""
        return self.cost.types() - 1  # do not include boundary

    def _assert_not_restricted(self):
        if self._segment_only:
            raise SegmentOnlyModelException()

    def _is_semisupervised(self) -> bool:
        return len(self._annotations) > 0

    def _update_annotation_choices(self):
        """Update the selection of alternative analyses in annotations.

        For semi-supervised models, select the most likely alternative
        analyses included in the annotations of the compounds.
        """
        if not self._is_semisupervised():
            return

        # Collect constructions from the most probable segmentations
        # and add missing compounds also to the unannotated data
        constructions = Counter()
        for compound, alternatives in self._annotations.items():
            if not self._seen_compound(compound):
                self._add_compound(compound, 1)

            analysis, cost = self._best_analysis(alternatives)
            for m in analysis:
                constructions[m] += self._get_corpus_frequency(compound)

        # Apply the selected constructions in annotated corpus coding
        self.cost.set_annot_constructions(constructions)
        for constr in constructions.keys():
            count = self.get_construction_count(constr)
            self.cost.set_annot_observed(constr, count)

    def _best_analysis(self, choices: Iterable[List[str]]):
        """Select the best analysis out of the given choices."""
        bestcost = None
        bestanalysis = None
        for analysis in choices:
            cost = 0.0
            for constr in analysis:
                count = self.get_construction_count(constr)
                if count > 0:
                    cost += math.log(self.cost.tokens()) - math.log(count)
                else:
                    cost -= self.penalty  # penalty is negative
            if bestcost is None or cost < bestcost:
                bestcost = cost
                bestanalysis = analysis
        return bestanalysis, bestcost

    @abstractmethod
    def _add_compound(self, compound: str, count: int):
        """Add compound with count c to data."""
        pass

    @abstractmethod
    def _seen_compound(self, compound: str) -> bool:
        pass

    def _clear_compound_analysis(self, compound: str):  # TODO [Bauwens]: Why is this implementation empty? Are we sure it shouldn't be like the body of self.clear_segmentations()?
        """Clear analysis of a compound from model"""
        pass

    def get_construction_count(self, construction):
        """Return (real) count of the construction."""
        return self.cost.counts.get(construction, 0)

    def get_cost(self) -> float:
        """Return current model encoding cost."""
        return sum(self.cost.cost())

    def _do_skip_analysis(self, construction: str) -> bool:
        """Return true if construction should be skipped."""
        if construction in self._analysis_counter:
            if random.random() > 1.0 / max(1,self._analysis_counter[construction]):
                return True
        self._analysis_counter[construction] += 1
        return False

    def get_pseudomodel(self, viterbismooth, viterbimaxlen):  # TODO [Bauwens]: Is this relevant to have for Morfessor Baseline? Does the tree consist of (recursive) Viterbi splits, or not?
        """
        Use the trained model to segment the training data.
        The resulting segmentations can be interpreted as if they are a Morfessor Baseline model.
        """
        self._assert_not_restricted()
        for word, rcount in sorted(self.get_compound_counts()):
            constructions, _ = self.viterbi_segment(word, viterbismooth, viterbimaxlen)
            yield rcount, word, constructions

    def _getViterbiBoundaryCost(self) -> float:
        return math.log(self.cost.tokens() + self.cost.compound_tokens()) \
                - math.log(self.cost.compound_tokens())

    def viterbi_segment(self, compound, addcount=1.0, maxlen=30,
                        allow_longer_unk_splits=False,
                        taboo=None):
        """Find optimal segmentation using the Viterbi algorithm.

        Arguments:
          compound: compound to be segmented
          addcount: constant for additive smoothing (0 = no smoothing)
          maxlen: maximum length for the constructions
          taboo: not allowed to use these constructions

        If additive smoothing is applied, new complex construction types can
        be selected during the search. Without smoothing, only new
        single-atom constructions can be selected.

        Returns the most probable segmentation and its log-probability.

        """
        #clen = len(compound)
        # indices = range(1, clen+1) if allowed_boundaries is None \
        #           else allowed_boundaries+[clen]

        grid = {None: (0.0, None)}
        tokens = self.cost.all_tokens() + addcount
        logtokens = math.log(tokens) if tokens > 0 else 0
        taboo = set() if taboo is None else set(taboo)

        newboundcost = self.cost.newbound_cost(addcount) if addcount > 0 else 0

        badlikelihood = self.cost.bad_likelihood(compound,addcount)

        for t in itertools.chain(self.cc.split_locations(compound), [None]):
            # Select the best path to current node.
            # Note that we can come from any node in history.
            bestpath = None
            bestcost = None

            for pt in tail(maxlen, itertools.chain([None], self.cc.split_locations(compound, stop=t))):
                if grid[pt][0] is None:
                    continue
                cost = grid[pt][0]
                construction = self.cc.slice(compound, pt, t)
                if construction in taboo:
                    continue
                count = self.get_construction_count(construction)
                if count > 0:
                    cost += (logtokens - math.log(count + addcount))
                elif addcount > 0:
                    if self.cost.tokens() == 0:
                        cost += (addcount * math.log(addcount) +
                                newboundcost + self.cost.get_coding_cost(construction))
                    else:
                        cost += (logtokens - math.log(addcount) +
                                newboundcost + self.cost.get_coding_cost(construction))

                elif self.cc.is_atom(construction):
                    cost += badlikelihood
                elif allow_longer_unk_splits:
                    # Some splits are forbidden, so longer unknown
                    # constructions have to be allowed
                    cost += len(self.cc.corpus_key(construction)) * badlikelihood
                else:
                    continue
                #_logger.debug("cost(%s)=%.2f", construction, cost)
                if bestcost is None or cost < bestcost:
                    bestcost = cost
                    bestpath = pt
            grid[t] = (bestcost, bestpath)

        splitlocs = []

        cost, path = grid[None]
        while path is not None:
            splitlocs.append(path)
            path = grid[path][1]

        constructions = list(self.cc.splitn(compound, list(reversed(splitlocs))))
        cost += self._getViterbiBoundaryCost()
        return constructions, cost

    def sample_segment(self, compound: str, theta: float=0.5, maxlen: int=30, taboo: Iterable[str]=None):
        """Sample a segmentation using the
        Forward-filter Backward-sample algorithm.

        Arguments:
          compound: compound to be segmented
          theta: sampling temperature. (1.0 = unsmoothed).
          maxlen: maximum length for the constructions
          taboo: not allowed to use these constructions

        Returns the sampled segmentation and its log-probability.

        """
        grid = {'start': (0.0, None)}
        tokens = self.cost.all_tokens()
        logtokens = math.log(tokens) if tokens > 0 else 0
        taboo = set() if taboo is None else set(taboo)

        badlikelihood = self.cost.bad_likelihood(compound, 0)
        extrabad = badlikelihood**2

        if len(compound) == 1:
            return [compound], 0

        ## Forward filtering pass
        for t in itertools.chain(self.cc.split_locations(compound), ['stop']):
            # logsum of all paths to current node.
            # Note that we can come from any node in history.
            negcosts = []

            for pt in tail(maxlen, itertools.chain(['start'], self.cc.split_locations(compound, stop=t))):
                if grid[pt][0] is None:
                    continue
                cost = grid[pt][0]
                construction = self.cc.slice(compound, pt, t)
                if construction in taboo:
                    continue
                count = self.get_construction_count(construction)
                if count > 0:
                    cost += (logtokens - theta * math.log(count))
                elif self.cc.is_atom(construction):
                    cost += badlikelihood
                else:
                    continue
                #_logger.debug("cost(%s)=%.2f", construction, cost)
                negcosts.append(-cost)
            if len(negcosts) == 0:
                grid[t] = (extrabad, None)
                continue
            # to compute sum (superposition) of path probabilities
            # in log space, use logsumexp
            totcost = -logsumexp(negcosts)
            grid[t] = (totcost, None)

        ## Backward sampling pass
        splitlocs = []
        t = 'stop'
        totcost = grid['stop'][0]
        path_cost = 0
        while t is not None:
            pts = []
            probs = []
            local_costs = []
            nt = None if t == 'stop' else t
            for pt in tail(maxlen,
                           itertools.chain(['start'], self.cc.split_locations(compound, stop=nt))):
                if grid[pt][0] is None:
                    continue
                cost = grid[pt][0]
                # cc.slice requires None for endpoints
                pt = None if pt == 'start' else pt
                construction = self.cc.slice(compound, pt, nt)
                if construction in taboo:
                    continue
                count = self.get_construction_count(construction)
                if count > 0:
                    #cost += theta * (logtokens - math.log(count) - totcost)
                    local_cost = logtokens - (theta * math.log(count))
                    cost += local_cost - totcost
                elif self.cc.is_atom(construction):
                    cost += badlikelihood
                    local_cost = badlikelihood
                else:
                    continue
                local_costs.append(local_cost)
                pts.append(pt)
                if cost < 0:
                    # FIXME: bug or imprecision?
                    probs.append(1)
                else:
                    probs.append(math.exp(-cost))
            if sum(probs) < EPS:
                # if nothing is valid, letterize
                if t == 'stop':
                    t = len(compound)
                sample = t - 1
                if sample <= 0:
                    sample = None
                path_cost += badlikelihood
            else:
                idx, sample = categorical(pts, probs)
                path_cost += local_costs[idx]
            if sample is not None:
                splitlocs.append(sample)
            t = sample

        constructions = list(self.cc.splitn(compound, list(reversed(splitlocs))))

        return constructions, path_cost

    #TODO project lambda
    def forward_logprob(self, compound: str):
        """Find log-probability of a compound using the forward algorithm.

        Arguments:
          compound: compound to process

        Returns the (negative) log-probability of the compound. If the
        probability is zero, returns a number that is larger than the
        value defined by the penalty attribute of the model object.

        """
        clen = len(compound)
        grid = [0.0]
        if self.cost._corpus_coding.tokens + self.cost._corpus_coding.boundaries > 0:
            logtokens = math.log(self.cost._corpus_coding.tokens +
                                 self.cost._corpus_coding.boundaries)
        else:
            logtokens = 0

        # Forward main loop
        for t in range(1, clen + 1):
            # Sum probabilities from all paths to the current node.
            # Note that we can come from any node in history.
            psum = 0.0
            for pt in range(0, t):
                cost = grid[pt]
                construction = compound[pt:t]
                count = self.get_construction_count(construction)
                if count > 0:
                    cost += (logtokens - math.log(count))
                else:
                    continue
                psum += math.exp(-cost)
            if psum > 0:
                grid.append(-math.log(psum))
            else:
                grid.append(-self.penalty)
        cost = grid[-1]

        cost += self._getViterbiBoundaryCost()
        return cost

    def viterbi_nbest(self, compound: str, n: int, addcount: float=0.0, theta: float=1.0, maxlen: int=30,
                      allow_longer_unk_splits: bool=False):
        """Find top-n optimal segmentations using the Viterbi algorithm.

        Arguments:
          compound: compound to be segmented
          n: how many segmentations to return
          addcount: constant for additive smoothing (0 = no smoothing)
          theta: sampling temperature. (1.0 = Viterbi).
          maxlen: maximum length for the constructions

        If additive smoothing is applied, new complex construction types can
        be selected during the search. Without smoothing, only new
        single-atom constructions can be selected.

        Returns the n most probable segmentations and their
        log-probabilities.

        """
        grid = {None: [(0.0, None)]}
        tokens = self.cost.all_tokens() + addcount
        logtokens = math.log(tokens) if tokens > 0 else 0

        newboundcost = self.cost.newbound_cost(addcount) if addcount > 0 else 0

        badlikelihood = self.cost.bad_likelihood(compound,addcount)

        # Viterbi main loop
        for t in itertools.chain(self.cc.split_locations(compound), [None]):
            # Select the best path to current node.
            # Note that we can come from any node in history.
            bestn = []
            for pt in tail(maxlen, itertools.chain([None], self.cc.split_locations(compound, stop=t))):
                for k in range(len(grid[pt])):
                    if grid[pt][k][0] is None:
                        continue
                    cost = -grid[pt][k][0]
                    construction = self.cc.slice(compound, pt, t)
                    count = self.get_construction_count(construction)
                    if count > 0:
                        cost += (logtokens - theta * math.log(count + addcount))
                    elif addcount > 0:
                        if self.cost.tokens() == 0:
                            cost += addcount * math.log(addcount) + newboundcost + self.cost.get_coding_cost(construction)
                        else:
                            cost += logtokens - math.log(addcount) + newboundcost + self.cost.get_coding_cost(construction)

                    elif self.cc.is_atom(construction):
                        cost += badlikelihood
                    elif allow_longer_unk_splits:
                        # Some splits are forbidden, so longer unknown
                        # constructions have to be allowed
                        cost += len(self.cc.corpus_key(construction)) * badlikelihood
                    else:
                        continue
                    if len(bestn) < n:
                        heapq.heappush(bestn, (-cost, pt, k))
                    else:
                        heapq.heappushpop(bestn, (-cost, pt, k))
            grid[t] = bestn
        results = []
        for k in range(len(grid[None])):
            constructions = []
            cost, path, ki = grid[None][k]
            cost = -cost
            lt = None
            if path is None:
                constructions = [compound]
            else:
                while True:
                    t = path
                    constructions.append(self.cc.slice(compound, t, lt))
                    path = grid[t][ki][1]
                    ki = grid[t][ki][2]
                    lt = t
                    if lt is None:
                        break
            constructions.reverse()
            cost += self._getViterbiBoundaryCost()
            results.append((cost, constructions))
        if len(results) == 0:
            results = [(badlikelihood, [compound])]
        return [(constr, cost) for cost, constr in sorted(results)]

    def get_corpus_coding_weight(self):
        return self.cost._corpus_coding.weight

    def set_corpus_coding_weight(self, weight: float):
        self._assert_not_restricted()
        self.cost.set_corpus_coding_weight(weight)

    def get_params(self) -> dict:
        """Returns a dict of hyperparameters."""
        params = {'corpusweight': self.get_corpus_coding_weight()}
        if self._is_semisupervised():
            params['annotationweight'] = self.cost._annot_coding.weight
        if isinstance(self.cc, BaseConstructionMethods):
            params['forcesplit'] = ''.join(sorted(self.cc._force_splits))
            if self.cc._nosplit:
                params['nosplit'] = self.cc._nosplit.pattern
        return params
