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

from ..util.cost import Cost
from ..util.constructions.base import _ConstructionMethods, BaseConstructionMethods
from ..util.corpus import FixedCorpusWeight
from ..util.utils import _progress, tail, logsumexp, categorical
from ..util.exception import SegmentOnlyModelException
from ..util.data import DataPoint

_logger = logging.getLogger(__name__)
EPS = 1e-8


@dataclass
class ConstructionNode:
    rcount: int  # root count (from corpus)
    count: int  # total count of the node
    splitloc: Union[int, Tuple[int,...]]  # Location(s) of the possible splits for virtual constructions; empty tuple or 0 if real construction


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

        Arguments:
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

        # For each construction a ConstrNode is stored.
        #  - All training data has a rcount (real count) > 0.
        #  - All real morphemes have no split locations.
        self._tree: Dict[str,ConstructionNode] = {}

        # Flag to indicate the mode in which the model is operating
        self._segment_only = False

        self._skip_frequent_reanalysis = skip_frequent_reanalysis
        self._analysis_counter = Counter()

        # Semi-supervised data
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

    def load_data(self, data: Iterable[DataPoint]):
        """Load data to initialize the model for batch training.

        Arguments:
            data: iterator of DataPoint tuples

        Adds the compounds in the corpus to the model lexicon. Returns
        the total cost.

        """
        self._assert_not_restricted()
        for dp in data:
            self._load_compound(dp)
        return self.get_cost()

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
            if compound not in self._tree:
                self._add_compound(compound, 1)

            analysis, cost = self._best_analysis(alternatives)
            for m in analysis:
                constructions[m] += self._tree[compound].rcount

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
    def _add_compound(self, compound: str, c: int):
        """Add compound with count c to data."""
        pass

    @abstractmethod
    def _load_compound(self, dp: DataPoint):
        pass

    def _clear_compound_analysis(self, compound: str):  # TODO [Bauwens]: Why is this implementation empty? Are we sure it shouldn't be like the body of self.clear_segmentations()?
        """Clear analysis of a compound from model"""
        pass

    def get_construction_count(self, construction):
        """Return (real) count of the construction."""
        return self.cost.counts.get(construction, 0)

    def _do_skip_analysis(self, construction):
        """Return true if construction should be skipped."""
        if construction in self._analysis_counter:
            if random.random() > 1.0 / max(1,self._analysis_counter[construction]):
                return True
        self._analysis_counter[construction] += 1
        return False

    def get_compounds(self):
        """Return the compound types stored by the model."""
        self._assert_not_restricted()
        return [w
                for w, node in self._tree.items()
                if node.rcount > 0]

    def get_compound_counts(self):
        """Return the compound types stored by the model."""
        self._assert_not_restricted()
        return [(word, node.rcount)
                for word, node in self._tree.items()
                if node.rcount > 0]

    def get_constructions(self):
        """Return a list of the present constructions and their counts."""
        return sorted((word, node.count)
                      for word, node in self._tree.items()
                      if not node.splitloc)

    def get_cost(self) -> float:
        """Return current model encoding cost."""
        return sum(self.cost.cost())

    def get_pseudomodel(self, viterbismooth, viterbimaxlen):
        self._assert_not_restricted()
        for w in sorted(self._tree.keys()):
            node = self._tree[w]
            if node.rcount == 0:
                continue
            constructions, _ = self.viterbi_segment(w, viterbismooth, viterbimaxlen)
            yield (node.rcount, w, constructions)

    def load_annotations(self, annotations: Dict[str,List[List[str]]], annotationweight: float):
        self._annotations = annotations
        self.cost.set_annot_coding_weight(annotationweight)
        self._update_annotation_choices()
        self.cost._annot_coding.update_weight()

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


class MorfessorBaselineSegmenter(ABC):
    @abstractmethod
    def segment(self, model: "MorfessorBaseline", compound: str) -> List[str]:
        pass


class ViterbiSegmenter(MorfessorBaselineSegmenter):

    def __init__(self, addcount: int=0, maxlen: int=30):
        """
        Optimize segmentation of the compound using the Viterbi algorithm.

        Arguments:
            addcount: constant for additive smoothing of Viterbi probs
            maxlen: maximum length for a construction
        """
        self._addcount = addcount
        self._maxlen = maxlen

    def segment(self, model: "MorfessorBaseline", compound: str) -> List[str]:
        if model._skip_frequent_reanalysis and model._do_skip_analysis(compound):
            return model._get_stored_analysis(compound)

        # Use Viterbi algorithm to optimize the subsegments
        constructions = []
        for part in model.cc.splitn(compound, model.cc.force_split_locations(compound)):
            constructions.extend(model.viterbi_segment(part, addcount=self._addcount, maxlen=self._maxlen)[0])
        model._set_compound_analysis(compound, constructions)
        return constructions


class RecursiveSegmenter(MorfessorBaselineSegmenter):
    """
    Optimize segmentation of the compound using recursive splitting.
    """

    def segment(self, model: "MorfessorBaseline", compound: str) -> List[str]:  # TODO: Possibly you want to put this _recursive_split into this class.
        # if self._use_skips and self._test_skip(compound):
        #     return self.segment(compound)
        # Collect forced subsegments

        parts = list(model.cc.splitn(compound, model.cc.force_split_locations(compound)))
        if len(parts) == 1:
            return model._recursive_split(compound)

        model._set_compound_analysis(compound, parts)
        # Use recursive algorithm to optimize the subsegments
        constructions = []
        for part in parts:
            constructions += model._recursive_split(part)
        return constructions


class FlatteningSegmenter(MorfessorBaselineSegmenter):

    def segment(self, model: "MorfessorBaseline", compound: str) -> List[str]:
        segments = model._get_stored_analysis(compound)
        model._clear_compound_analysis(compound)
        model._set_compound_analysis(compound, segments)
        return segments


class MorfessorBaseline(_CommonMorfessorBase):
    """
    Extends the Morfessor base with all methods needed to train Morfessor Baseline.

    Originally, these methods were in the parent class, and prefixed with an assertion that disallowed usage by
    all other subclasses except Morfessor Baseline. This assertion has been removed below, but that does not mean
    that they can be moved back to the parent class.
    """

    # FIXME [Grönroos]: refactor?
    def load_segmentations(self, segmentations: Iterable[Tuple[int,str,List[str]]]):
        for count, compound, constructions in segmentations:
            splitlocs = tuple(self.cc.parts_to_splitlocs(constructions))
            self._add_compound(compound, count)
            self._clear_compound_analysis(compound)
            self._set_compound_analysis(compound, self.cc.splitn(compound, splitlocs))
        return self.get_cost()

    def _add_compound(self, compound: str, c: int):
        self.cost.update_boundaries(compound, c)
        self._modify_construction_count(compound, c)
        self._tree[compound].rcount += c

    def _load_compound(self, dp: DataPoint):
        self._add_compound(dp.compound, dp.count)

        self._clear_compound_analysis(dp.compound)
        self._set_compound_analysis(dp.compound, self.cc.splitn(dp.compound, dp.splitlocs))

    def make_segment_only(self):
        """Reduce the size of this model by removing all non-morphs from the
        analyses. After calling this method it is not possible anymore to call
        any other method that would change the state of the model. Anyway
        doing so would throw an exception.

        """
        #self._num_compounds = len(self.get_compounds())
        self._segment_only = True

        self._tree = {k: v for (k, v) in self._tree.items()
                      if not v.splitloc}

    def _remove(self, construction: str) -> Tuple[int,int]:
        """Remove construction from model."""
        node = self._tree[construction]
        rcount, count = node.rcount, node.count
        self._modify_construction_count(construction, -count)
        return rcount, count

    def clear_segmentations(self):
        for compound in self.get_compounds():
            self._clear_compound_analysis(compound)
            self._set_compound_analysis(compound, [compound])

    def _get_stored_analysis(self, compound: str) -> List[str]:
        """Segment the compound by looking it up in the model analyses.

        Raises KeyError if compound is not present in the training
        data. For segmenting new words, use viterbi_segment(compound).
        """
        _, _, splitloc = self._tree[compound]
        constructions = []
        if splitloc:
            for part in self.cc.splitn(compound, splitloc):
                constructions += self._get_stored_analysis(part)
        else:
            constructions.append(compound)

        return constructions

    def train_batch(self, algorithm: MorfessorBaselineSegmenter=RecursiveSegmenter(),
                    finish_threshold=0.005, max_epochs=None):
        """Train the model in batch fashion.

        The model is trained with the data already loaded into the model (by
        using an existing model or calling one of the load_... methods).

        In each iteration (epoch) all compounds in the training data are
        optimized once, in a random order. If applicable, corpus weight,
        annotation cost, and random split counters are recalculated after
        each iteration.

        :param algorithm: the splitting algorithm used.
        :param finish_threshold: the stopping threshold. Training stops when
                                 the improvement of the last iteration is
                                 smaller then finish_threshold * #boundaries
        :param max_epochs: maximum number of epochs to train
        """
        epochs = 0
        min_epochs = max(1, self._epoch_update(epochs))
        newcost = self.get_cost()
        compounds = list(self.get_compounds())
        _logger.info(f"Compounds in training data: {len(compounds)} types / {self.cost.compound_tokens()} tokens")
        _logger.info("Starting batch training")
        _logger.info("Epochs: %s\tCost: %s" % (epochs, newcost))

        while True:  # Epoch iterator
            random.shuffle(compounds)
            for w in _progress(compounds):
                segments = algorithm.segment(self, w)
                _logger.debug(f"#{w} -> {' + '.join(self.cc.to_string(s) for s in segments)}")
            epochs += 1

            if isinstance(algorithm, FlatteningSegmenter):
                _logger.info("Flattened analysis tree.")
                return epochs, self.get_cost()

            _logger.debug("Cost before epoch update: %s" % self.get_cost())
            min_epochs = max(min_epochs, self._epoch_update(epochs))
            oldcost = newcost
            newcost = self.get_cost()
            lc, cc = self.cost.cost_before_tuning()

            self._epoch_checks()

            _logger.info("Epochs: %s\tCost: %s" % (epochs, newcost))
            _logger.info("Unweighted corpus cost: %s lexicon cost: %s" % (cc, lc))

            # Handle minimal and maximal epochs.
            if min_epochs <= 0:
                if newcost >= oldcost - finish_threshold*self.cost.compound_tokens():
                    break
            else:
                min_epochs -= 1

            if max_epochs is not None:
                if epochs >= max_epochs:
                    _logger.info("Max number of epochs reached, stop training")
                    break

        _logger.info("Done.")
        return epochs, newcost

    def train_online(self, data: Iterable[DataPoint], count_modifier=None, epoch_interval: int=10000,
                     algorithm: MorfessorBaselineSegmenter=RecursiveSegmenter(),
                     init_rand_split=None, max_epochs: int=None):
        """Train the model in online fashion.

        The model is trained with the data provided in the data argument.
        As example the data could come from a generator linked to standard in
        for live monitoring of the splitting.

        All compounds from data are only optimized once. After online
        training, batch training could be used for further optimization.

        Epochs are defined as a fixed number of compounds. After each epoch (
        like in batch training), the annotation cost, and random split counters
        are recalculated if applicable.

        Arguments:
            data: iterator of DataPoints. Every occurrence of the compound is taken with count 1
                    FIXME [Bauwens]: That's not true.
            count_modifier: function for adjusting the counts of each compound
            epoch_interval: number of compounds to process before starting a new epoch
            algorithm: the splitting algorithm used.
            init_rand_split: probability for random splitting a compound to
                               at any point for initializing the model. None
                               or 0 means no random splitting.
            max_epochs: maximum number of epochs to train
        """
        if isinstance(algorithm, FlatteningSegmenter):
            raise ValueError("Cannot use flattening during online training.")
        if count_modifier is not None:
            counts = {}

        epochs = 0
        i = 0
        more_tokens = True
        data = iter(data)

        _logger.info("Starting online training")
        while more_tokens:
            self._epoch_update(epochs)
            newcost = self.get_cost()
            _logger.info("Tokens processed: %s\tCost: %s" % (i, newcost))

            for _ in _progress(range(epoch_interval)):
                try:
                    dp = next(data)
                except StopIteration:
                    more_tokens = False
                    break

                self._add_compound(dp.compound, dp.count)
                self._clear_compound_analysis(dp.compound)
                self._set_compound_analysis(dp.compound, self.cc.splitn(dp.compound, dp.splitlocs))

                segments = algorithm.segment(self, dp.compound)
                _logger.debug(f"#{i}: {dp.compound} -> {' + '.join(self.cc.to_string(s) for s in segments)}")
                i += 1

            epochs += 1
            if max_epochs is not None and epochs >= max_epochs:
                _logger.info("Max number of epochs reached, stop training")
                break

        self._epoch_update(epochs)
        newcost = self.get_cost()
        _logger.info("Tokens processed: %s\tCost: %s" % (i, newcost))
        return epochs, newcost

    def _epoch_checks(self):
        """Apply per epoch checks"""
        # self._check_integrity()  # No longer exists...
        pass

    def _epoch_update(self, epoch_num: int) -> int:
        """Do model updates that are necessary between training epochs.

        The argument is the number of training epochs finished.

        In practice, this does two things:
        - If random skipping is in use, reset construction counters.
        - If semi-supervised learning is in use and there are alternative
          analyses in the annotated data, select the annotations that are
          most likely given the model parameters. If not hand-set, update
          the weight of the annotated corpus.

        This method should also be run prior to training (with the
        epoch number argument as 0).

        """
        forced_epochs = 0
        if self._corpus_weight_updater is not None:
            if self._corpus_weight_updater.update(self, epoch_num):
                forced_epochs += 2

        self._analysis_counter = Counter()
        if self._is_semisupervised():
            self._update_annotation_choices()
            self.cost._annot_coding.update_weight()

        return forced_epochs

    def _set_compound_analysis(self, compound: str, parts):
        """Set analysis of compound to according to given segmentation.

        Arguments:
            compound: compound to split
            parts: desired constructions of the compound

        """
        parts = list(parts)
        if len(parts) == 1:
            rcount, count = self._remove(compound)
            self._tree[compound] = ConstructionNode(rcount, 0, tuple())
            self._modify_construction_count(compound, count)
        else:
            rcount, count = self._remove(compound)

            splitloc = tuple(self.cc.parts_to_splitlocs(parts))
            self._tree[compound] = ConstructionNode(rcount, count, splitloc)
            for constr in parts:
                self._modify_construction_count(constr, count)

    def _modify_construction_count(self, construction: str, dcount: int):
        """Modify the count of construction by dcount.

        For virtual constructions, recurses to child nodes in the
        tree. For real constructions, adds/removes construction
        to/from the lexicon whenever necessary.

        """
        if dcount == 0 or construction is None:
            return
        if construction in self._tree:
            node = self._tree[construction]
            rcount, count, splitloc = node.rcount, node.count, node.splitloc
        else:
            rcount, count, splitloc = 0, 0, None
        newcount = count + dcount
        # observe that this comparison will not work correctly if counts
        # are floats rather than ints
        if newcount == 0:
            if construction in self._tree:
                del self._tree[construction]
        else:
            self._tree[construction] = ConstructionNode(rcount, newcount, splitloc)
        if splitloc:
            # Virtual construction
            for child in self.cc.splitn(construction, splitloc):
                self._modify_construction_count(child, dcount)
        else:
            self.cost.update(construction, newcount - count)  # Real construction

    def get_segmentations(self):
        """Retrieve segmentations for all compounds encoded by the model."""
        for w in sorted(self._tree.keys()):
            c = self._tree[w].rcount
            if c > 0:
                yield c, w, self._get_stored_analysis(w)

    def _recursive_split(self, construction: str):
        """Optimize segmentation of the construction by recursive splitting.

        Returns list of segments.

        """
        # if self._use_skips and self._test_skip(construction):
        #     return self.segment(construction)
        rcount, count = self._remove(construction)

        # Check all binary splits and no split
        self._modify_construction_count(construction, count)
        mincost = self.get_cost()
        self._modify_construction_count(construction, -count)

        best_splitloc = None

        for loc in self.cc.split_locations(construction):
            prefix, suffix = self.cc.split(construction, loc)
            self._modify_construction_count(prefix, count)
            self._modify_construction_count(suffix, count)
            cost = self.get_cost()
            self._modify_construction_count(prefix, -count)
            self._modify_construction_count(suffix, -count)
            if cost <= mincost:
                mincost = cost
                best_splitloc = loc

        if best_splitloc:
            # Virtual construction
            self._tree[construction] = ConstructionNode(rcount, count, best_splitloc)
            prefix, suffix = self.cc.split(construction, best_splitloc)
            self._modify_construction_count(prefix, count)
            self._modify_construction_count(suffix, count)
            lp = self._recursive_split(prefix)
            if suffix != prefix:
                return lp + self._recursive_split(suffix)
            else:
                return lp + lp
        else:
            # Real construction
            self._tree[construction] = ConstructionNode(rcount, 0, None)
            self._modify_construction_count(construction, count)
            return [construction]
