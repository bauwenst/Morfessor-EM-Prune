from abc import ABC, abstractmethod
from typing import List, Iterable, Tuple, Dict, Union, Iterator
from collections import Counter
from dataclasses import dataclass

import random
import logging

from ._common import _CommonMorfessorBase, DataPoint
from ..util.utils import _progress

_logger = logging.getLogger(__name__)


@dataclass
class ConstructionNode:
    rcount: int  # root count (from corpus); [Bauwens] From what I gather, the difference between the two counts is that `count` can go up or down depending on the structure of the Morfessor tree, whilst `rcount` is the "observed" count of the type in real text.
    count: int  # total count of the node
    splitloc: Union[int, Tuple[int,...]]  # Location(s) of the possible splits for virtual constructions; empty tuple or 0 if real construction


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
            return model.tree._get_stored_analysis(compound)

        pretokens = model.cc.splitn(compound, model.cc.force_split_locations(compound))
        constructions = []
        for pretoken in pretokens:
            constructions.extend(model.viterbi_segment(pretoken, addcount=self._addcount, maxlen=self._maxlen)[0])

        # Store the resulting segmentation.
        model.tree._set_compound_analysis(compound, constructions)
        return constructions


class RecursiveSegmenter(MorfessorBaselineSegmenter):
    """
    Optimize segmentation of the compound using recursive splitting.
    """

    def segment(self, model: "MorfessorBaseline", compound: str) -> List[str]:  # TODO: Possibly you want to put this _recursive_split into this class.
        # if model._skip_frequent_reanalysis and model._do_skip_analysis(compound):
        #     return model.tree._get_stored_analysis(compound)

        # Collect forced subsegments (a.k.a. pretokens)
        pretokens = list(model.cc.splitn(compound, model.cc.force_split_locations(compound)))
        if len(pretokens) > 1:
            model.tree._set_compound_analysis(compound, pretokens)

        # For each pretoken, apply _recursive_split.
        constructions = []
        for pretoken in pretokens:
            constructions += model.tree._recursive_split(pretoken)
        return constructions


class FlatteningSegmenter(MorfessorBaselineSegmenter):

    def segment(self, model: "MorfessorBaseline", compound: str) -> List[str]:
        segments = model.tree._get_stored_analysis(compound)
        model._clear_compound_analysis(compound)
        model.tree._set_compound_analysis(compound, segments)
        return segments


class MorfessorTree:

    def __init__(self, model: "MorfessorBaseline"):
        # For each construction a ConstrNode is stored.
        #  - All training data has a rcount (real count) > 0.
        #  - All real morphemes have no split locations.
        self._segmentation_tree: Dict[str, ConstructionNode] = {}
        self._model = model  # TODO: Ideally there wouldn't be a bidirectional association, but there is right now because we need access to model.cost 3 times.

    def _add(self, compound: str, count: int):
        self._segmentation_tree[compound].rcount += count

    def _remove(self, construction: str) -> Tuple[int,int]:
        """Remove construction from model."""
        node = self._segmentation_tree[construction]
        rcount, count = node.rcount, node.count
        self._modify_construction_count(construction, -count)
        return rcount, count

    def _set_compound_analysis(self, compound: str, parts):
        """Set analysis of compound to according to given segmentation.

        Arguments:
            compound: compound to split
            parts: desired constructions of the compound

        """
        parts = list(parts)
        if len(parts) == 1:
            rcount, count = self._remove(compound)
            self._segmentation_tree[compound] = ConstructionNode(rcount, 0, tuple())
            self._modify_construction_count(compound, count)
        else:
            rcount, count = self._remove(compound)

            splitloc = tuple(self._model.cc.parts_to_splitlocs(parts))
            self._segmentation_tree[compound] = ConstructionNode(rcount, count, splitloc)
            for constr in parts:
                self._modify_construction_count(constr, count)

    def _modify_construction_count(self, construction: str, delta_count: int):
        """Modify the count of construction by dcount.

        For virtual constructions, recurses to child nodes in the
        tree. For real constructions, adds/removes construction
        to/from the lexicon whenever necessary.
        """
        if delta_count == 0 or construction is None:
            return

        if construction in self._segmentation_tree:
            node = self._segmentation_tree[construction]
            rcount, count, splitloc = node.rcount, node.count, node.splitloc
        else:
            rcount, count, splitloc = 0, 0, None

        count += delta_count
        if count == 0:  # Note: this comparison may not work correctly if counts are floats rather than ints.
            if construction in self._segmentation_tree:
                self._segmentation_tree.pop(construction)
        else:
            self._segmentation_tree[construction] = ConstructionNode(rcount, count, splitloc)

        if splitloc:  # => Virtual construction
            for child in self._model.cc.splitn(construction, splitloc):
                self._modify_construction_count(child, delta_count)
        else:
            self._model.cost.update(construction, delta_count)  # Real construction

    def _recursive_split(self, construction: str):
        """
        Optimize segmentation of the construction by recursive splitting.
        Returns list of segments.
        
        This is the algorithm described on page 15 of this paper: https://users.ics.aalto.fi/mcreutz/papers/Creutz05tr.pdf
        Refactored into symbols in algorithm 2.4 of this thesis: https://bauwenst.github.io/cdn/doc/pdf/2023/masterthesis.pdf
        """
        # if self._use_skips and self._test_skip(construction):
        #     return self.segment(construction)
        rcount, count = self._remove(construction)

        # Check all binary splits and no split
        self._modify_construction_count(construction, count)
        mincost = self._model.get_cost()
        self._modify_construction_count(construction, -count)

        best_splitloc = None

        for loc in self._model.cc.split_locations(construction):
            prefix, suffix = self._model.cc.split(construction, loc)
            self._modify_construction_count(prefix, count)
            self._modify_construction_count(suffix, count)
            cost = self._model.get_cost()
            self._modify_construction_count(prefix, -count)
            self._modify_construction_count(suffix, -count)
            if cost <= mincost:
                mincost = cost
                best_splitloc = loc

        if best_splitloc:  # => Virtual construction
            self._segmentation_tree[construction] = ConstructionNode(rcount, count, best_splitloc)
            prefix, suffix = self._model.cc.split(construction, best_splitloc)
            self._modify_construction_count(prefix, count)
            self._modify_construction_count(suffix, count)
            lp = self._recursive_split(prefix)
            if suffix != prefix:
                return lp + self._recursive_split(suffix)
            else:
                return lp + lp
        else:  # => Real construction
            self._segmentation_tree[construction] = ConstructionNode(rcount, 0, None)
            self._modify_construction_count(construction, count)
            return [construction]

    def _get_stored_analysis(self, compound: str) -> List[str]:
        """Segment the compound by looking it up in the model analyses.

        Raises KeyError if compound is not present in the training
        data. For segmenting new words, use viterbi_segment(compound).
        """
        splitloc = self._segmentation_tree[compound].splitloc
        constructions = []
        if splitloc:
            for part in self._model.cc.splitn(compound, splitloc):
                constructions += self._get_stored_analysis(part)
        else:
            constructions.append(compound)

        return constructions

    def get_segmentations(self) -> Iterator[Tuple[int,str,List[str]]]:
        """Retrieve segmentations for all compounds encoded by the model.

           [Bauwens] To clarify: the Morfessor tree stores both fictitious types and types that were observed in a
           corpus. They are distinguished by having an "rcount" (count in a real corpus) versus not having one. This
           particular method is basically asking "please tokenise the words that the corpus put into the model"."""
        for w in sorted(self._segmentation_tree.keys()):
            c = self._segmentation_tree[w].rcount
            if c > 0:
                yield c, w, self._get_stored_analysis(w)

    def get_leaves(self) -> List[Tuple[str,int]]:
        return [(word, node.count)
                for word, node in self._segmentation_tree.items()
                if not node.splitloc]  # If the node is not split, it's a leaf, i.e. a "non-virtual construction".

    def freeze_leaves(self):
        """
        Reduce the size of this model by removing all non-morphs from the
        analyses. After calling this method it is not possible anymore to call
        any other method that would change the state of the model. Anyway
        doing so would throw an exception.
        """
        self._segmentation_tree = {k: v for (k, v) in self._segmentation_tree.items()
                                   if not v.splitloc}


class MorfessorBaseline(_CommonMorfessorBase):
    """
    Extends the Morfessor base with all methods needed to train Morfessor Baseline.

    Originally, these methods were in the parent class, and prefixed with an assertion that disallowed usage by
    all other subclasses except Morfessor Baseline. This assertion has been removed below, but that does not mean
    that they can be moved back to the parent class.
    """

    def __init__(self, corpusweight, skip_frequent_reanalysis, constr_methods):
        super().__init__(corpusweight=corpusweight, skip_frequent_reanalysis=skip_frequent_reanalysis, constr_methods=constr_methods)
        self.tree = MorfessorTree(self)

    # FIXME [Grönroos]: refactor?
    def load_segmentations(self, segmentations: Iterable[Tuple[int,str,List[str]]]):
        for count, compound, constructions in segmentations:
            splitlocs = tuple(self.cc.parts_to_splitlocs(constructions))
            self._add_compound(compound, count)
            self._clear_compound_analysis(compound)
            self.tree._set_compound_analysis(compound, self.cc.splitn(compound, splitlocs))
        return self.get_cost()

    def _add_compound(self, compound: str, count: int):
        self.cost.update_boundaries(compound, count)
        self.tree._modify_construction_count(compound, count)
        self.tree._add(compound, count)

    def _load_compound(self, dp: DataPoint):
        self._add_compound(dp.compound, dp.count)

        self._clear_compound_analysis(dp.compound)
        self.tree._set_compound_analysis(dp.compound, self.cc.splitn(dp.compound, dp.splitlocs))

    def _seen_compound(self, compound: str) -> bool:
        return compound in self.tree._segmentation_tree

    def _get_corpus_frequency(self, compound: str) -> int:
        return self.tree._segmentation_tree[compound].rcount

    def make_segment_only(self):
        self._segment_only = True
        self.tree.freeze_leaves()

    def get_constructions(self):
        """Return a list of the non-virtual constructions and their counts."""
        return self.tree.get_leaves()

    def get_compounds(self):
        self._assert_not_restricted()
        return [word
                for word, node in self.tree._segmentation_tree.items()
                if node.rcount > 0]  # Note that having a raw corpus count is not the same as being virtual/non-virtual. The nodes in the tree can or cannot appear in the corpus. Here we ask for the ones that do.

    def get_compound_counts(self):
        self._assert_not_restricted()
        return [(word, node.rcount)
                for word, node in self.tree._segmentation_tree.items()
                if node.rcount > 0]

    def clear_segmentations(self):
        for compound in self.get_compounds():
            self._clear_compound_analysis(compound)
            self.tree._set_compound_analysis(compound, [compound])

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
                self.tree._set_compound_analysis(dp.compound, self.cc.splitn(dp.compound, dp.splitlocs))

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
