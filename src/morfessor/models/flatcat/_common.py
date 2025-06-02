from __future__ import unicode_literals

import math
import re

from src.morfessor.models.flatcat import WORD_BOUNDARY, CategorizedMorph, MorphUsageProperties, utils
from src.morfessor.models.flatcat.categorizationscheme import get_categories, DEFAULT_CATEGORY
from src.morfessor.models.flatcat.flatcat import AnalysisAlternative, ViterbiNode, _wb_wrap, CostBreakdown, \
    SortedAnalysis
from src.morfessor.models.flatcat.utils import _is_string, LOGPROB_ZERO


class AbstractSegmenter(object):
    def __init__(self, corpus_coding, nosplit=None, postprocessing=None):
        self._initialized = False
        # None (= no corpus), "untagged", "partial", "full"
        self._corpus_tagging_level = None
        self._corpus_coding = corpus_coding
        # Do not allow splitting between a letter pair matching this regex
        if nosplit is None:
            self.nosplit_re = None
        elif _is_string(nosplit):
            self.nosplit_re = re.compile(nosplit, re.UNICODE)
        else:
            self.nosplit_re = nosplit
        self.annotations = None
        self._annotations_tagged = None
        self.postprocessing = postprocessing \
            if postprocessing is not None else []

    def initialize_hmm(self, min_difference_proportion=None):
        pass

    def viterbi_segment(self, segments, addcount=None, maxlen=None):
        """Compatibility with Morfessor Baseline.
        Postprocessing heuristics are applied to modify the segmentation.

        Note that the addcount and maxlen arguments are silently ignored.
        """
        analysis, logp = self.viterbi_analyze(segments)
        for processor in self.postprocessing:
            analysis = processor.apply_to(analysis, self)
        return (self.detag_word(analysis), logp)

    def viterbi_analyze(self, segments, strict_annot=True):
        """Simultaneously segment and tag a word using the learned model.
        Can be used to segment unseen words.

        Arguments:
            segments :  A word (or a list of morphs which will be
                        concatenated into a word) to resegment and tag.
            strict_annot :  If the word occurs in the annotated corpus,
                            only consider the segmentations in the annotation.
        Returns:
            best_analysis, :  The resegmented, retagged word
            best_cost      :  The cost of the returned solution
        """

        msg = 'Must initialize model and tag corpus before segmenting'
        assert (self._initialized and
            self._corpus_tagging_level == "full"), msg

        if _is_string(segments):
            word = segments
        else:
            # Throw away old category information, if any
            segments = self.detag_word(segments)
            # Merge potential segments
            word = ''.join(segments)

        # Return the best alternative from annotations if the word occurs there
        if word in self.annotations and strict_annot:
            annotation = self.annotations[word]
            alternatives = annotation.alternatives

            if not self._annotations_tagged:
                alternatives = tuple(self.viterbi_tag(alt, forbid_zzz=True)
                                     for alt in alternatives)

            sorted_alts = self.rank_analyses([AnalysisAlternative(alt, 0)
                                              for alt in alternatives])
            best = sorted_alts[0]
            return best.analysis, best.cost

        # To make sure that internally impossible states are penalized
        # even more than impossible states caused by zero parameters.
        extrazero = LOGPROB_ZERO ** 2

        # This function uses internally indices of categories,
        # instead of names and the word boundary object,
        # to remove the need to look them up constantly.
        categories = get_categories(wb=True)
        categories_nowb = [i for (i, c) in enumerate(categories)
                           if c != WORD_BOUNDARY]
        wb = categories.index(WORD_BOUNDARY)

        # Grid consisting of
        # the lowest accumulated cost ending in each possible state.
        # and back pointers that indicate the best path.
        # The grid is 3-dimensional:
        # grid [POSITION_IN_WORD]
        #      [MORPHLEN_OF_MORPH_ENDING_AT_POSITION - 1]
        #      [TAGINDEX_OF_MORPH_ENDING_AT_POSITION]
        # Initialized to pseudo-zero for all states
        zeros = [ViterbiNode(extrazero, None)] * len(categories)
        grid = [[zeros]]
        # Except probability one that first state is a word boundary
        grid[0][0][wb] = ViterbiNode(0, None)

        # Temporaries
        # Cumulative costs for each category at current time step
        cost = None
        best = ViterbiNode(extrazero, None)

        for pos in range(1, len(word) + 1):
            grid.append([])
            for next_len in range(1, pos + 1):
                grid[pos].append(list(zeros))
                prev_pos = pos - next_len
                morph = self._interned_morph(word[prev_pos:pos])

                if (self.nosplit_re and
                        pos < len(word) and
                        self.nosplit_re.match(word[(pos - 1):(pos + 1)])):
                    # Splitting at this point is forbidden
                    grid[pos][next_len - 1] = zeros
                    continue
                if morph not in self:
                    # The morph corresponding to this substring has not
                    # been encountered: zero probability for this solution
                    grid[pos][next_len - 1] = zeros
                    continue

                for next_cat in categories_nowb:
                    best = ViterbiNode(extrazero, None)
                    if prev_pos == 0:
                        # First morph in word
                        cost = self._corpus_coding.transit_emit_cost(
                            WORD_BOUNDARY, categories[next_cat], morph)
                        if cost <= best.cost:
                            best = ViterbiNode(cost, ((0, wb),
                                CategorizedMorph(morph, categories[next_cat])))
                    # implicit else: for-loop will be empty if prev_pos == 0
                    for prev_cat in categories_nowb:
                        t_e_cost = self._corpus_coding.transit_emit_cost(
                                        categories[prev_cat],
                                        categories[next_cat],
                                        morph)
                        for prev_len in range(1, prev_pos + 1):
                            cost = (t_e_cost +
                                grid[prev_pos][prev_len - 1][prev_cat].cost)
                            if cost <= best.cost:
                                best = ViterbiNode(cost, ((prev_len, prev_cat),
                                    CategorizedMorph(morph,
                                                     categories[next_cat])))
                    grid[pos][next_len - 1][next_cat] = best

        # Last transition must be to word boundary
        best = ViterbiNode(extrazero, None)
        for prev_len in range(1, len(word) + 1):
            for prev_cat in categories_nowb:
                cost = (grid[-1][prev_len - 1][prev_cat].cost +
                        self._corpus_coding.log_transitionprob(
                            categories[prev_cat],
                            WORD_BOUNDARY))
                if cost <= best.cost:
                    best = ViterbiNode(cost, ((prev_len, prev_cat),
                        CategorizedMorph(WORD_BOUNDARY, WORD_BOUNDARY)))

        if best.cost >= LOGPROB_ZERO:
            #_logger.warning(
            #    'No possible segmentation for word {}'.format(word))
            return [CategorizedMorph(word, DEFAULT_CATEGORY)], LOGPROB_ZERO

        # Backtrace for the best morph-category sequence
        result = []
        backtrace = best
        pos = len(word)
        bt_len = backtrace.backpointer[0][0]
        bt_cat = backtrace.backpointer[0][1]
        while pos > 0:
            backtrace = grid[pos][bt_len - 1][bt_cat]
            bt_len = backtrace.backpointer[0][0]
            bt_cat = backtrace.backpointer[0][1]
            result.insert(0, backtrace.backpointer[1])
            pos -= len(backtrace.backpointer[1])
        return tuple(result), best.cost

    def viterbi_tag(self, segments, forbid_zzz=False):
        """Tag a pre-segmented word using the learned model.

        Arguments:
            segments :  A list of morphs to tag.
                        Raises KeyError if morph is not present in the
                        training data.
                        For segmenting and tagging new words,
                        use viterbi_analyze(word).
            forbid_zzz :  If True, no morph can be tagged as a
                          non-morpheme.
        """

        # Throw away old category information, if any
        segments = self.detag_word(segments)
        return self._viterbi_tag_helper(segments, forbid_zzz=forbid_zzz)

    def fast_tag_gaps(self, segments):
        """Tag the gaps in a pre-segmented word where most morphs are already
        tagged. Existing tags can not be changed.
        """
        def constraint(i, cat):
            if segments[i].category is None:
                return False
            if cat == segments[i].category:
                return False
            return True

        return self._viterbi_tag_helper(segments, constraint,
                                        AbstractSegmenter.detag_morph)

    def _viterbi_tag_helper(self, segments,
                            constraint=None, mapping=lambda x: x,
                            forbid_zzz=False):
        # To make sure that internally impossible states are penalized
        # even more than impossible states caused by zero parameters.
        extrazero = LOGPROB_ZERO * 100

        # This function uses internally indices of categories,
        # instead of names and the word boundary object,
        # to remove the need to look them up constantly.
        categories = get_categories(wb=True)
        wb = categories.index(WORD_BOUNDARY)
        forbidden = []
        for (prev_cat, next_cat) in MorphUsageProperties.zero_transitions:
            forbidden.append((categories.index(prev_cat),
                              categories.index(next_cat)))
        if forbid_zzz:
            for (prev_cat, next_cat) in MorphUsageProperties.forbid_zzz:
                forbidden.append((categories.index(prev_cat),
                                  categories.index(next_cat)))

        # Grid consisting of
        # the lowest accumulated cost ending in each possible state.
        # and back pointers that indicate the best path.
        # Initialized to pseudo-zero for all states
        grid = [[ViterbiNode(extrazero, None)] * len(categories)]
        # Except probability one that first state is a word boundary
        grid[0][wb] = ViterbiNode(0, None)

        # Temporaries
        # Cumulative costs for each category at current time step
        cost = []
        best = []

        for (i, morph) in enumerate(segments):
            for (next_cat, nc_label) in enumerate(categories):
                if next_cat == wb:
                    # Impossible to visit boundary in the middle of the
                    # sequence
                    best.append(ViterbiNode(extrazero, None))
                    continue
                if constraint is not None and constraint(i, nc_label):
                    # lies outside the constrained path
                    best.append(ViterbiNode(extrazero, None))
                    continue
                morph = mapping(morph)
                for prev_cat in range(len(categories)):
                    if (prev_cat, next_cat) in forbidden:
                        cost.append(extrazero)
                        continue
                    # Cost of selecting prev_cat as previous state
                    # if now at next_cat
                    if grid[i][prev_cat].cost >= extrazero:
                        # This path is already failed
                        cost.append(extrazero)
                    else:
                        cost.append(grid[i][prev_cat].cost +
                                    self._corpus_coding.transit_emit_cost(
                                        categories[prev_cat],
                                        categories[next_cat], morph))
                best.append(ViterbiNode(*utils.minargmin(cost)))
                cost = []
            # Update grid to prepare for next iteration
            grid.append(best)
            best = []

        # Last transition must be to word boundary
        for prev_cat in range(len(categories)):
            pair = (categories[prev_cat], WORD_BOUNDARY)
            cost = (grid[-1][prev_cat].cost +
                    self._corpus_coding.log_transitionprob(*pair))
            best.append(cost)
        backtrace = ViterbiNode(*utils.minargmin(best))

        # Backtrace for the best category sequence
        result = [CategorizedMorph(
                    mapping(segments[-1]),
                    categories[backtrace.backpointer])]
        for i in range(len(segments) - 1, 0, -1):
            backtrace = grid[i + 1][backtrace.backpointer]
            morph = mapping(segments[i - 1])
            result.insert(0, CategorizedMorph(
                morph, categories[backtrace.backpointer]))
        return tuple(result)

    def forward_logprob(self, word):
        """Find log-probability of a word using the forward algorithm.

        Returns:
            cost      : (negative) log-probability of the word.
        """

        # To make sure that internally impossible states are penalized
        # even more than impossible states caused by zero parameters.
        extrazero = LOGPROB_ZERO ** 2

        # This function uses internally indices of categories,
        # instead of names and the word boundary object,
        # to remove the need to look them up constantly.
        categories = get_categories(wb=True)
        categories_nowb = [i for (i, c) in enumerate(categories)
                           if c != WORD_BOUNDARY]
        wb = categories.index(WORD_BOUNDARY)

        # Grid consisting of
        # the accumulated cost ending in each possible state.
        # The grid is 3-dimensional:
        # grid [POSITION_IN_WORD]
        #      [MORPHLEN_OF_MORPH_ENDING_AT_POSITION - 1]
        #      [TAGINDEX_OF_MORPH_ENDING_AT_POSITION]
        # Initialized to pseudo-zero for all states
        zeros = [extrazero] * len(categories)
        grid = [[zeros]]
        # Except probability one that first state is a word boundary
        grid[0][0][wb] = 0

        # Temporaries
        # Cumulative costs for each category at current time step
        cost = None
        psum = 0.0

        for pos in range(1, len(word) + 1):
            grid.append([])
            for next_len in range(1, pos + 1):
                grid[pos].append(list(zeros))
                prev_pos = pos - next_len
                morph = self._interned_morph(word[prev_pos:pos])

                if (self.nosplit_re and
                        pos < len(word) and
                        self.nosplit_re.match(word[(pos - 1):(pos + 1)])):
                    # Splitting at this point is forbidden
                    grid[pos][next_len - 1] = zeros
                    continue
                if morph not in self:
                    # The morph corresponding to this substring has not
                    # been encountered: zero probability for this solution
                    grid[pos][next_len - 1] = zeros
                    continue

                for next_cat in categories_nowb:
                    psum = 0.0
                    if prev_pos == 0:
                        # First morph in word
                        cost = self._corpus_coding.transit_emit_cost(
                            WORD_BOUNDARY, categories[next_cat], morph)
                        psum += math.exp(-cost)
                    # implicit else: for-loop will be empty if prev_pos == 0
                    for prev_cat in categories_nowb:
                        t_e_cost = self._corpus_coding.transit_emit_cost(
                                        categories[prev_cat],
                                        categories[next_cat],
                                        morph)
                        for prev_len in range(1, prev_pos + 1):
                            cost = (t_e_cost +
                                grid[prev_pos][prev_len - 1][prev_cat])
                            psum += math.exp(-cost)
                    if psum > 0:
                        cost = -math.log(psum)
                    else:
                        cost = LOGPROB_ZERO
                    grid[pos][next_len - 1][next_cat] = cost

        # Last transition must be to word boundary
        psum = 0.0
        for prev_len in range(1, len(word) + 1):
            for prev_cat in categories_nowb:
                cost = (grid[-1][prev_len - 1][prev_cat] +
                        self._corpus_coding.log_transitionprob(
                            categories[prev_cat],
                            WORD_BOUNDARY))
                psum += math.exp(-cost)
        if psum > 0:
            cost = -math.log(psum)
        else:
            cost = LOGPROB_ZERO

        return cost

    def rank_analyses(self, choices):
        """Choose the best analysis of a set of choices.

        Observe that the call and return signatures are different
        from baseline: this method is more versatile.

        Arguments:
            choices :  a sequence of AnalysisAlternative(analysis, penalty)
                       namedtuples.
                       The analysis must be a sequence of CategorizedMorphs,
                       (segmented and tagged).
                       The penalty is a float that is added to the cost
                       for this choice. Use 0 to disable.
        Returns:
            A sorted (by cost, ascending) list of
            SortedAnalysis(cost, analysis, index, breakdown) namedtuples. ::
                cost :  the contribution of this analysis to the corpus cost.
                analysis :  as in input.
                breakdown :  A CostBreakdown object, for diagnostics
        """
        out = []
        for (i, choice) in enumerate(choices):
            out.append(self.cost_breakdown(choice.analysis, choice.penalty, i))
        return sorted(out)

    def cost_breakdown(self, segmentation, penalty=0.0, index=0):
        """Returns breakdown of costs for the given tagged segmentation."""
        wrapped = _wb_wrap(segmentation)
        breakdown = CostBreakdown()
        for (prefix, suffix) in utils.ngrams(wrapped, n=2):
            cost = self._corpus_coding.log_transitionprob(prefix.category,
                                                          suffix.category)
            breakdown.transition(cost, prefix.category, suffix.category)
            if suffix.morph != WORD_BOUNDARY:
                cost = self._corpus_coding.log_emissionprob(
                        suffix.category, suffix.morph)
                breakdown.emission(cost, suffix.category, suffix.morph)
        if penalty != 0:
            breakdown.penalty(penalty)
        return SortedAnalysis(breakdown.cost, segmentation, index, breakdown)

    def _interned_morph(self, morph, store=False):
        """Override in subclass"""
        return morph

    def __contains__(self, morph):
        raise AttributeError('Must override __contains__')

    @staticmethod
    def get_categories(wb=False):
        """The category tags supported by this model.

        Arguments:
            wb :  If True, the word boundary will be included. Default: False.
        """
        return get_categories(wb)

    @staticmethod
    def detag_morph(morph):
        if isinstance(morph, CategorizedMorph):
            return morph.morph
        return morph

    @staticmethod
    def detag_word(segments):
        return tuple(AbstractSegmenter.detag_morph(x) for x in segments)

    @staticmethod
    def detag_list(segmentations):
        """Removes category tags from a segmented corpus."""
        for rcount, segments in segmentations:
            yield ((rcount, tuple(AbstractSegmenter.detag_morph(x)
                                  for x in segments)))

    @staticmethod
    def filter_untagged(segmentations):
        """Yields only the part of the corpus that is tagged."""
        for rcount, segments in segmentations:
            is_tagged = all(cmorph.category is not None
                            for cmorph in segments)
            if is_tagged:
                yield (rcount, segments)

    @property
    def word_tokens(self):
        return self._corpus_coding.boundaries
