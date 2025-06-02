from __future__ import unicode_literals

import collections
import math

from ..cost import LexiconEncoding, CorpusEncoding
from ...models.flatcat import utils, MorphUsageProperties
from ...models.flatcat.categorizationscheme import ByCategory, get_categories
from ...models.flatcat.flatcat import _logger
from ...models.flatcat.utils import LOGPROB_ZERO, zlog


class FlatcatLexiconEncoding(LexiconEncoding):
    """Extends LexiconEncoding to include the coding costs of the
    encoding cost of morph usage (context) features.

    Arguments:
        morph_usage :  A MorphUsageProperties object,
                       or something that quacks like it.
    """

    def __init__(self, morph_usage):
        super(FlatcatLexiconEncoding, self).__init__()
        self._morph_usage = morph_usage
        self.logfeaturesum = 0.0

    def clear(self):
        """Resets the cost variables.
        Use before fully reprocessing a segmented corpus."""
        self.logtokensum = 0.0
        self.logfeaturesum = 0.0
        self.tokens = 0
        self.boundaries = 0
        self.atoms.clear()

    def add(self, morph):
        super(FlatcatLexiconEncoding, self).add(morph)
        self.logfeaturesum += self._morph_usage.feature_cost(morph)

    def remove(self, morph):
        super(FlatcatLexiconEncoding, self).remove(morph)
        self.logfeaturesum -= self._morph_usage.feature_cost(morph)

    def get_cost(self):
        assert self.boundaries >= 0
        if self.boundaries == 0:
            return 0.0

        n = self.tokens + self.boundaries
        return ((n * math.log(n)
                 - self.boundaries * math.log(self.boundaries)
                 - self.logtokensum
                 + self.permutations_cost()
                 + self.logfeaturesum
                )  # * self.weight       # always 1
                + self.frequency_distribution_cost())

    def get_codelength(self, morph):
        cost = super(FlatcatLexiconEncoding, self).get_codelength(morph)
        cost += self._morph_usage.feature_cost(morph)
        return cost


class FlatcatEncoding(CorpusEncoding):
    """Class for calculating the encoding costs of the grammar and the
    corpus. Also stores the HMM parameters.

    tokens: the number of emissions observed.
    boundaries: the number of word tokens observed.
    """

    def __init__(self, morph_usage, lexicon_encoding, weight=1.0):
        self._morph_usage = morph_usage
        super(FlatcatEncoding, self).__init__(lexicon_encoding, weight)

        # Counts of emissions observed in the tagged corpus.
        # A dict of ByCategory objects indexed by morph. Counts occurences.
        self._emission_counts = utils.Sparse(
            default=utils._nt_zeros(ByCategory))

        # Counts of transitions between categories.
        # P(Category -> Category) can be calculated from these.
        # A dict of integers indexed by a tuple of categories.
        # Counts occurences.
        self._transition_counts = collections.Counter()

        # Counts of observed category tags.
        # Single Counter object (ByCategory is unsuitable, need break also).
        self._cat_tagcount = collections.Counter()

        # Caches for transition and emission logprobs,
        # to avoid wasting effort recalculating.
        self._log_transitionprob_cache = dict()
        self._log_emissionprob_cache = dict()
        self._persistent_log_emissionprob_cache = dict()
        # How frequent must a morph be to count as frequent
        self._persistence_limit = 3
        self._cache_size = 75000
        # Needed very often, reducing function calls
        self._categories = get_categories()

        self.logcondprobsum = 0.0

    # Transition count methods

    def get_transition_count(self, prev_cat, next_cat):
        return self._transition_counts[(prev_cat, next_cat)]

    def log_transitionprob(self, prev_cat, next_cat):
        """-Log of transition probability P(next_cat|prev_cat)"""
        pair = (prev_cat, next_cat)
        if pair not in self._log_transitionprob_cache:
            if self._cat_tagcount[prev_cat] == 0:
                self._log_transitionprob_cache[pair] = LOGPROB_ZERO
            else:
                self._log_transitionprob_cache[pair] = (
                    zlog(self._transition_counts[(prev_cat, next_cat)]) -
                    zlog(self._cat_tagcount[prev_cat]))
        # Assertion disabled due to performance hit
        #msg = 'transition {} -> {} has probability > 1'.format(
        #    prev_cat, next_cat)
        #assert self._log_transitionprob_cache[pair] >= 0, msg
        return self._log_transitionprob_cache[pair]

    def update_transition_count(self, prev_cat, next_cat, diff_count):
        """Updates the number of observed transitions between
        categories.
        OBSERVE! Clearing the cache is left to the caller.

        Arguments:
            prev_cat :  The name (not index) of the category
                        transitioned from.
            next_cat :  The name (not index) of the category
                        transitioned to.
            diff_count :  The change in the number of transitions.
        """

        # Assertion disabled due to performance hit
        #msg = 'update_transition_count needs category names, not indices'
        #assert not isinstance(prev_cat, int), msg
        #assert not isinstance(next_cat, int), msg
        pair = (prev_cat, next_cat)

        self._transition_counts[pair] += diff_count
        self._cat_tagcount[prev_cat] += diff_count

        # Assertion disabled due to performance hit
        #if self._transition_counts[pair] > 0:
        #    assert pair not in MorphUsageProperties.zero_transitions

        # Assertion disabled due to performance hit
        #msg = 'subzero transition count for {}'.format(pair)
        #assert self._transition_counts[pair] >= 0, msg
        #assert self._cat_tagcount[prev_cat] >= 0

    def clear_transition_counts(self):
        """Resets transition counts, costs and cache.
        Use before fully reprocessing a tagged segmented corpus."""
        self._transition_counts.clear()
        self._cat_tagcount.clear()
        self._log_transitionprob_cache.clear()

    # Emission count methods

    def get_emission_counts(self, morph):
        return self._emission_counts[morph]

    def log_emissionprob(self, category, morph, extrazero=False):
        """-Log of posterior emission probability P(morph|category)"""
        cat_index = self._categories.index(category)
        value = self._emission_helper(morph)[cat_index]
        # Assertion disabled due to performance hit
        #msg = 'emission {} -> {} has probability > 1'.format(category, morph)
        #assert value >= 0, msg
        if extrazero and value >= LOGPROB_ZERO:
            return value ** 2
        return value

    def _emission_helper(self, morph):
        if morph in self._persistent_log_emissionprob_cache:
            return self._persistent_log_emissionprob_cache[morph]
        if morph in self._log_emissionprob_cache:
            return self._log_emissionprob_cache[morph]
        count = self._morph_usage.count(morph)
        zlcount = zlog(count)
        zlctc = self._morph_usage.zlog_category_token_count()
        condprobs = self._morph_usage.condprobs(morph)
        tmp = []
        for (cat_index, cat) in enumerate(self._categories):
            # Not equal to what you get by:
            # zlog(self._emission_counts[morph][cat_index]) +
            if self._cat_tagcount[cat] == 0 or count == 0:
                value = LOGPROB_ZERO
            else:
                value = (zlcount +
                         zlog(condprobs[cat_index]) -
                         zlctc[cat_index])
            tmp.append(value)
        tmp = ByCategory(*tmp)
        if count >= self._persistence_limit:
            if len(self._persistent_log_emissionprob_cache) > self._cache_size:
                # Dont let the cache grow too big
                self._persistent_log_emissionprob_cache.clear()
                self._persistence_limit += 1
            self._persistent_log_emissionprob_cache[morph] = tmp
            return tmp
        if len(self._log_emissionprob_cache) > 10:
            # Small cache regularly emptied
            self._log_emissionprob_cache.clear()
        self._log_emissionprob_cache[morph] = tmp
        return tmp

    def update_emission_count(self, category, morph, diff_count):
        """Updates the number of observed emissions of a single morph from a
        single category, and the logtokensum (which is category independent).
        Updates logcondprobsum.

        Arguments:
            category :  name of category from which emission occurs.
            morph :  string representation of the morph.
            diff_count :  the change in the number of occurences.
        """
        if diff_count == 0:
            return
        assert category is not None
        cat_index = self._categories.index(category)
        old_count = self._emission_counts[morph][cat_index]
        new_count = old_count + diff_count
        logcondprob = -zlog(self._morph_usage.condprobs(morph)[cat_index])
        if old_count > 0:
            self.logcondprobsum -= old_count * logcondprob
        if new_count > 0:
            self.logcondprobsum += new_count * logcondprob
        new_counts = self._emission_counts[morph]._replace(
            **{category: new_count})
        self._set_emission_counts(morph, new_counts)

        # cached probabilities no longer valid
        self.clear_emission_cache()

    def _set_emission_counts(self, morph, new_counts):
        """Set the number of emissions of a morph from all categories
        simultaneously.
        Does not update logcondprobsum.

        Arguments:
            morph :  string representation of the morph.
            new_counts :  ByCategory object with new counts.
        """

        old_total = sum(self._emission_counts[morph])
        self._emission_counts[morph] = new_counts
        new_total = sum(new_counts)

        if old_total > 0:
            if old_total > 1:
                self.logtokensum -= old_total * math.log(old_total)
            self.tokens -= old_total
        if new_total > 0:
            if new_total > 1:
                self.logtokensum += new_total * math.log(new_total)
            self.tokens += new_total

        # cached probabilities no longer valid
        self.clear_emission_cache()

    def clear_emission_counts(self):
        """Resets emission counts and costs.
        Use before fully reprocessing a tagged segmented corpus."""
        self.tokens = 0
        self.logtokensum = 0.0
        self.logcondprobsum = 0.0
        self._emission_counts.clear()
        self._persistent_log_emissionprob_cache.clear()
        self._log_emissionprob_cache.clear()

    def clear_emission_cache(self):
        """Clears the cache for emission probability values.
        Use if an incremental change invalidates cached values."""
        self._persistent_log_emissionprob_cache.clear()
        self._log_emissionprob_cache.clear()

    def clear_transition_cache(self):
        """Clears the cache for emission probability values.
        Use if an incremental change invalidates cached values."""
        self._log_transitionprob_cache.clear()

    # General methods

    def transit_emit_cost(self, prev_cat, next_cat, morph):
        """Cost of transitioning from prev_cat to next_cat and emitting
        the morph."""
        if (prev_cat, next_cat) in MorphUsageProperties.zero_transitions:
            return LOGPROB_ZERO
        return (self.log_transitionprob(prev_cat, next_cat) +
                self.log_emissionprob(next_cat, morph))

    def update_count(self, construction, old_count, new_count):
        raise Exception('Inherited method not appropriate for FlatcatEncoding')

    def logtransitionsum(self):
        """Returns the term of the cost function associated with the
        transition probabilities. This term is recalculated on each call
        to get_cost, as the transition matrix is small and
        each segmentation change is likely to modify
        a large part of the transition matrix,
        making cumulative updates unnecessary.
        """
        categories = get_categories(wb=True)
        t_cost = 0.0
        # FIXME: this can be optimized using the same running tally
        # as logtokensum, when getting rid of the assertions
        # except if implementing hierarchy: then the incoming == outgoing
        # assumption doesn't necessarily hold anymore
        sum_transitions_from = collections.Counter()
        sum_transitions_to = collections.Counter()
        forbidden = MorphUsageProperties.zero_transitions
        for prev_cat in categories:
            for next_cat in categories:
                if (prev_cat, next_cat) in forbidden:
                    continue
                count = self._transition_counts[(prev_cat, next_cat)]
                if count == 0:
                    continue
                sum_transitions_from[prev_cat] += count
                sum_transitions_to[next_cat] += count
                t_cost += count * math.log(count)
        for cat in categories:
            # These hold, because for each incoming transition there is
            # exactly one outgoing transition (except for word boundary,
            # of which there are one of each in every word)
            assert sum_transitions_from[cat] == sum_transitions_to[cat]
            assert sum_transitions_to[cat] == self._cat_tagcount[cat]

        assert t_cost >= 0
        return t_cost

    def get_cost(self):
        """Override for the Encoding get_cost function.

        This is P( D_W | theta, Y )
        """
        if self.boundaries == 0:
            return 0.0

        n = self.tokens + self.boundaries
        return ((self.tokens * math.log(self.tokens)
                 - self.logtokensum
                 - self.logcondprobsum
                 - self.logtransitionsum()
                 + n * math.log(n)
                ) * self.weight
                + self.frequency_distribution_cost()
               )


class FlatcatAnnotatedCorpusEncoding(object):
    """Class for calculating the cost of encoding the annotated corpus"""
    def __init__(self, corpus_coding, weight=None):
        self.corpus_coding = corpus_coding
        if weight is None:
            self.weight = 1.0
            self.do_update_weight = True
        else:
            self.weight = weight
            self.do_update_weight = False
        self.logemissionsum = 0.0
        self.boundaries = 0

        # Counts of emissions observed in the tagged corpus.
        # A dict of ByCategory objects indexed by morph. Counts occurences.
        self._emission_counts = utils.Sparse(
            default=utils._nt_zeros(ByCategory))

        # Counts of transitions between categories.
        # P(Category -> Category) can be calculated from these.
        # A dict of integers indexed by a tuple of categories.
        # Counts occurences.
        self._transition_counts = collections.Counter()

    def set_counts(self, counts):
        """Sets the counts of emissions and transitions occurring
        in the annotated corpus to precalculated values."""
        self._emission_counts = utils.Sparse(
            default=utils._nt_zeros(ByCategory))
        self._transition_counts = collections.Counter()

        for (cmorph, new_count) in counts.emissions.items():
            assert new_count >= 0
            new_counts = self._emission_counts[cmorph.morph]._replace(
                **{cmorph.category: new_count})
            self._emission_counts[cmorph.morph] = new_counts

        for (pair, count) in counts.transitions.items():
            self._transition_counts[pair] = count
            assert self._transition_counts[pair] >= 0

    def update_counts(self, counts):
        """Updates the counts of emissions and transitions occurring
        in the annotated corpus, building on earlier counts."""
        for (cmorph, delta) in counts.emissions.items():
            cat_index = get_categories().index(cmorph.category)
            new_count = self._emission_counts[cmorph.morph][cat_index] + delta
            assert new_count >= 0
            new_counts = self._emission_counts[cmorph.morph]._replace(
                **{cmorph.category: new_count})
            self._emission_counts[cmorph.morph] = new_counts

        for (pair, delta) in counts.transitions.items():
            self._transition_counts[pair] += delta
            assert self._transition_counts[pair] >= 0

    def reset_contributions(self):
        """Recalculates the contributions of all morphs."""
        self.logemissionsum = 0.0
        categories = get_categories()
        for (morph, counts) in self._emission_counts.items():
            for (i, category) in enumerate(categories):
                msg = 'Annotation emission {} -> {} was subzero {}'.format(
                    category, morph, counts[i])
                assert counts[i] >= 0, msg
                self._contribution_helper(morph, category, counts[i])

    def modify_contribution(self, morph, direction):
        """Removes or readds the complete contribution of a morph to the
        cost function. The contribution must be removed using the same
        probability value as was used when adding it, making ordering of
        operations important.
        """
        categories = get_categories()
        counts = self._emission_counts[morph]
        for (i, category) in enumerate(categories):
            self._contribution_helper(morph, category, counts[i] * direction)

    def transition_cost(self):
        """Returns the term of the cost function associated with the
        transition probabilities. This term is recalculated on each call
        to get_cost, as the transition matrix is small and
        each segmentation change is likely to modify
        a large part of the transition matrix,
        making cumulative updates unnecessary.
        """
        cost = 0.0
        valid_transitions = MorphUsageProperties.valid_transitions()
        for pair in valid_transitions:
            count = self._transition_counts[pair]
            cost += count * self.corpus_coding.log_transitionprob(*pair)
        return cost

    def get_cost(self):
        """Returns the cost of encoding the annotated corpus"""
        if self.boundaries == 0:
            return 0.0
        tc = self.transition_cost()
        assert self.logemissionsum >= 0
        assert tc >= 0
        return (self.logemissionsum + tc) * self.weight

    def update_weight(self):
        """Update the weight of the Encoding by taking the ratio of the
        corpus boundaries and annotated boundaries.
        Does not scale by corpus weight,, unlike Morfessor Baseline.
        """
        if not self.do_update_weight:
            return
        old = self.weight
        self.weight = float(self.corpus_coding.boundaries) / self.boundaries
        if self.weight != old:
            _logger.info('Corpus weight of annotated data set to {}'.format(
                         self.weight))

    def _contribution_helper(self, morph, category, count):
        if count == 0:
            return
        self.logemissionsum += count * self.corpus_coding.log_emissionprob(
            category, morph, extrazero=True)
