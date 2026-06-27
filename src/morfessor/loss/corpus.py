"""Implementations for corpus and lexicon encoding and weighting"""
from __future__ import unicode_literals
from typing import List
from abc import ABC, abstractmethod

import logging
import re

import collections
import math

from ..models.flatcat.flatcat import MorphUsageProperties
from ..util.misc import _progress, LOGPROB_ZERO, zlog, Sparse, _nt_zeros
from ..util.constructions.base import _ConstructionMethods
from ..util.flatcat.categorizationscheme import ByCategory, get_categories

_logger = logging.getLogger(__name__)


class Encoding(object):
    """Base class for calculating the entropy (encoding length) of a corpus
    or lexicon.

    Commonly subclassed to redefine specific methods.

    """
    def __init__(self, weight=1.0):
        """Initizalize class

        Arguments:
            weight: weight used for this encoding
        """
        self.logtokensum = 0.0
        self.tokens = 0
        self.boundaries = 0
        self.weight = weight

    # constant used for speeding up logfactorial calculations with Stirling's
    # approximation
    _log2pi = math.log(2 * math.pi)

    @property
    def types(self):
        """Define number of types as 0. types is made a property method to
        ensure easy redefinition in subclasses

        """
        return 0

    @classmethod
    def _logfactorial(cls, n):
        """Calculate logarithm of n!.

        For large n (n > 20), use Stirling's approximation.

        """
        if n < 2:
            return 0.0
        if n < 20:
            return math.log(math.factorial(n))
        logn = math.log(n)
        return n * logn - n + 0.5 * (logn + cls._log2pi)

    def frequency_distribution_cost(self):
        """Calculate -log[(u - 1)! (v - u)! / (v - 1)!]

        v is the number of tokens+boundaries and u the number of types

        """
        if self.types < 2:
            return 0.0
        tokens = self.tokens + self.boundaries
        return (self._logfactorial(tokens - 1) -
                self._logfactorial(self.types - 1) -
                self._logfactorial(tokens - self.types))

    def permutations_cost(self):
        """The permutations cost for the encoding."""
        return -self._logfactorial(self.boundaries)

    def update_count(self, construction, old_count, new_count):
        """Update the counts in the encoding."""
        self.tokens += new_count - old_count
        if old_count > 1:
            self.logtokensum -= old_count * math.log(old_count)
        if new_count > 1:
            self.logtokensum += new_count * math.log(new_count)

    def get_cost(self):
        """Calculate the cost for encoding the corpus/lexicon"""
        if self.boundaries == 0:
            return 0.0

        n = self.tokens + self.boundaries
        return ((n * math.log(n)
                 - self.boundaries * math.log(self.boundaries)
                 - self.logtokensum
                 + self.permutations_cost()) * self.weight
                + self.frequency_distribution_cost())


class CorpusEncoding(Encoding):
    """Encoding the corpus class

    The basic difference to a normal encoding is that the number of types is
    not stored directly but fetched from the lexicon encoding. Also does the
    cost function not contain any permutation cost.
    """
    def __init__(self, lexicon_encoding, weight=1.0):
        super(CorpusEncoding, self).__init__(weight)
        self.lexicon_encoding = lexicon_encoding

    @property
    def types(self):
        """Return the number of types of the corpus, which is the same as the
         number of boundaries in the lexicon + 1

        """
        return self.lexicon_encoding.boundaries + 1

    def frequency_distribution_cost(self):
        """Calculate -log[(M - 1)! (N - M)! / (N - 1)!] for M types and N
        tokens.

        """
        if self.types < 2:
            return 0.0
        tokens = self.tokens
        return (self._logfactorial(tokens - 1) -
                self._logfactorial(self.types - 2) -
                self._logfactorial(tokens - self.types + 1))

    def get_cost(self):
        """Override for the Encoding get_cost function. A corpus does not
        have a permutation cost

        """
        if self.boundaries == 0:
            return 0.0

        n = self.tokens + self.boundaries
        return ((n * math.log(n)
                 - self.boundaries * math.log(self.boundaries)
                 - self.logtokensum) * self.weight
                + self.frequency_distribution_cost())


class AnnotatedCorpusEncoding(Encoding):
    """Encoding the cost of an Annotated Corpus.

    In this encoding constructions that are missing are penalized.

    """
    def __init__(self, corpus_coding, weight=None, penalty=-9999.9):
        """
        Initialize encoding with appropriate meta data

        Arguments:
            corpus_coding: CorpusEncoding instance used for retrieving the
                             number of tokens and boundaries in the corpus
            weight: The weight of this encoding. If the weight is None,
                      it is updated automatically to be in balance with the
                      corpus
            penalty: log penalty used for missing constructions

        """
        super(AnnotatedCorpusEncoding, self).__init__()
        self.do_update_weight = True
        self.weight = 1.0
        if weight is not None:
            self.do_update_weight = False
            self.weight = weight
        self.corpus_coding = corpus_coding
        self.penalty = penalty
        self.constructions = collections.Counter()

    def set_constructions(self, constructions):
        """Method for re-initializing the constructions. The count of the
        constructions must still be set with a call to set_count

        """
        self.constructions = constructions
        self.boundaries = len(constructions)
        self.tokens = sum(constructions.values())
        self.logtokensum = 0.0

    def set_count(self, construction, count):
        """Set an initial count for each construction. Missing constructions
        are penalized
        """
        annot_count = self.constructions[construction]
        if count > 0:
            self.logtokensum += annot_count * math.log(count)
        else:
            self.logtokensum += annot_count * self.penalty

    def update_count(self, construction, old_count, new_count):
        """Update the counts in the Encoding, setting (or removing) a penalty
         for missing constructions

        """
        if construction in self.constructions:
            annot_count = self.constructions[construction]
            if old_count > 0:
                self.logtokensum -= annot_count * math.log(old_count)
            else:
                self.logtokensum -= annot_count * self.penalty
            if new_count > 0:
                self.logtokensum += annot_count * math.log(new_count)
            else:
                self.logtokensum += annot_count * self.penalty

    def update_weight(self):
        """Update the weight of the Encoding by taking the ratio of the
        corpus boundaries and annotated boundaries
        """
        if not self.do_update_weight:
            return
        old = self.weight
        self.weight = (float(self.corpus_coding.boundaries) / self.boundaries)
        if self.weight != old:
            _logger.info("Corpus weight of annotated data set to %s"
                         % self.weight)

    def get_cost(self):
        """Return the cost of the Annotation Corpus."""
        if self.boundaries == 0:
            return 0.0
        n = self.tokens + self.boundaries
        # parametrization changed: both alpha and beta are applied
        weight = self.corpus_coding.weight * self.weight
        return ((n * math.log(self.corpus_coding.tokens +
                              self.corpus_coding.boundaries)
                 - self.boundaries * math.log(self.corpus_coding.boundaries)
                 - self.logtokensum) * weight)


class LexiconEncoding(Encoding):
    """Class for calculating the encoding cost for the Lexicon"""

    def __init__(self):
        """Initialize Lexcion Encoding"""
        super(LexiconEncoding, self).__init__()
        self.atoms = collections.Counter()

    @property
    def types(self):
        """Return the number of different atoms in the lexicon + 1 for the
        compound-end-token

        """
        return len(self.atoms) + 1

    def add(self, construction):
        """Add a construction to the lexicon, updating automatically the
        count for its atoms

        """
        self.boundaries += 1
        for atom in construction:
            c = self.atoms[atom]
            self.atoms[atom] = c + 1
            self.update_count(atom, c, c + 1)

    def remove(self, construction):
        """Remove construction from the lexicon, updating automatically the
        count for its atoms

        """
        self.boundaries -= 1
        for atom in construction:
            c = self.atoms[atom]
            self.atoms[atom] = c - 1
            self.update_count(atom, c, c - 1)

    def get_codelength(self, construction):
        """Return an approximate codelength for new construction."""
        l = len(construction) + 1
        cost = l * math.log(self.tokens + l)
        cost -= math.log(self.boundaries + 1)
        for atom in construction:
            if atom in self.atoms:
                c = max(1, self.atoms[atom])
            else:
                c = 1
            cost -= math.log(c)
        return cost


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
        self._emission_counts = Sparse(default=_nt_zeros(ByCategory))

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
        self._emission_counts = Sparse(default=_nt_zeros(ByCategory))

        # Counts of transitions between categories.
        # P(Category -> Category) can be calculated from these.
        # A dict of integers indexed by a tuple of categories.
        # Counts occurences.
        self._transition_counts = collections.Counter()

    def set_counts(self, counts):
        """Sets the counts of emissions and transitions occurring
        in the annotated corpus to precalculated values."""
        self._emission_counts = Sparse(default=_nt_zeros(ByCategory))
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


###################################################################################"


class CorpusWeight(ABC):

    @abstractmethod
    def update(self, model, epoch: int):
        pass

    @classmethod
    def move_direction(cls, model, direction, epoch):
        if direction != 0:
            weight = model.get_corpus_coding_weight()
            if direction > 0:
                weight *= 1 + 2.0 / epoch
            else:
                weight *= 1.0 / (1 + 2.0 / epoch)
            model.set_corpus_coding_weight(weight)
            _logger.info("Corpus weight set to {}".format(weight))
            return True
        return False


class FixedCorpusWeight(CorpusWeight):
    def __init__(self, weight):
        self.weight = weight

    def update(self, model, epoch: int):
        model.set_corpus_coding_weight(self.weight)
        return False


class AlignedTokenCountCorpusWeight(CorpusWeight):
    """Class for using a sentence-aligned parallel bilingual corpus
    to set the corpus weight in such a way that the number of
    morphs in corpus of the language to be segmented
    is as similar as possible to the number of tokens on the reference side.
    """
    re_token_sep = re.compile(r'\s+', re.UNICODE)
    align_losses = ('abs', 'square', 'zeroone', 'tot')

    def __init__(self,
                 unsegmented_dev,
                 reference_dev,
                 threshold=0.01,
                 loss='abs',
                 linguistic_dev=None):
        self.unsegmented_dev = list(self.tokenize(unsegmented_dev))
        self.reference_counts = list(len(x) for x
                                     in self.tokenize(reference_dev))
        _logger.info('Total reference tokens {}'.format(
            sum(self.reference_counts)))
        self.threshold = threshold
        self.align_loss_idx = self.align_losses.index(loss)
        assert len(self.unsegmented_dev) == len(self.reference_counts)
        self.previous_weight = None
        self.previous_cost = None
        self.previous_d = None
        if linguistic_dev is not None:
            self.linguistic_dev = list(self.tokenize(linguistic_dev))
            assert len(self.linguistic_dev) == len(self.reference_counts)
        else:
            self.linguistic_dev = None

    def update(self, model, epoch: int):
        if epoch < 1:
            # Can't use viterbi_segment before first epoch
            return False
        weight = model.get_corpus_coding_weight()
        (cost, d) = self.evaluation(model)
        if self.previous_cost is not None:
            absdiff = abs(cost - self.previous_cost)
            absthresh = self.previous_cost * self.threshold
            if absdiff < absthresh:
                _logger.info("Align cost delta {} is below threshold {}. "
                    "Weight learning stopped".format(absdiff, absthresh))
                return False
        if self.previous_weight is None or cost < self.previous_cost:
            # accept the previous step
            self.previous_weight = weight
            self.previous_cost = cost
            self.previous_d = d
            _logger.info("Accepting step to {}".format(weight))
        else:
            # revert the previous step
            weight = self.previous_weight
            _logger.info("Reverting weight to {}".format(weight))
            model.set_corpus_coding_weight(weight)
            cost = self.previous_cost
            d = self.previous_d
        # new step
        return self.move_direction(model, d, epoch)

    @classmethod
    def tokenize(cls, lines):
        for line in lines:
            line = line.strip()
            yield cls.re_token_sep.split(line)

    def evaluation(self, model):
        costs, d, _ = self.calculate_costs(model)
        cost = costs[self.align_loss_idx]
        return (cost, d)

    def calculate_costs(self, model):
        abs_cost = 0.0
        sq_cost = 0.0
        zeroone_cost = 0.0
        tot_cost = 0.0
        direction = 0
        tot_tokens = 0
        cache = {}
        if self.linguistic_dev is not None:
            self.morph_totals = collections.Counter()
            self.morph_scores_pos = collections.Counter()
            self.morph_scores_neg = collections.Counter()
            linguistic_dev_iter = iter(self.linguistic_dev)
        _logger.info('Segmenting aligned parallel corpus for weight learning')
        for (tokens, ref) in zip(_progress(self.unsegmented_dev),
                                 self.reference_counts):
            segments = collections.Counter()
            for w in tokens:
                segments.update(self._cached_seg(model, cache, w))
            segcount = sum(segments.values())
            tot_tokens += segcount
            diff = segcount - ref
            if diff > 0:
                d = 1
            elif diff < 0:
                d = -1
            else:
                d = 0
            direction += d
            abs_cost += abs(diff)
            sq_cost += diff**2
            if diff != 0:
                zeroone_cost += 1
            tot_cost += diff
            if self.linguistic_dev is not None:
                # also count morph-type-level scores
                ling_morphs = collections.Counter(next(linguistic_dev_iter))
                self.morph_totals.update(ling_morphs)
                # Observe: - operator (as opposed to .subtract)
                #   uses multiset semantics, 
                #   and will not result in negative counts.
                not_in_seg = ling_morphs - segments
                in_seg = ling_morphs - not_in_seg
                if diff > 0:
                    # oversegmented
                    for morph in ling_morphs:
                        # strong plus if split in an overseg sentence
                        self.morph_scores_pos[morph] += in_seg[morph]
                        # weak plus if joined in an overseg sentence
                        self.morph_scores_neg[morph] -= not_in_seg[morph]
                elif diff < 0:
                    # undersegmented
                    for morph in ling_morphs:
                        # strong minus if joined in an underseg sentence
                        self.morph_scores_neg[morph] += not_in_seg[morph]
                        # weak minus if split in an underseg sentence
                        self.morph_scores_pos[morph] -= in_seg[morph]
        tot_cost = abs(tot_cost)
        costs = (abs_cost, sq_cost, zeroone_cost, tot_cost)
        _logger.info('Align costs {}, direction {}, total tokens {}'.format(
            costs, direction, tot_tokens))
        return (costs, direction, tot_tokens)

    def _cached_seg(self, model, cache, word):
        if word not in cache:
            try:
                seg = model._get_stored_analysis(word)
            except (KeyError, AttributeError):
                # don't use viterbi_segment: the only unseen words should be
                # unanalyzable words, which are not split anyhow
                #seg = model.viterbi_segment(word)[0]
                seg = [word]
            cache[word] = seg
        return cache[word]


class AnnotationCorpusWeight(CorpusWeight):
    """Class for using development annotations to update the corpus weight
    during batch training."""

    def __init__(self, devel_set, threshold, cc: _ConstructionMethods):
        self.data = devel_set
        self.threshold = threshold
        self.cc = cc

    def update(self, model, epoch: int):
        """Tune model corpus weight based on the precision and
        recall of the development data, trying to keep them equal"""
        if epoch < 1:
            return False
        tmp = self.data.items()
        wlist, annotations = zip(*tmp)
        segments = [model.viterbi_segment(w)[0] for w in wlist]
        d = self._estimate_segmentation_dir(segments, annotations)

        return self.move_direction(model, d, epoch)

    def _boundary_recall(self, prediction: List[List[List[str]]], reference: List[List[List[str]]]):
        """Calculate average boundary recall for given segmentations.
           You can have multiple predictions per example and multiple references per example."""
        rec_total = 0
        rec_sum = 0.0
        for example_predictions, example_references in zip(prediction, reference):
            # For this example, find the best prediction-reference pair.
            best = -1
            for ref in example_references:
                reference_boundaries = set(self.cc.parts_to_splitlocs(ref))
                if not reference_boundaries:  # By definition, this has recall of 1.0, and you can't do better than this, so terminate early.
                    best = 1.0
                    break

                for pre in example_predictions:
                    prediction_boundaries = set(self.cc.parts_to_splitlocs(pre))
                    r = len(reference_boundaries & prediction_boundaries) / len(reference_boundaries)
                    if r > best:
                        best = r
            if best >= 0:
                rec_sum += best
                rec_total += 1
        return rec_sum, rec_total

    def _bpr_evaluation(self, prediction, reference):
        """Return boundary precision, recall, and F-score for segmentations."""
        rec_s, rec_t = self._boundary_recall(prediction, reference)
        pre_s, pre_t = self._boundary_recall(reference, prediction)
        rec = rec_s / rec_t
        pre = pre_s / pre_t
        f = 2.0 * pre * rec / (pre + rec)
        return pre, rec, f

    def _estimate_segmentation_dir(self, segments, annotations):
        """Estimate if the given compounds are under- or oversegmented.

        The decision is based on the difference between boundary precision
        and recall values for the given sample of segmented data.

        Arguments:
          segments: list of predicted segmentations
          annotations: list of reference segmentations

        Return 1 in the case of oversegmentation, -1 in the case of
        undersegmentation, and 0 if no changes are required.

        """
        pre, rec, f = self._bpr_evaluation([[x] for x in segments], annotations)
        _logger.info("Boundary evaluation: precision %.4f; recall %.4f" % (pre, rec))
        if abs(pre - rec) < self.threshold:
            return 0
        elif rec > pre:
            return 1
        else:
            return -1


class MorphLengthCorpusWeight(CorpusWeight):
    def __init__(self, morph_length, threshold=0.01):
        self.morph_length = morph_length
        self.threshold = threshold

    def update(self, model, epoch: int):
        if epoch < 1:
            return False
        cur_length = self.calc_morph_length(model)

        _logger.info("Current morph-length: {}".format(cur_length))

        if abs(self.morph_length - cur_length) / self.morph_length > self.threshold:
            d = abs(self.morph_length - cur_length) / (self.morph_length - cur_length)
            return self.move_direction(model, d, epoch)
        return False

    @classmethod
    def calc_morph_length(cls, model):
        total_constructions = 0
        total_atoms = 0
        for compound in model.get_compounds():
            constructions = model._get_stored_analysis(compound)
            for construction in constructions:
                total_constructions += 1
                total_atoms += len(construction)
        if total_constructions > 0:
            return float(total_atoms) / total_constructions
        else:
            return 0.0


class NumMorphCorpusWeight(CorpusWeight):
    def __init__(self, num_morph_types, threshold=0.01):
        self.num_morph_types = num_morph_types
        self.threshold = threshold

    def update(self, model, epoch: int):
        if epoch < 1:
            return False
        cur_morph_types = model._lexicon_coding.boundaries

        _logger.info("Number of morph types: {}".format(cur_morph_types))

        if abs(self.num_morph_types - cur_morph_types) / self.num_morph_types > self.threshold:
            d = abs(self.num_morph_types - cur_morph_types) / (self.num_morph_types - cur_morph_types)
            return self.move_direction(model, d, epoch)
        return False
