from typing import List, Tuple
from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Enum

import logging
import math

_logger = logging.getLogger(__name__)
EPS = 1e-6


class PruneDecision(Enum):
    ALWAYS = 0
    GAIN   = 1
    LOSS   = 2
    NEVER_DBL        = 3
    NEVER_NO_ALT     = 4
    NEVER_SUPERVISED = 5
    NEVER_CHAR       = 6


@dataclass
class PruneStats:
    construction: str
    threshold_alpha: float
    delta_lc: float
    delta_cc: float
    delta_cost: float
    decision: PruneDecision


def prune_cost_at_alpha(alpha, delta_lc, delta_cc):
    delta_cost = delta_lc + (alpha * delta_cc)
    # tuning can't affect if both deltas have the same sign
    if delta_lc < 0 and delta_cc < 0:
        decision = PruneDecision.ALWAYS
        threshold_alpha = math.inf
    elif delta_lc > 0 and delta_cc > 0:
        decision = PruneDecision.NEVER_DBL
        threshold_alpha = -math.inf
    else:
        # if deltas have opposite sign,
        # compute the threshold alpha for which they cancel out
        threshold_alpha = abs(delta_lc / (delta_cc + EPS))
        # decicion based on current alpha
        decision = PruneDecision.GAIN if delta_cost < 0 else PruneDecision.LOSS
    return threshold_alpha, delta_cost, decision


class PruningCriterion(ABC):
    @abstractmethod
    def prune(self, pruning_stats: List[PruneStats]) -> Tuple[List[str], bool]:
        pass


class LexiconSizePruningCriterion(PruningCriterion):
    """
    Prune until `goal_lexicon` is reached.
    Prune at most `proportion`.
    """

    def __init__(self, proportion: float, goal_lexicon: int):
        self.proportion = proportion
        self.goal_lexicon = goal_lexicon

    def prune(self, pruning_stats: List[PruneStats]) -> Tuple[List[str], bool]:
        n_tot = len(pruning_stats)
        max_prune_prop = int(math.ceil(n_tot * self.proportion))
        max_prune_goal = max(0, int(n_tot - self.goal_lexicon))
        max_prune = min(max_prune_prop, max_prune_goal)
        # done unless epoch quota was the stopping reason
        done = max_prune_goal <= max_prune_prop
        pruning_stats.sort(key=lambda x: (x.decision.value, x.delta_cost))
        pruned = [x.construction for x in pruning_stats[:max_prune]]
        return pruned, done


class MDLPruningCriterion(PruningCriterion):
    """
    Prune based on decision.
    Prune at most proportion.
    """

    def __init__(self, proportion: float):
        self.proportion = proportion

    def prune(self, pruning_stats: List[PruneStats]) -> Tuple[List[str], bool]:
        n_tot = len(pruning_stats)
        max_prune_prop = int(math.ceil(n_tot * self.proportion))
        pruning_stats.sort(key=lambda x: (x.decision.value, x.delta_cost))
        pruned = []
        for (i, stat) in enumerate(pruning_stats):
            if i >= max_prune_prop:
                return pruned, False
            if stat.decision not in {PruneDecision.ALWAYS, PruneDecision.GAIN}:
                return pruned, True
            pruned.append(stat.construction)
        # pruned everything
        _logger.info('pruned everything!')
        return pruned, True


class AutotunePruningCriterion(PruningCriterion):
    """
    determine optimal alpha. prune at most proportion. prune based on decision
    """

    def __init__(self, proportion: float, goal_lexicon: int, first_prune_proportion=None):
        self.proportion = proportion
        self.goal_lexicon = goal_lexicon
        self.first_prune_proportion = first_prune_proportion

        self.optimal_alpha = None
        self.is_first_prune = True

    def prune(self, pruning_stats: List[PruneStats]) -> Tuple[List[str], bool]:
        # determine optimal alpha
        if len(pruning_stats) < self.goal_lexicon:
            _logger.info('already below goal')
            return [], True

        pruning_stats.sort(key=lambda x: (x.decision.value, -x.threshold_alpha))
        optimal_alpha = pruning_stats[-int(self.goal_lexicon)].threshold_alpha
        if optimal_alpha == -math.inf:
            _logger.info('cannot reach goal lexicon by tuning: too many always keep')
            optimal_alpha = min(x.threshold_alpha for x in pruning_stats
                                if x.decision.value in (PruneDecision.GAIN, PruneDecision.LOSS))
        if optimal_alpha == math.inf:
            _logger.info('cannot reach goal lexicon by tuning: infinite alpha')
            optimal_alpha = max(x.threshold_alpha for x in pruning_stats
                                if x.decision.value in (PruneDecision.GAIN, PruneDecision.LOSS))
        _logger.info(f"New optimal corpus weight is {optimal_alpha}")
        self.optimal_alpha = optimal_alpha  # Accessible from the outside.
        prune_stats = list(self.reweight_prune_stats(pruning_stats, optimal_alpha))

        # continue with pruning
        n_tot = len(prune_stats)
        prop = self.first_prune_proportion if self.is_first_prune and self.first_prune_proportion is not None else self.proportion
        max_prune_prop = int(math.ceil(n_tot * prop))
        max_prune_goal = max(0, int(n_tot - self.goal_lexicon))
        max_prune = min(max_prune_prop, max_prune_goal)
        # done unless epoch quota was the stopping reason
        done = max_prune_goal <= max_prune_prop
        prune_stats.sort(key=lambda x: (x.decision.value, x.delta_cost))
        pruned = []
        for (i, stat) in enumerate(prune_stats):
            if i >= max_prune:
                return pruned, done
            if stat.decision not in {PruneDecision.ALWAYS, PruneDecision.GAIN}:
                return pruned, done
            pruned.append(stat.construction)
        # pruned everything
        _logger.info('pruned everything!')
        self.is_first_prune = False
        return pruned, True

    def reweight_prune_stats(self, prune_stats, optimal_alpha: float):
        for stat in prune_stats:
            threshold_alpha, delta_cost, decision = prune_cost_at_alpha(optimal_alpha, stat.delta_lc, stat.delta_cc)
            yield PruneStats(
                stat.construction,
                stat.threshold_alpha,
                stat.delta_lc, stat.delta_cc,
                delta_cost,
                decision
            )
