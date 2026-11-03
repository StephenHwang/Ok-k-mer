"""Tie-correct Spearman scoring and exact integer weight grids."""
from decimal import Decimal, InvalidOperation
import numpy as np
from scipy.stats import rankdata


def weight_denominator(step):
    try:
        step = Decimal(str(step))
        if not step.is_finite() or not 0 < step <= 1:
            raise ValueError('w-step must be finite and in (0,1]')
        reciprocal = 1 / step
        if reciprocal != reciprocal.to_integral_value():
            raise ValueError('w-step must divide 1 exactly, e.g. 0.05 or 0.1')
        return int(reciprocal)
    except InvalidOperation as exc:
        raise ValueError('Invalid w-step') from exc


def compositions(total, size):
    """Positive integer coefficients; divided by total they sum exactly to one."""
    if size == 1:
        yield (total,)
    else:
        for first in range(1, total-size+2):
            for rest in compositions(total-first, size-1):
                yield (first, *rest)


def centered_ranks(values):
    ranks = rankdata(values, method='average')
    ranks -= ranks.mean()
    norm = np.sqrt(np.sum(ranks*ranks))
    return ranks, norm


class Scorer:
    def __init__(self, distances, target):
        self.target, self.target_norm = centered_ranks(target)
        if self.target_norm == 0:
            raise ValueError('Reference tree distances are constant')
        # Average ranks are multiples of 1/2. Integer scores retain exact ties
        # even when floating-point weight arithmetic would perturb them.
        self.ranks = np.asarray([np.rint(2*rankdata(col, method='average')).astype(np.int64)
                                 for col in distances.T])

    def score(self, indices, coefficients):
        values = np.zeros(self.ranks.shape[1], dtype=np.int64)
        for i, coefficient in zip(indices, coefficients):
            values += int(coefficient) * self.ranks[i]
        ranks, norm = centered_ranks(values)
        if norm == 0:
            return None
        return float(np.clip(np.sum(ranks*self.target) / (norm*self.target_norm), -1, 1))
