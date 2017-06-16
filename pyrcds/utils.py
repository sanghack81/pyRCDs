import types
import typing
from collections import OrderedDict
from itertools import groupby, chain

import numpy as np
from numpy import median
from numpy.random import randint, randn

T = typing.TypeVar('T')


def safe_iter(iterable, sort=False):
    """Iterator (generator) that can skip removed items."""
    copied = list(iterable)
    if sort:
        copied = sorted(copied)
    for y in copied:
        if y in iterable:
            yield y


def pass_through(x: T, *args, **kwargs) -> T:
    return x


def unions(sets):
    """Big Union -- union of sets or union of generators"""
    if isinstance(sets, types.GeneratorType):
        return set(chain(*list(sets)))
    else:
        return set(chain(*sets))


def group_by(xs: typing.Iterable[T], keyfunc, sort=False) -> typing.Iterator[typing.Tuple[T, typing.List]]:
    """A generator of tuples of a key and its group as a `list`"""
    if not sort:
        return ((k, list(g)) for k, g in groupby(sorted(xs, key=keyfunc), key=keyfunc))
    else:
        kgs = [(k, list(sorted(g))) for k, g in groupby(sorted(xs, key=keyfunc), key=keyfunc)]
        kgs = sorted(kgs, key=lambda kg: kg[0])  # TODO Is it necessary?
        return iter(kgs)


class between_sampler:
    """Random integer sampler which returns an `int` between given `min_inclusive` and `max_inclusive`."""

    def __init__(self, min_inclusive, max_inclusive):
        assert min_inclusive <= max_inclusive
        self.m = min_inclusive
        self.M = max_inclusive

    def sample(self, size=None):
        if size is None:
            return randint(self.m, self.M + 1)
        else:
            return randint(self.m, self.M + 1, size=size)


class unif_sampler:
    def __init__(self, left=0.0, right=1.0):
        assert left <= right
        self.left = left
        self.right = right

    def sample(self):
        return np.random.rand() * (self.right - self.left) + self.left


class normal_sampler:
    def __init__(self, mu=0.0, sd=1.0):
        self.mu = mu
        self.sd = sd

    def sample(self):
        return self.sd * randn() + self.mu


class laplacian_sampler:
    def __init__(self, mu=0.0, sd=1.0):
        self.mu = mu
        self.sd = sd

    def sample(self):
        return self.sd * np.random.laplace() + self.mu


def average_agg(default=0.0):
    """A function that returns the average of a given input or `default` value if the given input is empty."""

    return lambda items: (sum(items) / len(items)) if len(items) > 0 else default


def sum_agg(default=0.0):
    return lambda items: sum(items)


def sum_sqrt_agg(default=0.0):
    return lambda items: np.sqrt(sum(items)) if sum(items) >= 0 else -np.sqrt(-sum(items))


def max_agg(default=0.0):
    """A function that returns the maximum value of a given input
     or default value if the given input is empty.
     """

    return lambda items: max(items) if len(items) > 0 else default


def linear_gaussian(parameters: dict, aggregator, error):
    """A linear model with an additive Gaussian noise.

    Parameters
    ----------
    parameters : parameters for the linear model. e.g., parameter = parameters[cause_rvar]
    aggregator: a function that maps multiple values to a single value.
    error: additive noise distribution, err = error.sample()

    Returns
    -------
    function
        a linear Gaussian model with given parameters and an aggregator
    """

    def func(values, cause_item_attrs):
        value = 0

        for rvar in sorted(parameters.keys()):
            item_attr_values = [values[item_attr] for item_attr in cause_item_attrs[rvar]]
            value += parameters[rvar] * aggregator(item_attr_values)

        return value + error.sample()

    return func


def xors(parameters, flip=0):
    """A function that xor-s given values."""

    def func(values, cause_item_attrs):
        value = 0 if parameters else randint(2)

        for rvar in sorted(parameters.keys()):
            for item_attr in cause_item_attrs[rvar]:
                value ^= values[item_attr]

        if np.random.rand() < flip:
            return 1 - value
        return value

    return func


def noisy_xors(parameters, flip=0.0, error_std=0.2):
    """A function that xor-s given values."""

    def to_0_or_1(x):
        if abs(x) < abs(x - 1):
            return 0
        else:
            return 1

    def func(values, cause_item_attrs):
        value = 0 if parameters else randint(2)

        for rvar in sorted(parameters.keys()):
            for item_attr in cause_item_attrs[rvar]:
                value ^= to_0_or_1(values[item_attr])

        if np.random.rand() < flip:
            return 1 - value
        return value + error_std * randn()

    return func


def median_except_diag(D, exclude_inf=True, default=1):
    """A median value of a matrix except diagonal elements.

    Parameters
    ----------
    D : matrix-like
    exclude_inf : bool
        whether to exclude infinity
    default : int
        default return value if there is no value to compute median
    """
    return stat_except_diag(D, exclude_inf, default, median)


def mean_except_diag(D, exclude_inf=True, default=1):
    """A mean value of a matrix except diagonal elements.

    Parameters
    ----------
    D : array_like
        Array to be measured
    exclude_inf : bool
        whether to exclude infinity
    default : int
        default return value if there is no value to compute mean
    """
    return stat_except_diag(D, exclude_inf, default, np.mean)


def stat_except_diag(D, exclude_inf=True, default=1, func=median):
    if D.ndim != 2:
        raise TypeError('not a matrix')
    if D.shape[0] != D.shape[1]:
        raise TypeError('not a square matrix')
    if len(D) <= 1:
        raise ValueError('No non-diagonal element')

    lower = D[np.tri(len(D), k=-1, dtype=bool)]
    upper = D.transpose()[np.tri(len(D), k=-1, dtype=bool)]
    non_diagonal = np.concatenate((lower, upper))
    if exclude_inf:
        non_diagonal = non_diagonal[non_diagonal != float('inf')]

    if len(non_diagonal):
        return func(non_diagonal)
    else:
        return default


def centering(M: np.ndarray) -> np.ndarray:
    """Matrix Centering"""
    nr, nc = M.shape
    assert nr == nc
    n = nr
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ M @ H


class LRUCache:
    def __init__(self, max_capacity=128):
        assert max_capacity >= 1
        self.max_capacity = max_capacity
        self.cache = OrderedDict()

    def __contains__(self, item):
        return item in self.cache

    def __getitem__(self, key):
        """Get a key-associating value or None"""
        if key in self.cache:
            value = self.cache[key]
            self.cache.move_to_end(key)
            return value
        return None

    def __setitem__(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.max_capacity:
            self.cache.popitem(last=False)

    def lazy(self, key, func, *args, **kwargs):
        if key in self.cache:
            return self[key]
        self[key] = func(*args, **kwargs)
        return self[key]

    def __len__(self):
        return len(self.cache)

    def __iter__(self):
        return iter(self.cache)


class NdarrayLRUCache:
    def __init__(self, max_n_megabytes=1024):
        assert max_n_megabytes >= 1
        self.max_n_megabytes = max_n_megabytes
        self.used_n_bytes = 0
        self.cache = OrderedDict()

    def __contains__(self, key):
        return key in self.cache

    def __getitem__(self, key):
        """Get a key-associating value or None"""
        if key in self.cache:
            np_array = self.cache[key]
            self.cache.move_to_end(key)
            return np_array
        return None

    def __setitem__(self, key, np_array):
        if key not in self:
            self.used_n_bytes += np_array.nbytes
            if self.used_n_bytes > 1024 * 1024 * self.max_n_megabytes:
                _, v = self.cache.popitem(last=False)
                self.used_n_bytes -= v.nbytes

            self.cache[key] = np_array

        self.cache.move_to_end(key)

    def __len__(self):
        return len(self.cache)

    def __iter__(self):
        return iter(self.cache)
