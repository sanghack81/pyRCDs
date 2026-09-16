import types
import typing
from collections import OrderedDict, deque
from itertools import groupby, chain

import numpy as np
from numpy import median
from numpy.random import randint, randn

T = typing.TypeVar("T")


def safe_iter(iterable, sort=False):
    """
    Creates a generator that iterates safely over an iterable, yielding only items
    that still exist in the iterable at the time they are yielded.

    Parameters
    ----------
    iterable : iterable
        The collection of items to iterate over.
    sort : bool, optional
        Whether to sort the copied elements before iterating (default is False).
    """
    copied = list(iterable)
    if sort:
        copied = sorted(copied)
    for y in copied:
        if y in iterable:
            yield y


def pass_through(x: T, *args, **kwargs) -> T:
    """
    A simple pass-through function that returns its first argument unmodified.
    Useful as a default callback or identity function.
    """
    return x


def unions(sets):
    """
    Computes the union of multiple sets or set-yielding generators.

    Parameters
    ----------
    sets : iterable of sets or a generator
        The collections of iterable elements to be unified.

    Returns
    -------
    set
        A single set containing all unique elements from the provided sets.
    """
    if isinstance(sets, types.GeneratorType):
        return set(chain(*list(sets)))
    else:
        return set(chain(*sets))


def group_by(
    xs: typing.Iterable[T], keyfunc, sort=False
) -> typing.Iterator[tuple[typing.Any, list[T]]]:
    """
    Groups an iterable of items by a specific key function.

    Parameters
    ----------
    xs : iterable
        An iterable of items to group.
    keyfunc : callable
        A function to compute the group key for each item.
    sort : bool, optional
        If True, the items within each group list are sorted (default is False).

    Returns
    -------
    iterator of tuple
        An iterator yielding `(key, group_list)` tuples.
    """
    if not sort:
        return ((k, list(g)) for k, g in groupby(sorted(xs, key=keyfunc), key=keyfunc))
    else:
        kgs = [(k, list(sorted(g))) for k, g in groupby(sorted(xs, key=keyfunc), key=keyfunc)]
        kgs = sorted(kgs, key=lambda kg: kg[0])  # TODO Is it necessary?
        return iter(kgs)


class ratio_sampler:
    """
    A probabilistic sampler that yields values based on predefined relative ratios (weights).
    """

    def __init__(self, value_ratio_dict: dict):
        """
        Parameters
        ----------
        value_ratio_dict : dict
            A dictionary mapping values to strictly positive fractional ratios (weights).
        """
        self.values = list(value_ratio_dict.keys())
        self.ratios = np.array([value_ratio_dict[v] for v in self.values])
        assert np.all(self.ratios > 0)
        self.ratios = self.ratios / self.ratios.sum()
        self.pre_sampled = deque()

    def sample(self, size=None):
        """
        Samples a value based on the initialized probability distribution.
        If size is None, returns a single value, utilizing an internal pre-sampled queue for speed.
        """
        if size is None:
            if not self.pre_sampled:
                self.pre_sampled.extend(
                    np.random.choice(self.values, size=32, replace=True, p=self.ratios)
                )
            return self.pre_sampled.pop()
        else:
            return np.random.choice(self.values, size=size, replace=True, p=self.ratios)


class between_sampler:
    """Random integer sampler which returns an `int` between given `min_inclusive` and `max_inclusive`."""

    def __init__(self, min_inclusive, max_inclusive):
        assert min_inclusive <= max_inclusive
        self.m = min_inclusive
        self.M = max_inclusive

    def sample(self, size=None):
        """Returns a single integer if `size` is None, otherwise an array of integers."""
        if size is None:
            return randint(self.m, self.M + 1)
        else:
            return randint(self.m, self.M + 1, size=size)


class unif_sampler:
    """
    A continuous uniform distribution sampler.
    """

    def __init__(self, left=0.0, right=1.0):
        """
        Parameters
        ----------
        left : float
            The lower boundary of the uniform distribution.
        right : float
            The upper boundary of the uniform distribution.
        """
        assert left <= right
        self.left = left
        self.right = right

    def sample(self):
        """Returns a single uniformly sampled float."""
        return np.random.rand() * (self.right - self.left) + self.left


class normal_sampler:
    """
    A continuous normal (Gaussian) distribution sampler.
    """

    def __init__(self, mu=0.0, sd=1.0):
        """
        Parameters
        ----------
        mu : float
            Mean of the normal distribution.
        sd : float
            Standard deviation of the normal distribution.
        """
        self.mu = mu
        self.sd = sd

    def sample(self):
        """Returns a single normally sampled float."""
        return self.sd * randn() + self.mu


class laplacian_sampler:
    """
    A continuous Laplacian distribution sampler.
    """

    def __init__(self, mu=0.0, sd=1.0):
        """
        Parameters
        ----------
        mu : float
            Location parameter (mean) of the Laplacian distribution.
        sd : float
            Scale parameter (diversity) of the Laplacian distribution.
        """
        self.mu = mu
        self.sd = sd

    def sample(self):
        """Returns a single Laplacian sampled float."""
        return self.sd * np.random.laplace() + self.mu


def average_agg(default=0.0):
    """
    A function that returns a closure to calculate the average of a given input
    or `default` value if the given input is empty.
    """
    return lambda items: (sum(items) / len(items)) if len(items) > 0 else default


def sum_agg(default=0.0):
    """
    A function that returns a closure to calculate the sum of an input iterable.
    The `default` parameter is unused but maintained for signature consistency.
    """
    return lambda items: sum(items)


def sum_sqrt_agg(default=0.0):
    """
    A function that returns a closure to calculate the signed square root
    of the sum of an input iterable.
    Maintains the sign indicating the direction of aggregated values.
    """
    return lambda items: np.sqrt(sum(items)) if sum(items) >= 0 else -np.sqrt(-sum(items))


def max_agg(default=0.0):
    """
    A function that returns a closure to calculate the maximum value of a given input
    or default value if the given input is empty.
    """
    return lambda items: max(items) if len(items) > 0 else default


def linear_gaussian(parameters: dict, aggregator, error) -> typing.Callable:
    """
    A linear model factory with additive Gaussian noise.

    Parameters
    ----------
    parameters : dict
        Parameters (weights) for the linear model mapped by relational variable.
    aggregator : callable
        A function that maps multiple values to a single aggregate value.
    error : sampler
        An additive noise distribution sampler instance (e.g., `normal_sampler`).

    Returns
    -------
    callable
        A function computing the linear Gaussian model prediction based on item attributes.
    """

    def func(values, cause_item_attrs):
        value = 0

        for rvar in sorted(parameters.keys()):
            item_attr_values = [values[item_attr] for item_attr in cause_item_attrs[rvar]]
            value += parameters[rvar] * aggregator(item_attr_values)

        return value + error.sample()

    return func


def xors(parameters, flip=0):
    """
    A boolean model factory that XORs given values across relational variables.

    Parameters
    ----------
    parameters : dict
        Weights/presence indicators mapped by relational variable.
    flip : float, optional
        A probability indicating the likelihood to flip (negate) the final XOR
        computation result (default is 0).

    Returns
    -------
    callable
        A deterministic (or slightly randomized if `flip` > 0) XOR-based boolean function.
    """

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
    """
    A noisy continuous approximation of the boolean XOR model factory.
    Inputs are rounded to binary spaces, then combined using XORs, and finally
    subjected to additive Gaussian noise.

    Parameters
    ----------
    parameters : dict
        Influence specifications per relational variable.
    flip : float
        Probability of boolean negation before continuous noise addition.
    error_std : float
        The standard deviation of the additive continuous Gaussian noise.

    Returns
    -------
    callable
        A noisy XOR model inference function.
    """

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
    """
    Computes the median value of a matrix excluding its diagonal elements.

    Parameters
    ----------
    D : numpy.ndarray
        A square matrix.
    exclude_inf : bool, optional
        Whether to exclude infinity values prior to median computation (default is True).
    default : int, optional
        Default scalar returned if there are no valid elements to compute the median across.
    """
    return stat_except_diag(D, exclude_inf, default, median)


def mean_except_diag(D, exclude_inf=True, default=1):
    """
    Computes the mean value of a matrix excluding its diagonal elements.

    Parameters
    ----------
    D : numpy.ndarray
        A square matrix.
    exclude_inf : bool, optional
        Whether to exclude infinity values prior to mean computation (default is True).
    default : int, optional
        Default scalar returned if there are no valid elements to compute the mean across.
    """
    return stat_except_diag(D, exclude_inf, default, np.mean)


def stat_except_diag(D, exclude_inf=True, default=1, func=median):
    """
    Applies a given statistical function to all non-diagonal elements of a square matrix.

    Raises
    ------
    TypeError
        If the matrix `D` is not 2-dimensional or not square.
    ValueError
        If the matrix `D` contains 1 or fewer rows/columns.
    """
    if D.ndim != 2:
        raise TypeError("not a matrix")
    if D.shape[0] != D.shape[1]:
        raise TypeError("not a square matrix")
    if len(D) <= 1:
        raise ValueError("No non-diagonal element")

    lower = D[np.tri(len(D), k=-1, dtype=bool)]
    upper = D.transpose()[np.tri(len(D), k=-1, dtype=bool)]
    non_diagonal = np.concatenate((lower, upper))
    if exclude_inf:
        non_diagonal = non_diagonal[non_diagonal != float("inf")]

    if len(non_diagonal):
        return func(non_diagonal)
    else:
        return default


def centering(M: np.ndarray) -> np.ndarray:
    """
    Performs double centering on a square matrix (subtracting row/column means and adding grand mean).

    Parameters
    ----------
    M : numpy.ndarray
        The input square matrix.

    Returns
    -------
    numpy.ndarray
        The double-centered formulation of `M`.
    """
    nr, nc = M.shape
    assert nr == nc
    n = nr
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ M @ H


class LRUCache:
    """
    A basic Least Recently Used (LRU) Cache data structure built upon OrderedDict.
    Automatically evicts the oldest items when exceeding its maximum capacity.
    """

    def __init__(self, max_capacity=128):
        assert max_capacity >= 1
        self.max_capacity = max_capacity
        self.cache = OrderedDict()

    def __contains__(self, item):
        return item in self.cache

    def __getitem__(self, key):
        """Get a key-associated value if it exists, subsequently promoting its usage rank. Returns None otherwise."""
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
        """
        Retrieves a key from the cache.
        If it does not exist, executes `func(*args, **kwargs)`, stores the result, and returns it.
        """
        if key in self.cache:
            return self[key]
        self[key] = func(*args, **kwargs)
        return self[key]

    def __len__(self):
        return len(self.cache)

    def __iter__(self):
        return iter(self.cache)


class NdarrayLRUCache:
    """
    A size-aware Least Recently Used (LRU) Cache specific to numpy arrays.
    It evicts older items from memory when the cumulative memory footprint (in Megabytes)
    exceeds the configured threshold byte limit.
    """

    def __init__(self, max_n_megabytes=1024):
        assert max_n_megabytes >= 1
        self.max_n_megabytes = max_n_megabytes
        self.used_n_bytes = 0
        self.cache = OrderedDict()

    def __contains__(self, key):
        return key in self.cache

    def __getitem__(self, key):
        """Get an array if it exists and updates its usage status, otherwise returns None."""
        if key in self.cache:
            np_array = self.cache[key]
            self.cache.move_to_end(key)
            return np_array
        return None

    def __setitem__(self, key, np_array):
        """Caches a new numpy array. Progressively removes strictly older items if memory is over limit."""
        if key not in self:
            self.used_n_bytes += np_array.nbytes
            while self.used_n_bytes > 1024 * 1024 * self.max_n_megabytes and self.cache:
                _, v = self.cache.popitem(last=False)
                self.used_n_bytes -= v.nbytes

            self.cache[key] = np_array

        self.cache.move_to_end(key)

    def __len__(self):
        return len(self.cache)

    def __iter__(self):
        return iter(self.cache)
