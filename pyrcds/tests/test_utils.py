import typing

from pyrcds.utils import between_sampler, average_agg, max_agg, normal_sampler, linear_gaussian, group_by, \
    unions, LRUCache, safe_iter


def test_group_by():
    grouped = list(group_by(((1, 2), (2, 3), (3, 2), (4, 4)), lambda x: x[1]))
    assert len(grouped) == 3
    d = dict(grouped)
    assert set(d.keys()) == {2, 3, 4}
    assert d[2] == [(1, 2), (3, 2)]


def test_safe_iter():
    x = [1, 2, 3, 4, 5]
    it = safe_iter(x)
    x.remove(3)
    assert 1 == next(it)
    assert 2 == next(it)
    assert 4 == next(it)
    x.remove(5)
    try:
        next(it)
    except StopIteration:
        return
    assert False


def test_unions():
    xs = ({i, i + 1} for i in range(1, 10, 3))
    assert unions(xs) == {1, 2, 4, 5, 7, 8}
    xs = [{1, 2}, {4, 5, 6}, {1, 3, 5}]
    assert unions(xs) == {1, 2, 3, 4, 5, 6}


def test_sampler():
    sampler = between_sampler(10, 11)
    x, y, z = sampler.sample(), sampler.sample(), sampler.sample()
    assert 10 <= min(x, y, z) <= max(x, y, z) <= 11
    samples = sampler.sample(1000)
    assert len(samples) == 1000
    samples = list(sorted(samples))
    assert samples[0] == 10
    assert samples[-1] == 11


def test_aggregators():
    agg = average_agg(1.0)
    assert 2.0 == agg([1, 2, 3])
    assert 1.0 == agg(set())

    agg = max_agg(1.0)
    assert 3.0 == agg([1, 2, 3])
    assert 1.0 == agg(set())


def test_linear_gaussian():
    lgm = linear_gaussian({'x': 2.0, 'y': 1.0}, average_agg(), normal_sampler(0.0, 0.0))  # type: typing.Callable
    values = {'a': 1.0, 'b': 2.0, 'c': 3.0}
    cause_attrs = {'x': ['a', 'b', 'c'], 'y': ['c']}

    v = lgm(values, cause_attrs)
    assert v == 2.0 * (sum(values.values()) / len(values)) + 1.0 * 3.0


def test_linear_max_gaussian():
    lgm = linear_gaussian({'x': 2.0, 'y': 1.0}, max_agg(), normal_sampler(1.0, 0.0))  # type: typing.Callable
    values = {'a': 1.0, 'b': 2.0, 'c': 3.0}
    cause_attrs = {'x': ['a', 'b', 'c'], 'y': ['c']}

    v = lgm(values, cause_attrs)
    assert v == 2.0 * 3.0 + 1.0 * 3.0 + 1.0


def test_LRUCache():
    cache = LRUCache(3)
    cache[0] = 'a'
    cache[1] = 'b'
    cache[2] = 'c'
    # 0,1,2
    cache[3] = 'd'
    # 1,2,3
    assert 'b' == cache[1]  # this will make 0,2,3,1
    # 2,3,1
    assert (0 not in set(cache))
    assert (1 in set(cache))
    cache.lazy(4, lambda a: a + 1, 4)
    assert cache[4] == 5
    assert (2 not in set(cache))
    print('passed must be printed next line')
    cache.lazy(4, lambda a: exit(-1), 4)
    assert cache[4] == 5
    assert set(cache) == {1, 3, 4}
    print('passed')
