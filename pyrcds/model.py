import functools
import typing
from collections import defaultdict, namedtuple, deque
from itertools import cycle, product, combinations

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from numpy.random.mtrand import choice, shuffle, randint, randn

from pyrcds.domain import EntityClass, RelationshipClass, ItemClass, RelationalSchema, AttributeClass, RelationalSkeleton, SkItem, ImmutableRSkeleton
from pyrcds.graphs import PDAG
from pyrcds.utils import average_agg, normal_sampler, group_by, linear_gaussian, unions


def is_valid_rpath(path) -> bool:
    """Whether the given array-like item classes can be a valid relational path.

    Parameters
    ----------
    path : array_like of `I_Class`
    """
    assert not isinstance(path, RelationalPath)
    E, R = EntityClass, RelationshipClass
    assert path is not None and len(path) >= 1
    assert isinstance(path[0], E) or isinstance(path[0], R)

    # alternating sequence
    if isinstance(path[0], E):
        classes = cycle([E, R])
    else:
        classes = cycle([R, E])
    if not all(isinstance(item_class, cls) for item_class, cls in zip(path, classes)):
        return False

    # participation
    def is_related(x, y):
        return (x in y.entities) if isinstance(y, R) else (y in x.entities)

    if not all(is_related(x, y) for x, y in zip(path, path[1:])):
        return False

    # ERE, RER constraints
    zipped = zip(path[::2], path[1::2], path[2::2])
    shfited_zipped = zip(path[1::2], path[2::2], path[3::2])
    if isinstance(path[0], E):
        ere, rer = zipped, shfited_zipped
    else:
        ere, rer = shfited_zipped, zipped
    return all(e1 != e2 for e1, r, e2 in ere) and all(r1 != r2 or r1.is_many(e) for r1, e, r2 in rer)


@functools.total_ordering
class RelationalPath:
    """Relational path"""

    def __init__(self, item_classes, backdoor=False):
        """

        Parameters
        ----------
        item_classes : ItemClass or iterable of I_Class
        backdoor : bool
            whether to omit validity of given `item_classes`
        """
        assert item_classes is not None
        if isinstance(item_classes, ItemClass):
            item_classes = (item_classes,)
        if not backdoor:
            if not is_valid_rpath(item_classes):
                raise ValueError('not a valid path: {}'.format(item_classes))
        self.__item_classes = tuple(item_classes)
        self.__h = hash(self.__item_classes)
        # self.__getitem__ = functools.lru_cache(maxsize=10)(self.__getitem__)

    def __hash__(self):
        return self.__h

    def __iter__(self):
        return iter(self.__item_classes)

    def __eq__(self, other):
        return isinstance(other, RelationalPath) and \
               self.__h == other.__h and \
               self.__item_classes == other.__item_classes

    def __bool__(self):
        return True

    def __le__(self, other):
        return self.__item_classes <= other.__item_classes

    def __getitem__(self, item):
        """a subpath based on a slice.

        Notes
        -----
        Unlike usual slicing operator,
        Both start and stop are inclusive.
        start <= stop.
        step, if given, must be -1.
        """
        if isinstance(item, int):
            return self.__item_classes[item]
        elif isinstance(item, slice):
            start = 0 if item.start is None else item.start
            stop = len(self) if item.stop is None else item.stop + 1
            assert 0 <= start < stop <= len(self)

            if item.step == -1:
                return RelationalPath(tuple(reversed(self.__item_classes[start:stop])), True)
            else:
                return RelationalPath(self.__item_classes[start:stop], True)

        else:
            raise ValueError('unknown {}'.format(item))

    def __pow__(self, other):
        """Concatenate two paths."""
        if self.joinable(other):
            return self.join(other, True)
        return None

    @property
    def is_canonical(self):
        return len(self.__item_classes) == 1

    def subpath(self, start, end):
        assert 0 <= start < end <= len(self)
        return RelationalPath(self.__item_classes[start:end])

    def __reversed__(self):
        return RelationalPath(tuple(reversed(self.__item_classes)), True)

    @property
    def hop_len(self):
        return len(self) - 1

    @property
    def terminal(self):
        return self.__item_classes[-1]

    @property
    def base(self):
        return self.__item_classes[0]

    def __len__(self):
        return len(self.__item_classes)

    def appended_or_none(self, item_class: ItemClass):
        if len(self) > 1:
            if is_valid_rpath(self.__item_classes[-2:] + (item_class,)):
                return RelationalPath(self.__item_classes + (item_class,), True)
        else:
            if item_class.is_relationship_class:
                if self.terminal in item_class.entities:
                    return RelationalPath(self.__item_classes + (item_class,), True)
            elif self.terminal.is_relationship_class:
                if item_class in self.terminal.entities:
                    return RelationalPath(self.__item_classes + (item_class,), True)
        return None

    def joinable(self, rpath):
        if self.terminal != rpath.base:
            return False
        if len(self) > 1 and len(rpath) > 1:
            return is_valid_rpath([self.__item_classes[-2], self.__item_classes[-1], rpath.__item_classes[1]])
        return True

    def join(self, rpath, backdoor=False):
        if not backdoor:
            assert self.joinable(rpath)
        return RelationalPath(self.__item_classes + rpath.__item_classes[1:], True)

    def __str__(self):
        return '[' + (', '.join(str(i) for i in self.__item_classes)) + ']'

    def __repr__(self):
        return str(self)


def llrsp(p1: RelationalPath, p2: RelationalPath) -> int:
    """Longest Length of Required Shared Path

    References
    ----------
    [1] Sanghack Lee and Vasant Honavar (2015),
        Lifted Representation of Relational Causal Models Revisited:
        Implications for Reasoning and Structure Learning,
        In Proceedings of Workshop on Advances in Causal Inference co-located with UAI 2015
    """
    prev = None
    for i, (x, y) in enumerate(zip(p1, p2)):
        if x != y or (i > 0 and x.is_relationship_class and x.is_many(prev)):
            return i
        prev = x

    return min(len(p1), len(p2))


def eqint(p1: RelationalPath, p2: RelationalPath):
    """Equal or intersectible"""
    if (p1.base, p1.terminal) != (p2.base, p2.terminal):
        return False
    return p1 == p2 or llrsp(p1, p2) + llrsp(reversed(p1), reversed(p2)) <= min(len(p1), len(p2))


# Immutable
@functools.total_ordering
class RelationalVariable:
    """Relational variable"""

    def __init__(self, rpath, attr):
        if not isinstance(rpath, RelationalPath):
            rpath = RelationalPath(rpath)
        if isinstance(attr, str):
            attr = AttributeClass(attr)
        assert attr in rpath.terminal.attrs
        self.rpath = rpath
        self.attr = attr
        self.__h = hash(self.rpath) ^ hash(self.attr)

    @property
    def terminal(self):
        return self.rpath.terminal

    @property
    def base(self):
        return self.rpath.base

    def __le__(self, other):
        return (self.rpath, self.attr) <= (other.rpath, other.attr)

    @property
    def is_canonical(self):
        return self.rpath.is_canonical

    def __iter__(self):
        return iter((self.rpath, self.attr))

    def __len__(self):
        return len(self.rpath)

    def __hash__(self):
        return self.__h

    def __eq__(self, other):
        return isinstance(other, RelationalVariable) and \
               self.__h == other.__h and \
               self.rpath == other.rpath and \
               self.attr == other.attr

    def __str__(self):
        return str(self.rpath) + '.' + str(self.attr)

    def __repr__(self):
        return str(self)


def canonical_rvars(schema: RelationalSchema):
    """Returns all canonical relational variables given schema"""
    return set(RelationalVariable(RelationalPath(item_class), attr)
               for item_class in schema.item_classes
               for attr in item_class.attrs)


@functools.total_ordering
class RelationalDependency:
    """Relational dependency"""

    def __init__(self, cause: RelationalVariable, effect: RelationalVariable):
        assert effect.is_canonical
        assert cause.attr != effect.attr
        self.cause = cause
        self.effect = effect
        self.__h = hash(self.cause) ^ hash(self.effect)

    def __hash__(self):
        return self.__h

    def __eq__(self, other):
        return isinstance(other, RelationalDependency) and \
               self.cause == other.cause and \
               self.effect == other.effect

    def __le__(self, other):
        return (self.cause, self.effect) <= (other.cause, other.effect)

    def __len__(self):
        return len(self.cause.rpath)

    def __iter__(self):
        return iter([self.cause, self.effect])

    @property
    def hop_len(self) -> int:
        return len(self) - 1

    # reversed(P.X --> Vy) = ~P.Y --> Vx
    def __reversed__(self):
        new_cause = RelationalVariable(reversed(self.cause.rpath), self.effect.attr)
        new_effect = RelationalVariable(RelationalPath(self.cause.terminal), self.cause.attr)
        return RelationalDependency(new_cause, new_effect)

    def opposite(self) -> 'RelationalDependency':
        new_cause = RelationalVariable(reversed(self.cause.rpath), self.effect.attr)
        new_effect = RelationalVariable(RelationalPath(self.cause.terminal), self.cause.attr)
        return RelationalDependency(new_cause, new_effect)

    # dual(P.X --> Vy) = (Vx, ~P.Y)
    @property
    def dual(self):
        new_cause = RelationalVariable(reversed(self.cause.rpath), self.effect.attr)
        new_effect = RelationalVariable(RelationalPath(self.cause.terminal), self.cause.attr)
        return new_effect, new_cause

    def attrfy(self):
        return self.cause.attr, self.effect.attr

    def __str__(self):
        return str(self.cause) + " -> " + str(self.effect)

    def __repr__(self):
        return str(self)


@functools.total_ordering
class SymTriple:
    """Symmetric triple where (X,Y,Z) == (Z,Y,X)"""

    def __init__(self, left, middle, right):
        self.left, self.middle, self.right = left, middle, right

    def __hash__(self):
        return (hash(self.left) + hash(self.right)) ^ hash(self.middle)

    def __eq__(self, other):
        return isinstance(other, SymTriple) and \
               (self.left, self.right, self.middle) == (other.left, other.right, other.middle) or \
               (self.right, self.left, self.middle) == (other.left, other.right, other.middle)

    def __iter__(self):
        return iter((self.left, self.middle, self.right))

    def sides(self):
        return {self.left, self.right}

    @property
    def dual(self):
        return SymTriple(self.right, self.middle, self.left)

    def __str__(self):
        return '<' + (', '.join(str(t) for t in (self.left, self.middle, self.right))) + '>'

    def __lt__(self, other):
        return sorted([*self]) < sorted([*other])


@functools.total_ordering
class UndirectedRDep:
    """Undirected representation of relational dependency"""

    def __init__(self, rdep: RelationalDependency):
        assert isinstance(rdep, RelationalDependency)
        self.rdeps = frozenset({rdep, rdep.opposite()})
        # self.rdeps = frozenset({rdep, reversed(rdep)})

    def __lt__(self, other):
        return sorted(self.rdeps) < sorted(other.rdeps)

    def __eq__(self, other):
        return isinstance(other, UndirectedRDep) and self.rdeps == other.rdeps

    def __hash__(self):
        return hash(self.rdeps)

    def __iter__(self):
        return iter(self.rdeps)

    @property
    def hop_len(self) -> int:
        return next(iter(self.rdeps)).hop_len

    def __str__(self):
        c, e = next(iter(sorted(self)))
        return str(c) + " -- " + str(e)

    def __repr__(self):
        return str(self)

    def attrfy(self):
        dep = next(iter(self.rdeps))
        return frozenset({dep.cause.attr, dep.effect.attr})


class PRCM:
    """Partially-directed Relational Causal Model"""

    def __init__(self, schema: RelationalSchema, dependencies=None):
        if dependencies is None:
            dependencies = frozenset()
        self.schema = schema

        self.directed_dependencies = set()
        self.undirected_dependencies = set()

        self.parents = defaultdict(set)
        self.children = defaultdict(set)
        self.neighbors = defaultdict(set)

        self.add(dependencies)

    @property
    def valid_dependencies(self):
        """A set of feasible relational dependencies"""
        return self.directed_dependencies | {d for u in self.undirected_dependencies for d in u}

    @property
    def full_dependencies(self):
        return {d_ for d in self.directed_dependencies for d_ in (d, (reversed(d)))} | \
               {d for u in self.undirected_dependencies for d in u}

    @property
    def degree(self):
        """The maximum number of causes (or causes to be) of a canonical relational variable in the PRCM.

        Notes
        -----
        The number can be decreased as more relational dependencies are oriented.
        """
        return max(len(self.adj(v)) for v in (self.parents.keys() | self.neighbors.keys()))

    def pa(self, rvar: RelationalVariable):
        return self.parents[rvar]

    def ch(self, rvar: RelationalVariable):
        return self.children[rvar]

    def ne(self, rvar: RelationalVariable):
        return self.neighbors[rvar]

    def adj(self, rvar: RelationalVariable):
        return self.neighbors[rvar] | self.parents[rvar] | self.children[rvar]

    @property
    def max_hop(self) -> int:
        """The maximum hop length of either undirected and directed relational dependencies.

        Returns
        -------
        int
            -1 if there is no dependency
        """
        a = max(len(v) for k, vs in self.parents.items() for v in vs) if self.directed_dependencies else 0
        b = max(len(v) for k, vs in self.neighbors.items() for v in vs) if self.undirected_dependencies else 0
        return -1 + max(a, b)

    # TODO make it as a member variable
    @property
    def class_dependency_graph(self) -> PDAG:
        """The class dependency graph reflecting the current status of PRCM.

        Notes
        -----
        Not a view. A new instance is created every call.
        """

        cdg = PDAG()
        cdg.add_nodes(self.schema.attrs)
        cdg.add_edges((cause.attr, effect.attr) for effect, causes in self.parents.items() for cause in causes)
        cdg.add_undirected_edges((k.attr, v.attr) for k, vs in self.neighbors.items() for v in vs)

        return cdg

    def add(self, d):
        """Add an (undirected) relational dependency"""
        if isinstance(d, RelationalDependency):
            cause, effect = d  # Px --> Vy
            if d in self.directed_dependencies:
                return
            if UndirectedRDep(d) in self.undirected_dependencies:
                raise ValueError('undirected dependency exists for {}'.format(d))
            if reversed(d) in self.directed_dependencies:
                raise ValueError('opposite-directed dependency exists for {}'.format(d))
            self.parents[effect].add(cause)
            dual_cause, dual_effect = d.dual
            self.children[dual_cause].add(dual_effect)
            self.directed_dependencies.add(d)
        elif isinstance(d, UndirectedRDep):
            d1, d2 = d
            if d in self.undirected_dependencies:
                return
            if d1 in self.directed_dependencies:
                raise ValueError('directed dependency {} exists'.format(d1))
            if d2 in self.directed_dependencies:
                raise ValueError('directed dependency {} exists'.format(d2))
            self.neighbors[d1.effect].add(d1.cause)
            self.neighbors[d2.effect].add(d2.cause)
            self.undirected_dependencies.add(d)
        else:
            for x in d:  # delegate as far as it is iterable
                self.add(x)

    def remove(self, d):
        if isinstance(d, RelationalDependency) and d not in self.directed_dependencies:
            if d.cause in self.ne(d.effect):
                self.orient_as(reversed(d))
                return
        elif isinstance(d, RelationalDependency) and d in self.directed_dependencies:
            self.parents[d.effect].remove(d.cause)
            dual_cause, dual_effect = d.dual
            self.children[dual_cause].discard(dual_effect)
            self.directed_dependencies.remove(d)

        elif isinstance(d, UndirectedRDep) and d in self.undirected_dependencies:
            d1, d2 = d
            self.neighbors[d1.effect].remove(d1.cause)
            self.neighbors[d2.effect].discard(d2.cause)
            self.undirected_dependencies.remove(d)

        else:
            for x in d:  # delegate as far as it is iterable
                self.remove(x)

    def orient_as(self, edge):
        if edge in self.directed_dependencies:
            return False
        assert isinstance(edge, RelationalDependency)
        cause, effect = edge
        assert cause in self.ne(effect)
        self.remove(UndirectedRDep(edge))
        self.add(edge)
        return True

    def orient_with(self, x, y):
        """Orient all undirected dependencies of P.X--Vy (or ~P.Y--Vx) to P.X-->Vy."""
        for udep in list(self.undirected_dependencies):
            if udep.attrfy() == frozenset({x, y}):
                for dep in udep:
                    if dep.attrfy() == (x, y):
                        self.orient_as(dep)
                        break

    def __eq__(self, other):
        return isinstance(other, PRCM) and \
               self.directed_dependencies == other.directed_dependencies and \
               self.undirected_dependencies == other.undirected_dependencies


class RCM(PRCM):
    """Relational Causal Model"""

    def __init__(self, schema: RelationalSchema, dependencies=None):
        super().__init__(schema, dependencies)

    def add(self, d):
        assert not isinstance(d, UndirectedRDep)
        super().add(d)

        # def is_non_descendant(self, x, y):
        #     self.class_dependency_graph


class ParamRCM(RCM):
    """Parametrized Relational Causal Model"""

    def __init__(self, schema: RelationalSchema, dependencies, functions: dict):
        super().__init__(schema, dependencies)
        self.functions = functions


def terminal_set(skeleton: RelationalSkeleton, rpath, base_item: SkItem, semantics='path') -> typing.Set[SkItem]:
    """A terminal set of given relational path from the base item on the given relational skeleton.

    Parameters
    ----------
    base_item
    rpath
    skeleton
    semantics : {'path', 'bridge-burning'}
    """
    if semantics == 'bridge-burning':
        return terminal_set_bbs(skeleton, rpath, base_item)

    if isinstance(rpath, RelationalDependency):
        rpath = rpath.cause.rpath
    elif isinstance(rpath, RelationalVariable):
        rpath = rpath.rpath
    if not isinstance(rpath, RelationalPath):
        raise TypeError('{} is not RPath but {}'.format(rpath, type(rpath)))
    assert rpath[0] == base_item.item_class
    item_paths = [[base_item]]
    next_paths = []

    iterator = iter(rpath)
    next(iterator)
    for item_class in iterator:
        for item_path in item_paths:
            next_items = skeleton.neighbors(item_path[-1], item_class) - set(item_path)
            next_paths += [item_path + [item, ] for item in next_items]

        if not next_paths:
            return set()

        item_paths = next_paths
        next_paths = []

    assert all(len(path) == len(rpath) for path in item_paths)
    return {path[-1] for path in item_paths}


def terminal_set_bbs(skeleton: RelationalSkeleton, rpath: RelationalPath, base_item: SkItem):
    if isinstance(rpath, RelationalDependency):
        rpath = rpath.cause.rpath
    elif isinstance(rpath, RelationalVariable):
        rpath = rpath.rpath
    if not isinstance(rpath, RelationalPath):
        raise TypeError('{} is not RPath but {}'.format(rpath, type(rpath)))
    assert rpath[0] == base_item.item_class
    Pi = {base_item}
    previouses = set(Pi)
    for i in range(1, len(rpath)):
        Pi = unions(skeleton.neighbors(p, rpath[i]) for p in Pi) - previouses
        previouses |= Pi
    return Pi


def flatten(skeleton: RelationalSkeleton, rvars, with_base_items=False, value_only=False, n_jobs=1, verbose=False):
    rvars = list(rvars)
    assert len({rvar.base for rvar in rvars}) == 1
    base_class = rvars[0].base
    base_items = sorted(skeleton.items(base_class))

    data = np.empty([len(base_items), (1 if with_base_items else 0) + len(rvars)], dtype=object)
    if with_base_items:
        data[:, 0] = base_items

    inner_data = Parallel(n_jobs, verbose=5 if verbose else 0)(delayed(__inner_flatten)(skeleton, rvar, base_items, value_only) for j, rvar in enumerate(rvars, start=1 if with_base_items else 0))

    shift = 1 if with_base_items else 0
    for j in range(len(rvars)):
        data[:, j + shift] = inner_data[j]

    return data


def __inner_flatten(skeleton, rvar, base_items, value_only):
    inner_data = np.zeros((len(base_items),), dtype=object)
    for i, base_item in enumerate(base_items):
        terminal = sorted(terminal_set(skeleton, rvar.rpath, base_item))
        if value_only:
            inner_data[i] = tuple(item[rvar.attr] for item in terminal)
        else:
            inner_data[i] = tuple((item, item[rvar.attr]) for item in terminal)
    return inner_data


class SkeletonDataInterface:
    """Fetches relational data as a table from given skeleton"""

    def __init__(self, skeleton: RelationalSkeleton, cache_maxsize=128, to_shuffle=True, skip_copy=False):
        self.skeleton = skeleton if skip_copy else ImmutableRSkeleton(skeleton)
        self.base_items = dict()
        for ic in sorted(self.skeleton.schema.item_classes):
            # shuffle sorted preserves reproducibility
            base_items = np.array(sorted(skeleton.items(ic)))
            if to_shuffle:
                np.random.shuffle(base_items)
            self.base_items[ic] = base_items

        # enable lru cache
        self.inner_flatten = functools.lru_cache(maxsize=cache_maxsize)(self.inner_flatten)

    def flatten(self, rvars, with_base_items=False, value_only=False):
        """Relational data as a table where columns corresponding to rvars. If with_base_items set True, base items are used as an index (the first column)"""
        rvars = tuple(rvars)
        assert len({rvar.base for rvar in rvars}) == 1, "Relational variables do not share the same base class: {}".format(rvars)
        base_class = rvars[0].base
        base_items = self.base_items[base_class]

        data = np.empty([len(base_items), (1 if with_base_items else 0) + len(rvars)], dtype=object)
        if with_base_items:
            data[:, 0] = base_items

        for j, rvar in enumerate(rvars, start=1 if with_base_items else 0):
            data[:, j] = self.inner_flatten(rvar, value_only)

        return data

    def inner_flatten(self, rvar, value_only=False, sort=True):
        base_items = self.base_items[rvar.base]
        skeleton = self.skeleton

        inner_data = np.zeros((len(base_items),), dtype=object)
        for i, base_item in enumerate(base_items):
            terminal = sorted(terminal_set(skeleton, rvar.rpath, base_item))
            if value_only:
                if sort:
                    inner_data[i] = tuple(sorted(item[rvar.attr] for item in terminal))
                else:
                    inner_data[i] = tuple(item[rvar.attr] for item in terminal)
            else:
                if sort:
                    inner_data[i] = tuple(sorted(((item, item[rvar.attr]) for item in terminal), key=lambda iv: iv[1]))
                else:
                    inner_data[i] = tuple((item, item[rvar.attr]) for item in terminal)
        return inner_data

    def fetch_singleton(self, rvar: RelationalVariable):
        assert rvar.is_canonical
        return np.array([next(iter(terminal_set(self.skeleton, rvar.rpath, b)))[rvar.attr] for b in self.base_items[rvar.base]])


class GroundGraph:
    """Ground graph"""

    def __init__(self, rcm: PRCM, skeleton: RelationalSkeleton):
        """

        Parameters
        ----------
        rcm
        skeleton
        """
        self.schema = skeleton.schema
        self.skeleton = skeleton
        self.rcm = rcm
        self.g = PDAG()

        def k_fun(d):
            return d.effect.rpath.base

        ItemAttribute = namedtuple('ItemAttribute', ['item', 'attr'])

        as_rdeps = {dep1 for dep1, _ in rcm.undirected_dependencies}
        for base_item_class, rdeps in group_by(as_rdeps, k_fun):
            for base_item, rdep in product(skeleton.items(base_item_class), rdeps):
                for dest_item in terminal_set(skeleton, rdep.cause.rpath, base_item):
                    self.g.add_undirected_edge(ItemAttribute(dest_item, rdep.cause.attr),
                                               ItemAttribute(base_item, rdep.effect.attr))

        for base_item_class, rdeps in group_by(rcm.directed_dependencies, k_fun):
            for base_item, rdep in product(skeleton.items(base_item_class), rdeps):
                for dest_item in terminal_set(skeleton, rdep.cause.rpath, base_item):
                    self.g.add_edge(ItemAttribute(dest_item, rdep.cause.attr),
                                    ItemAttribute(base_item, rdep.effect.attr))

        included = set(self.g.vertices())
        for skitem in self.skeleton.items():
            for attr in skitem.item_class.attrs:
                temp = ItemAttribute(skitem, attr)
                if temp not in included:
                    self.g.add_nodes([temp])

    def __str__(self):
        return str(self.g)

    def as_networkx_dag(self):
        return self.g.as_networkx_dag().copy()

    def adj(self, x, attr_type=None):
        if attr_type:
            assert isinstance(attr_type, AttributeClass)
            return set(filter(lambda y: y.attr == attr_type, self.g.adj(x)))
        return self.g.adj(x)

    def ne(self, x, attr_type=None):
        if attr_type:
            assert isinstance(attr_type, AttributeClass)
            return set(filter(lambda y: y.attr == attr_type, self.g.ne(x)))
        return self.g.ne(x)

    def pa(self, x, attr_type=None):
        if attr_type:
            assert isinstance(attr_type, AttributeClass)
            return set(filter(lambda y: y.attr == attr_type, self.g.pa(x)))
        return self.g.pa(x)

    def unshielded_triples(self):
        uts = set()
        for middle in sorted(self.g):
            for left, right in combinations(sorted(self.g.adj(middle)), 2):
                if not self.g.is_adj(left, right):
                    uts.add(SymTriple(left, middle, right))
        return uts


def generate_rpath(schema: RelationalSchema, base: ItemClass = None, length=None):
    """Generate a random relational path from the given base, if provided.

    Notes
    -----
    The length is the upper bound of the generated relational path.
    """
    assert length is None or 1 <= length
    if base is None:
        base = choice(sorted(schema.entities | schema.relationships))
    assert base in schema

    rpath_inner = [base, ]
    curr_item = base
    prev_item = None
    while len(rpath_inner) < length:
        next_items = set(schema.relateds(curr_item))
        if prev_item is not None:
            if curr_item.is_relationship_class or not prev_item.is_many(curr_item):
                next_items.remove(prev_item)
        if not next_items:
            break
        next_item = choice(sorted(next_items))
        rpath_inner.append(next_item)

        prev_item = curr_item
        curr_item = next_item

    return RelationalPath(rpath_inner, True)


def generate_rcm(schema: RelationalSchema, num_dependencies=10, max_degree=5, max_hop=6):
    """Generate a random relational causal model."""

    FAILED_LIMIT = len(schema.entities) + len(schema.relationships)
    # ordered attributes
    attr_order = sorted(schema.attrs)
    shuffle(attr_order)

    def causable(cause_attr_candidate):
        return attr_order.index(cause_attr_candidate) < attr_order.index(effect_attr)

    # schema may not be a single component
    rcm = RCM(schema)

    for effect_attr in attr_order:
        base_class = schema.item_class_of(effect_attr)
        effect = RelationalVariable(RelationalPath(base_class), effect_attr)

        degree = randint(1, max_degree + 1)  # 1<= <= max_degree

        failed_count = 0
        while len(rcm.pa(effect)) < degree and failed_count < FAILED_LIMIT:
            rpath = generate_rpath(schema, base_class, randint(1, max_hop + 1 + 1))
            cause_attr_candidates = list(filter(causable, rpath.terminal.attrs - {effect_attr, }))
            if not cause_attr_candidates:
                failed_count += 1
                continue

            cause_attr = choice(sorted(cause_attr_candidates))
            cause = RelationalVariable(rpath, cause_attr)
            candidate = RelationalDependency(cause, effect)
            if candidate not in rcm.directed_dependencies:
                rcm.add(candidate)
                failed_count = 0
            else:
                failed_count += 1

    if len(rcm.directed_dependencies) > num_dependencies:
        return RCM(schema, choice(sorted(rcm.directed_dependencies), num_dependencies).tolist())
    return rcm


def _item_attributes(items, attr: AttributeClass):
    return {(item, attr) for item in items}


def generate_values_for_skeleton(rcm: ParamRCM, skeleton: RelationalSkeleton):
    """
    Generate values for the given skeleton based on functions specified in the parametrized RCM.

    Parameters
    ----------
    rcm : ParamRCM
        a parameterized RCM, where its functions are used to generate values on skeleton.
    skeleton : RelationalSkeleton
        a skeleton where values will be assigned to its item-attributes
    """
    cdg = rcm.class_dependency_graph
    nx_cdg = cdg.as_networkx_dag()
    ordered_attributes = nx.topological_sort(nx_cdg)
    ordered_attributes += list(set(rcm.schema.attrs) - set(ordered_attributes))

    for attr in ordered_attributes:
        base_item_class = rcm.schema.item_class_of(attr)
        effect = RelationalVariable(RelationalPath(base_item_class), attr)
        causes = rcm.pa(effect)

        for base_item in sorted(skeleton.items(base_item_class)):
            cause_item_attrs = {cause: _item_attributes(terminal_set(skeleton, cause.rpath, base_item), cause.attr)
                                for cause in causes}

            v = rcm.functions[effect](skeleton, cause_item_attrs)
            skeleton[(base_item, attr)] = v


def normalize_skeleton(skeleton: RelationalSkeleton):
    """Normalize (mean 0, standard deviation 1) each attribute in the given skeleton."""
    schema = skeleton.schema
    A = schema.attrs

    crvs = {RelationalVariable(RelationalPath(schema.item_class_of(attr)), attr) for attr in A}
    for crv in crvs:
        data = flatten(skeleton, (crv,), with_base_items=False, value_only=True)
        data = [d[0] for d in data[:, 0]]
        # if len(data) > 0 and data[0] is None:
        #     continue
        mu = np.mean(data)
        std = np.std(data)
        for item in skeleton.items(crv.rpath.terminal):
            item[crv.attr] = (item[crv.attr] - mu) / std


def linear_gaussians_rcm(rcm: RCM):
    """Parameterized RCM as a linear model with Gaussian additive noise."""
    functions = dict()
    effects = {RelationalVariable(RelationalPath(rcm.schema.item_class_of(attr)), attr) for attr in rcm.schema.attrs}

    for e in effects:
        parameters = {cause: 1.0 + 0.1 * abs(randn()) for cause in rcm.pa(e)}
        functions[e] = linear_gaussian(parameters, average_agg(), normal_sampler(0, 0.1))

    return ParamRCM(rcm.schema, rcm.directed_dependencies, functions)


def __validate_item_class(base_item_class, schema):
    if not isinstance(base_item_class, ItemClass):
        raise TypeError('{} is not a valid item class.'.format(type(base_item_class)))
    if base_item_class not in schema:
        raise ValueError('{} is not in the relational schema.'.format(base_item_class))


def __validate_hop(hop):
    if 0 > hop:
        raise ValueError('Hop must be a non-negative integer. Received {}'.format(hop))


def enumerate_rpaths(schema: RelationalSchema, hop, base_item_class=None):
    """Returns a generator that enumerates all valid relational paths up to the given hop."""
    __validate_hop(hop)
    if base_item_class:
        __validate_item_class(base_item_class, schema)

    Ps = deque()
    if base_item_class is not None:
        Ps.append(RelationalPath(base_item_class))
    else:
        Ps.extend((RelationalPath(ic) for ic in schema.item_classes))

    while Ps:
        P = Ps.pop()
        yield P
        if P.hop_len < hop:
            Ps.extend(filter(lambda x: x is not None, (P.appended_or_none(i) for i in schema.relateds(P.terminal))))


def enumerate_rvars(schema: RelationalSchema, hop):
    """Returns a generator that enumerates all valid relational variables up to the given hop."""

    __validate_hop(hop)

    for base_item_class in schema.item_classes:
        for P in enumerate_rpaths(schema, hop, base_item_class):
            for attr in P.terminal.attrs:
                yield RelationalVariable(P, attr)


class interner(dict):
    def __missing__(self, key):
        self[key] = key
        return key


def enumerate_rdeps(schema: RelationalSchema, hop):
    """Returns a generator that enumerates all valid relational dependencies up to the given hop."""
    __validate_hop(hop)

    c = interner()
    for base_item_class in schema.item_classes:
        if not base_item_class.attrs:
            continue
        for P in enumerate_rpaths(schema, hop, base_item_class):
            for cause_attr in P.terminal.attrs:
                for effect_attr in base_item_class.attrs:
                    if effect_attr != cause_attr:
                        yield RelationalDependency(c[RelationalVariable(P, cause_attr)], c[RelationalVariable(base_item_class, effect_attr)])
