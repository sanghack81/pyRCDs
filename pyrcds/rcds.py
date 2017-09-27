import functools
import itertools
import logging
from collections import deque, defaultdict
from itertools import takewhile, count, combinations

import networkx as nx

from pyrcds.domain import RelationalSkeleton, RelationalSchema, SkItem
from pyrcds.graphs import PDAG
from pyrcds.model import RelationalDependency, PRCM, llrsp, RelationalVariable, eqint, RelationalPath, RCM, UndirectedRDep, SymTriple, enumerate_rvars, \
    enumerate_rdeps
from pyrcds.utils import group_by, unions, safe_iter


class CIQuery:
    """Conditional independence query"""

    def __init__(self, x, y, zs):
        self.pair = frozenset({x, y})
        self.conds = frozenset(zs)

    def __eq__(self, other):
        return isinstance(other, CIQuery) and self.pair == other.pair and self.conds == other.conds

    def __hash__(self):
        return hash(self.pair) ^ hash(self.conds)

    def __str__(self):
        x, y = sorted(self.pair)
        zs = ', '.join([str(z) for z in sorted(self.conds)])
        return str(x) + " _||_ " + str(y) + " | {" + zs + "}"


class CITestResult:
    """Conditional independence test result"""

    def __init__(self, query, ci: bool, p=None):
        self.query = query
        self.ci = bool(ci)
        self.p = p

    def __hash__(self):
        return hash(self.query) ^ hash(self.ci) ^ hash(self.p)

    def __bool__(self):
        return self.ci

    def __str__(self):
        return "{}: {} with p={:.4f}".format(self.query, "independent" if self.ci else "dependent", self.p)


class CITester:
    """Abstract class for conditional independence tester"""

    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def ci_test(self, x: RelationalVariable, y: RelationalVariable, zs=tuple(), **options) -> CITestResult:
        raise NotImplementedError()

    @property
    def is_p_value_available(self):
        raise NotImplementedError()


class interner(dict):
    def __missing__(self, key):
        self[key] = key
        return key


def __validate_rpath(Q):
    if not isinstance(Q, RelationalPath):
        raise TypeError('RPath is expected. {} is given.'.format(type(Q)))


def extend(P: RelationalPath, Q: RelationalPath):
    """See [1] for the description of extend

    References
    ----------
    [1] Marc Maier, Causal Discovery for Relational Domains: Representation, Reasoning, and Learning
        Ph.D. Dissertation, University of Massachusetts, Amherst
    """
    # __validate_rpath(P)
    # __validate_rpath(Q)
    if P.terminal != Q.base:
        raise ValueError('The terminal of {} must be the same as the base of {}'.format(P, Q))

    m, n = len(P), len(Q)
    for pivot in takewhile(lambda piv: P[m - 1 - piv] == Q[piv], range(min(m, n))):
        if P[:m - 1 - pivot].joinable(Q[pivot:]):  # double?
            yield P[:m - 1 - pivot] ** Q[pivot:]


def intersectible(P: RelationalPath, Q: RelationalPath):
    """See [1,2] for the description of intersectible

    References
    ----------
    [1] Marc Maier (2014),
        Causal Discovery for Relational Domains:
        Representation, Reasoning, and Learning,
        Ph.D. Dissertation, University of Massachusetts, Amherst
    [2] Sanghack Lee and Vasant Honavar (2015),
        Lifted Representation of Relational Causal Models Revisited:
        Implications for Reasoning and Structure Learning,
        In Proceedings of Workshop on Advances in Causal Inference co-located with UAI 2015
    """
    __validate_rpath(P)
    __validate_rpath(Q)
    if P == Q:
        raise ValueError('{} == {}'.format(P, Q))

    return P.base == Q.base and P.terminal == Q.terminal and llrsp(P, Q) + llrsp(reversed(P), reversed(Q)) <= min(
        len(P), len(Q))


def co_intersectible(Q: RelationalPath, R: RelationalPath, P: RelationalPath, P_prime: RelationalPath):
    """See [1] for the description of co-intersectible.

    References
    ----------
    [1] Sanghack Lee and Vasant Honavar (2015),
        Lifted Representation of Relational Causal Models Revisited:
        Implications for Reasoning and Structure Learning,
        In Proceedings of Workshop on Advances in Causal Inference co-located with UAI 2015
    """

    __validate_rpath(Q)
    __validate_rpath(R)
    __validate_rpath(P)
    __validate_rpath(P_prime)
    check = (Q.terminal == R.base and
             Q.base == P.base == P_prime.base and
             R.terminal == P.terminal == P_prime.terminal and
             intersectible(P, P_prime))
    if not check:
        raise ValueError('not a valid arguments: {}'.format([Q, R, P, P_prime]))

    ll = 1 + (len(Q) + len(R) - 1 - len(P)) // 2
    Qm, Rp = Q[:len(Q) - ll], R[ll - 1:]
    l1, l2 = llrsp(Q, P_prime), llrsp(reversed(R), reversed(P_prime))
    return (l1 < len(Qm) or l2 < len(Rp)) and l1 + l2 <= len(P_prime)


class UnvisitedQueue:
    """A queue that accept only previously un-queued items."""

    def __init__(self, iterable=()):
        self.visited = set(iterable)
        self.queue = deque(self.visited)

    def put(self, x):
        if x not in self.visited:
            self.visited.add(x)
            self.queue.append(x)

    def puts(self, xs):
        for x in xs:
            self.put(x)

    def __len__(self):
        return len(self.queue)

    def pop(self):
        return self.queue.popleft()

    def __bool__(self):
        return bool(self.queue)


def d_separated(dag: nx.DiGraph, x, y, zs=frozenset()):
    """A simple implementation of d-separation. """
    assert x != y
    assert x not in zs and y not in zs

    qq = UnvisitedQueue(((x, '>'), (x, '<')))
    while qq:
        node, direction = qq.pop()
        if direction == '>':
            if node not in zs:
                qq.puts((ch, '>') for ch in dag.successors_iter(node))
            else:
                qq.puts((pa, '<') for pa in dag.predecessors_iter(node))

        else:  # '<'
            if node not in zs:
                qq.puts((ch, '>') for ch in dag.successors_iter(node))
                qq.puts((pa, '<') for pa in dag.predecessors_iter(node))

        if {(y, '>'), (y, '<')} & qq.visited:
            return False

    return True


class AbstractGroundGraph(CITester):
    """Revised Abstract Ground Graph(s) taking intersectibility and co-intersectibility described in [1].


    References
    ----------
    [1] Sanghack Lee and Vasant Honavar (2015),
        Lifted Representation of Relational Causal Models Revisited:
        Implications for Reasoning and Structure Learning,
        In Proceedings of Workshop on Advances in Causal Inference co-located with UAI 2015

    Notes
    -----
    AGGs does not correctly reason about the relational d-separation.
    """

    def ci_test_data(self, x, y, zs, attrs, base_items, query_name='') -> CITestResult:
        raise NotImplementedError('AGG does not handle real-data.')

    def __init__(self, rcm: RCM, h: int):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
        c1 = interner()  # memory-saver, takes time...
        c2 = interner()  # memory-saver, takes time...
        #
        self.RVs = set(c1[rv] for rv in enumerate_rvars(rcm.schema, h))
        #
        self.RVEs = set()
        self.IVs = set()
        self.IVEs = set()
        # IVs
        self.combined = defaultdict(set)
        self.extend = defaultdict(set)
        for rv in self.RVs:
            self.extend[rv].add(rv)
        for _, rvs in group_by(self.RVs, lambda rv: (rv.rpath.base, rv.attr)):
            for rv1, rv2 in combinations(rvs, 2):
                if intersectible(rv1.rpath, rv2.rpath):
                    iv = c2[frozenset((rv1, rv2))]
                    self.IVs.add(iv)
                    self.combined[rv1.rpath].add(rv2.rpath)
                    self.combined[rv2.rpath].add(rv1.rpath)

                    self.extend[rv1].add(iv)
                    self.extend[rv2].add(iv)

                    if not (len(self.IVs) % 1000):
                        self.logger.info('creating {} of IVs.'.format(len(self.IVs)))
        # RVEs and IVEs
        for Y, Qys in group_by(self.RVs, lambda rv: rv.attr):
            for RxVy in filter(lambda d: d.effect.attr == Y, rcm.directed_dependencies):
                for Qy in Qys:
                    Q, (R, X) = Qy.rpath, RxVy.cause
                    for P in filter(lambda p: p.hop_len <= h, extend(Q, R)):
                        Px = c1[RelationalVariable(P, X)]
                        self.RVEs.add((Px, Qy))  # P.X --> Q.Y

                        if not (len(self.RVEs) % 1000):
                            self.logger.info('creating {} of RVEs.'.format(len(self.RVEs)))

                        # Q, R, P, P_prime
                        for P_prime in self.combined[P]:
                            if co_intersectible(Q, R, P, P_prime):
                                iv = c2[frozenset((Px, c1[RelationalVariable(P_prime, X)]))]
                                self.IVEs.add((iv, Qy))

                                if not (len(self.IVEs) % 1000):
                                    self.logger.info('creating {} of IVEs.'.format(len(self.IVEs)))
                        # P, reversed(R), Q, Q_prime
                        for Q_prime in self.combined[Q]:
                            if co_intersectible(P, reversed(R), Q, Q_prime):
                                iv = c2[frozenset((Qy, c1[RelationalVariable(Q_prime, Y)]))]
                                self.IVEs.add((Px, iv))

                                if not (len(self.IVEs) % 1000):
                                    self.logger.info('creating {} of IVEs.'.format(len(self.IVEs)))

        self.RVs = frozenset(self.RVs)
        self.RVEs = frozenset(self.RVEs)
        self.IVs = frozenset(self.IVs)
        self.IVEs = frozenset(self.IVEs)

        self.agg = nx.DiGraph()
        self.agg.add_nodes_from(self.RVs)
        self.agg.add_nodes_from(self.IVs)
        self.agg.add_edges_from(self.RVEs)
        self.agg.add_edges_from(self.IVEs)

        self.ci_test = functools.lru_cache(maxsize=None)(self.ci_test)

    def ci_test(self, x: RelationalVariable, y: RelationalVariable, zs=frozenset(), **options):
        assert x != y
        assert x not in zs and y not in zs
        assert len({x.base, y.base} | {z.base for z in zs}) == 1
        assert y.is_canonical

        query = CIQuery(x, y, zs)
        zs_bar = unions(self.extend[z] for z in zs)
        x_bar = self.extend[x] - zs_bar  # {x} | self.combined[x]
        y_bar = self.extend[y] - zs_bar  # {y} | self.combined[y]

        if x_bar & y_bar:
            return CITestResult(query, False)
        # all disjoint
        for x_ in x_bar:
            for y_ in y_bar:
                if not d_separated(self.agg, x_, y_, zs_bar):
                    return CITestResult(query, False)
        return CITestResult(query, True)

    @property
    def is_p_value_available(self):
        return False


class AbstractRCD:
    """Abstract class for RCD-related causal discovery algorithm"""

    def __init__(self, schema, h_max, ci_tester, verbose=False):
        # self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

        self.schema = schema
        self.h_max = h_max
        self.ci_tester = ci_tester

        self.sepset = defaultdict(lambda: None)
        self.prcm = PRCM(schema)
        self.verbose = verbose

    def ci_test(self, cause: RelationalVariable, effect: RelationalVariable, conds, size: int):
        assert 0 <= size and effect.is_canonical

        for cond in combinations(conds, size):
            if self.verbose:
                print('ci testing: {} _||_ {} | {}'.format(cause, effect, cond))
            ci_result = self.ci_tester.ci_test(cause, effect, cond)
            if bool(ci_result):
                if self.verbose:
                    if self.ci_tester.is_p_value_available:
                        print('p={}\t{} _||_ {} | {}'.format(ci_result.p, cause, effect, cond))
                    else:
                        print('p=unknown\t{} _||_ {} | {}'.format(cause, effect, cond))
                self.sepset[frozenset((cause, effect))] = set(cond)
                return True
            else:
                if self.verbose:
                    if self.ci_tester.is_p_value_available:
                        print('p={}\t{} _NOT||_ {} | {}'.format(ci_result.p, cause, effect, cond))
                    else:
                        print('p=unknown\t{} _NOT||_ {} | {}'.format(cause, effect, cond))
        if self.verbose:
            print('NOT {} _||_ {} | {} with {}-subsets'.format(cause, effect, conds, size))
        return False

    def phase_I(self, truth: RCM = None):
        """Find adjacencies of the underlying RCM."""
        if self.verbose:
            print('phase I: started.')
        prcm, schema, ci_tester = self.prcm, self.schema, self.ci_tester

        # Initialize an undirected RCM
        udeps_to_be_tested = set(UndirectedRDep(dep) for dep in enumerate_rdeps(self.schema, self.h_max))
        prcm.add(udeps_to_be_tested)
        deps_to_be_tested = {dep for udep in udeps_to_be_tested for dep in udep}

        for d in count():
            if self.verbose:
                print('phase I: checking depth: {}'.format(d))
            to_remove = set()
            for dep in safe_iter(deps_to_be_tested):
                if self.verbose:
                    print('phase I: checking: {}'.format(dep))
                cause, effect = dep
                assert len(prcm.ne(effect)) - 1 >= d
                if self.ci_test(cause, effect, prcm.ne(effect) - {cause}, d):
                    if truth and cause in truth.adj(effect):
                        if self.verbose:
                            print('False negative: {}'.format(dep))
                    else:
                        deps_to_be_tested -= {dep, reversed(dep)}
                        to_remove.add(UndirectedRDep(dep))
                else:
                    if truth and len(truth.pa(effect)) == d and cause not in truth.pa(effect) and \
                            cause.attr not in truth.class_dependency_graph.de(effect.attr):
                        if self.verbose:
                            print('False positive: {}'.format(dep))
                        deps_to_be_tested -= {dep, reversed(dep)}
                        to_remove.add(UndirectedRDep(dep))

            # post clean up
            prcm.remove(to_remove)
            deps_to_be_tested = set(filter(lambda dep: len(prcm.ne(dep.effect)) - 1 >= d + 1, deps_to_be_tested))
            if not deps_to_be_tested:
                break

        if self.verbose:
            print('phase I: finished.')
        return self.prcm

    def find_sepset(self, Vx, Rz):
        """Find a separating set only from the neighbors of canonical relational variable (i.e., Vx)"""
        assert Vx.is_canonical
        pair_key = frozenset((Vx, Rz))
        if self.sepset[pair_key] is not None:
            return self.sepset[pair_key]

        candidates = list(self.prcm.adj(Vx) - {Rz})
        for size in range(len(candidates) + 1):
            if self.ci_test(Rz, Vx, candidates, size):
                return self.sepset[pair_key]
        return None

    def phase_II(self):
        raise NotImplementedError()


def sound_rules(g: PDAG, non_colliders=(), purge=True):
    """Orient edges in the given PDAG where non-colliders information may not be completed."""
    while True:
        mark = len(g.oriented())
        for non_collider in tuple(non_colliders):
            x, y, z = non_collider
            # R1 X-->Y--Z (shielded, and unshielded)
            if g.is_oriented_as(x, y) and g.is_unoriented(y, z):
                g.orient(y, z)
            # R1' Z-->Y--X (shielded, and unshielded)
            if g.is_oriented_as(z, y) and g.is_unoriented(y, x):
                g.orient(y, x)

            # R3 (do not check x--z)
            for w in g.ch(x) & g.adj(y) & g.ch(z):
                if g.is_unoriented(w, y):
                    g.orient(y, w)

            # R4   z-->w-->x
            if g.pa(x) & g.adj(y) & g.ch(z):
                if g.is_unoriented(x, y):
                    g.orient(y, x)
            # R4'   x-->w-->z
            if g.pa(z) & g.adj(y) & g.ch(x):
                if g.is_unoriented(y, z):
                    g.orient(y, z)

            if {x, z} <= g.ne(y):
                if z in g.ch(x):
                    g.orient(y, z)
                if x in g.ch(z):
                    g.orient(y, x)

        # R2
        for y in g:
            # x -> y ->z and x -- z
            for x in g.pa(y):
                for z in g.ch(y) & g.ne(x):
                    g.orient(x, z)

        if purge:
            for non_collider in list(non_colliders):
                x, y, z = non_collider
                if (not (g.ne(y) & {x, z})) or ({x, z} & g.ch(y)):
                    non_colliders.discard(non_collider)

        if len(g.oriented()) == mark:
            break


def completes(g: PDAG, non_colliders):
    """Maximally orients edges in the given PDAG with non-collider constraints"""
    U = unions({(x, y), (y, x)} for x, y in g.unoriented())

    # filter out directions, which violates non-collider constraints.
    # Nothing will be changed if sound rules (R1) are applied to g before 'completes' is called.
    for x, y in safe_iter(U):
        # x-->y
        if any(SymTriple(x, y, z) in non_colliders for z in g.pa(y)):
            g.orient(y, x)
            U -= {(x, y), (y, x)}

    for x, y in safe_iter(U):
        h = g.copy()
        h.orient(x, y)
        if ext(h, non_colliders):
            U -= g.oriented()
        else:
            g.orient(y, x)
            U -= {(x, y), (y, x)}


def ext(g: PDAG, NC):
    """Extensibility where non-colliders are completely identified."""
    h = g.copy()
    while h:
        for y in h:
            if not h.ch(y) and all(SymTriple(x, y, z) not in NC for x, z in combinations(h.adj(y), 2)):
                for x in h.ne(y):
                    g.orient(x, y)
                h.remove_vertex(y)
                break
        else:
            return False
    return True


class RpCD(AbstractRCD):
    """RpCD as in [1]

    References
    ----------
    [1] Sanghack Lee, Vasant Honavar, (2016)
        A Characterization of Markov Equivalence Classes of Relational Causal Model under Path Semantics
        In Proceedings of the Thirty-Second Conference on Uncertainty in Artificial Intelligence.
    """

    def __init__(self, schema, h_max, ci_tester, verbose=False):
        super().__init__(schema, h_max, ci_tester, verbose)

    def enumerate_CUTs(self):
        """Enumerate CUTs whose attribute classes are distinct."""
        by_cause_attr = defaultdict(set)
        by_effect_attr = defaultdict(set)
        for ud in self.prcm.undirected_dependencies:
            for d in ud:
                by_cause_attr[d.cause.attr].add(d)
                by_effect_attr[d.effect.attr].add(d)

        done = set()
        for Y in self.schema.attrs:
            for QzVy in by_effect_attr[Y]:
                for PyVx in by_cause_attr[Y]:
                    X, Z = PyVx.effect.attr, QzVy.cause.attr
                    # distinct only
                    if (X, Y, Z) not in done:
                        cut = canonical_unshielded_triples(self.prcm, PyVx, QzVy)
                        if cut is not None:
                            yield cut
                            done.add((X, Y, Z))  # ordered triple

    def phase_II(self, background_knowledge=()):
        """Maximally-orient undirected dependencies"""

        if self.verbose:
            print('phase II: started.')
        pcdg = self.prcm.class_dependency_graph

        if background_knowledge:
            for edge in background_knowledge:
                pcdg.orient(*edge)
            sound_rules(pcdg, set())

        NC = set()
        for Vx, PPy, Rz in self.enumerate_CUTs():
            X, Y, Z = Vx.attr, next(iter(PPy)).attr, Rz.attr

            # inactive non-collider / fully-oriented / already in non-colliders
            if ({X, Z} & pcdg.ch(Y)) or (not ({X, Z} & pcdg.ne(Y))) or (SymTriple(X, Y, Z) in NC):
                continue
            #
            sepset = self.find_sepset(Vx, Rz)
            if sepset is not None:  # can be checked with dual-RUT
                if not (PPy & sepset):
                    pcdg.orient(X, Y)
                    pcdg.orient(Z, Y)
                else:
                    if X == Z:  # RBO
                        pcdg.orient(Y, X)
                    else:
                        NC.add(SymTriple(X, Y, Z))
            sound_rules(pcdg, NC)

        completes(pcdg, NC)

        for x, y in pcdg.oriented():
            self.prcm.orient_with(x, y)

        if self.verbose:
            print('phase II: finished.')
        return self.prcm


def joinable(p, *args):
    for q in args:
        p = p ** q
        if p is None:
            return False
    return True


def restore_anchors(P, Q, a_r, b_r, a_s, b_s, a_t=None, b_t=None):
    """Given characteristic anchors, construct a fully-specified set of anchors."""
    last_P, first_Q = len(P) - 1, 0  # for readability
    assert a_s <= a_r
    assert b_s <= b_r
    if a_t is not None or b_t is not None:
        assert a_t is not None and b_t is not None
        assert a_s <= a_r <= a_t
        assert b_t <= b_s <= b_r

    # (...|P|-1, 0...)
    # (a_r..., ...b_r) (...a_r, ...b_r)
    # (a_s..., b_s...) (a_s..., ...b_s)
    J = {(last_P - i, first_Q + i) for i in range(llrsp(P[::-1], Q))} | \
        {(a_r + i, b_r - i) for i in range(llrsp(P[a_r:], Q[:b_r:-1]))} | \
        {(a_r - i, b_r - i) for i in range(llrsp(P[:a_r:-1], Q[:b_r:-1]))} | \
        {(a_s + i, b_s + i) for i in range(llrsp(P[a_s:], Q[b_s:]))} | \
        {(a_s + i, b_s - i) for i in range(llrsp(P[a_s:], Q[:b_s:-1]))}

    if a_t is not None and b_t is not None:
        # (a_t..., ...b_t) (...a_t, ...b_t)
        J |= {(a_t + i, b_t - i) for i in range(llrsp(P[a_t:], Q[:b_t:-1]))} | \
             {(a_t - i, b_t - i) for i in range(llrsp(P[:a_t:-1], Q[:b_t:-1]))}

    return J


def anchors_to_skeleton(schema: RelationalSchema, P: RelationalPath, Q: RelationalPath, J):
    """Given anchors, construct a relational skeleton, which admits the anchors.

    Notes
    -----
    The resulting name of items will be subject to change.
    """
    temp_g = nx.Graph()

    # Both
    pp = [None] * len(P)
    qq = [None] * len(Q)
    for a, b in J:
        pp[a] = qq[b] = SkItem('p' + str(a) + 'q' + str(b), P[a])
        temp_g.add_node(pp[a])

    # P only
    aa, bb = zip(*J)
    for a in set(range(len(P))) - set(aa):
        pp[a] = SkItem('p' + str(a), P[a])
        temp_g.add_node(pp[a])
    # Q only
    for b in set(range(len(Q))) - set(bb):
        qq[b] = SkItem('q' + str(b), Q[b])
        temp_g.add_node(qq[b])

    temp_g.add_edges_from(list(zip(pp, pp[1:])))
    temp_g.add_edges_from(list(zip(qq, qq[1:])))

    all_auxs = set()
    cc = count()
    for v in list(temp_g.nodes()):
        if v.item_class.is_relationship_class:
            missing = set(v.item_class.entities) - {ne.item_class for ne in temp_g.neighbors(v)}
            auxs = {SkItem('aux' + str(next(cc)), ic) for ic in missing}
            all_auxs |= auxs
            temp_g.add_nodes_from(auxs)
            for aux in auxs:
                temp_g.add_edge(v, aux)

    skeleton = RelationalSkeleton(schema, True)
    entities = list(filter(lambda vv: vv.item_class.is_entity_class, temp_g.nodes()))
    relationships = list(filter(lambda vv: vv.item_class.is_relationship_class, temp_g.nodes()))
    skeleton.add_entities(*entities)
    for r in relationships:
        skeleton.add_relationship(r, set(temp_g.neighbors(r)))

    return skeleton, pp, qq, all_auxs


# written for readability
# can be faster by employing a view-class for RPath for the slicing operator
def canonical_unshielded_triples(M: PRCM, PyVx: RelationalDependency = None, QzVy: RelationalDependency = None, single=True, with_anchors=False):
    """Returns a CUT, if exists, or generate CUTs with/without anchors

    Parameters
    ----------
    with_anchors
    QzVy
    PyVx
    M
    single : bool
        whether only a CUT (if exists) is generated per a triple of attribute classes (e.g., (X,Y,Z))

    Notes
    -----
    If `single` is set and relational dependencies are given, a CUT is given if exists else None.
    If `single` is set and relational dependencies are not given, a list of CUTs is given, which can be empty.
    If `single` is not set, a generator of CUTs (of given relational dependencies, if provided) will be given.
    """
    if PyVx is None or QzVy is None:
        assert PyVx is None and QzVy is None
        all_ds = {dd for d in M.directed_dependencies for dd in (d, reversed(d))} | \
                 {d for u in M.undirected_dependencies for d in u}
        to_chain = []
        skip = set()
        for PyVx in all_ds:
            (_, Y), (_, X) = PyVx
            for QzVy in all_ds:
                (_, Z), (_, Y2) = QzVy
                if Y == Y2:
                    if single:  # single(s)
                        if (X, Y, Z) in skip:
                            continue
                        for cut in inner_canonical_unshielded_triples(M, PyVx, QzVy, with_anchors):
                            skip.add((X, Y, Z))
                            to_chain.append(cut)
                            break
                    else:
                        to_chain.append(inner_canonical_unshielded_triples(M, PyVx, QzVy, with_anchors))
        if single:
            return to_chain
        else:
            return itertools.chain(*to_chain)

    if single:
        # returns the first cut (will return None if no such CUT exists)
        for cut in inner_canonical_unshielded_triples(M, PyVx, QzVy, with_anchors):
            return cut
    else:
        # returns a generator
        return inner_canonical_unshielded_triples(M, PyVx, QzVy, with_anchors)


def inner_canonical_unshielded_triples(M: PRCM, PyVx: RelationalDependency, QzVy: RelationalDependency, with_anchors=False):
    """A generator that yields CUTs sufficient to identify the pattern of an RCM under path semantics."""
    LL = llrsp

    Py, Vx = PyVx
    Qz, Vy = QzVy
    P, Y = Py
    Q, Z = Qz
    V, Y2 = Vy

    if Y != Y2:
        raise ValueError("{} and {} do not share the common attribute class.".format(PyVx, QzVy))

    m, n = len(P), len(Q)
    l = LL(reversed(P), Q)
    a_x, b_x = m - l, l - 1

    # A set of candidate anchors
    J = set()
    for a in range(a_x + 1):  # 0 <= <= a_x
        for b in range(b_x, n):  # b_x <=  <= |Q|
            if P[a] == Q[b]:
                J.add((a, b))

    # the first characteristic anchor (a_r,b_r)
    for a_r, b_r in J:
        if not (LL(P[:a_r:-1], Q[b_r:]) == LL(P[a_r:], Q[b_r:]) == 1):
            continue
        if not joinable(P[:a_r], Q[b_r:]):
            continue
        RrZ = RelationalVariable(P[:a_r] ** Q[b_r:], Z)
        if RrZ in M.adj(Vx) or RrZ == Vx:
            continue

        l_alpha = LL(Q[b_x:b_r:-1], P[:a_r:-1])
        if l_alpha == 1:
            if eqint(P[a_r:a_x], Q[b_x:b_r:-1]):
                assert Vx != RrZ
                cut = (Vx, frozenset({Py, RelationalVariable(P[:a_r] ** Q[:b_r:-1], Y)}), RrZ)
                if with_anchors:
                    yield cut, restore_anchors(P, Q, a_r, b_r, a_r, b_r)
                else:
                    yield cut

        elif l_alpha < b_r - b_x + 1 and a_r < a_x and b_x < b_r:
            a_y, b_y = a_r - l_alpha + 1, b_r - l_alpha + 1

            # the second characteristic anchor
            for a_s, b_s in J:
                if not (a_s <= a_y and b_x < b_s <= b_y):
                    continue
                if not joinable(P[:a_s], Q[b_s:]):
                    continue
                RsZ = RelationalVariable(P[:a_s] ** Q[b_s:], Z)
                if RsZ in M.adj(Vx):
                    continue

                PA, PB, QA, QB = P[:a_s:-1], P[a_s:a_y], Q[b_s:b_y], Q[b_x:b_s:-1]

                if LL(PA, QA) > 1 or LL(PA, QB) > 1:
                    continue

                l_beta = LL(PB, QB)
                if (not eqint(PB, QA)) or (l_beta > 1 and l_beta == min(len(PB), len(QB))):
                    continue
                assert eqint(PB, QA) or 1 < l_beta < min(len(PB), len(QB))

                a_z, b_z = a_s + l_beta - 1, b_s - l_beta + 1
                # the third characteristic anchor
                for a_t, b_t in J:
                    if not (a_r < a_t <= a_x and b_x <= b_t < b_z):
                        continue
                    if not joinable(P[:a_s], Q[b_t:b_s:-1], P[a_r:a_t:-1], Q[b_r:]):
                        continue
                    RtZ = RelationalVariable(P[:a_s] ** Q[b_t:b_s:-1] ** P[a_r:a_t:-1] ** Q[b_r:], Z)
                    if RtZ in M.adj(Vx):
                        continue

                    PC, PD, QC, QD = P[a_r:a_t:-1], P[a_t:a_x], Q[b_t:b_z], Q[b_x:b_t:-1]

                    if LL(PC, QC) > 1 or LL(PD, QC) > 1:
                        continue

                    l_gamma = LL(PC, QD)
                    assert 1 <= l_gamma
                    if l_gamma == 1 and eqint(PD, QD) or 1 < l_gamma < min(len(PC),
                                                                           len(QD)) and a_t < a_x and b_x < b_t:
                        a_w, b_w = a_t - l_gamma + 1, b_t - l_gamma + 1

                        PP = {P,
                              P[:a_w] ** Q[:b_w:-1],
                              P[:a_s] ** Q[:b_s:-1],
                              P[:a_s] ** Q[b_t:b_s:-1] ** P[a_t:],
                              P[:a_s] ** Q[b_s:b_r] ** P[a_r:],
                              P[:a_s] ** Q[b_s:b_r] ** P[a_r:a_w] ** Q[:b_w:-1]}

                        PP_Y = frozenset({RelationalVariable(PP_i, Y) for PP_i in PP})

                        if with_anchors:
                            JJ = restore_anchors(P, Q, a_r, b_r, a_s, b_s, a_t, b_t)
                            for R in {RrZ, RsZ, RtZ}:
                                assert Vx != R
                                yield (Vx, PP_Y, R), JJ
                        else:
                            for R in {RrZ, RsZ, RtZ}:
                                assert Vx != R
                                yield Vx, PP_Y, R


def markov_equivalence(model: RCM, verbose=False) -> PRCM:
    """A unique representation for Markov Equivalence of RCM under path semantics [1]

    References
    ----------
    [1] Sanghack Lee, Vasant Honavar, (2016)
        A Characterization of Markov Equivalence Classes of Relational Causal Model under Path Semantics
        In Proceedings of the Thirty-Second Conference on Uncertainty in Artificial Intelligence.
    """

    cdg = model.class_dependency_graph

    # undirected
    cprcm = PRCM(model.schema, {UndirectedRDep(d) for d in model.directed_dependencies})

    # pattern of RCM
    NC = set()
    for cut in canonical_unshielded_triples(model):
        Vx, PPy, Rz = cut
        Py = next(iter(PPy))
        (_, X), (_, Y), (_, Z) = Vx, Py, Rz

        # not so efficient
        if cprcm.class_dependency_graph.is_oriented_as(X, Y) and cprcm.class_dependency_graph.is_oriented_as(Z, Y):
            continue
        if cprcm.class_dependency_graph.is_oriented_as(Y, X) or cprcm.class_dependency_graph.is_oriented_as(Y, Z):
            continue
        if SymTriple(X, Y, Z) in NC:
            continue

        # canonical unshielded colliders
        if cdg.is_oriented_as(X, Y) and cdg.is_oriented_as(Z, Y):
            cprcm.orient_with(X, Y)
            cprcm.orient_with(Z, Y)
            if verbose:
                print("CUT: {}".format(cut))
                print("orienting {} -> {} <- {}".format(X, Y, Z))
        elif X == Z:  # RBO
            cprcm.orient_with(Y, X)
            if verbose:
                print("CUT: {}".format(cut))
                print("orienting {} <- {} -> {}".format(X, Y, X))
        else:
            # canonical unshielded non-colliders
            NC.add(SymTriple(X, Y, Z))
            if verbose:
                print("CUT: {}".format(cut))
                print("as a non-collider {} - {} - {}".format(X, Y, Z))

    # CPRCM
    pcdg = cprcm.class_dependency_graph
    sound_rules(pcdg, NC)
    completes(pcdg, NC)

    for x, y in pcdg.oriented():
        cprcm.orient_with(x, y)

    return cprcm


def markov_equivalent(model1: RCM, model2: RCM):
    """Two RCMs under path semantics are equivalent based on Markov condition."""
    return markov_equivalence(model1) == markov_equivalence(model2)


def new_extend(P: RelationalPath, Q: RelationalPath, with_anchors=False):
    return list(new_extend_iter(P, Q, with_anchors))


def new_extend_iter(P: RelationalPath, Q: RelationalPath, with_anchors=False):
    """Relationship after joining P and Q, from the perspective of item class of X except the item class of X itself"""
    LL = llrsp

    m, n = len(P), len(Q)
    l = LL(reversed(P), Q)
    a_x, b_x = m - l, l - 1

    # A set of candidate anchors
    J = set()
    for a in range(a_x + 1):  # 0 <= <= a_x
        for b in range(b_x, n):  # b_x <=  <= |Q|
            if P[a] == Q[b]:
                J.add((a, b))

    # the first characteristic anchor (a_r,b_r)
    for a_r, b_r in J:
        if not (LL(P[:a_r:-1], Q[b_r:]) == LL(P[a_r:], Q[b_r:]) == 1):
            continue
        if not joinable(P[:a_r], Q[b_r:]):
            continue
        Rr = P[:a_r] ** Q[b_r:]

        l_alpha = LL(Q[b_x:b_r:-1], P[:a_r:-1])
        if l_alpha == 1:
            if eqint(P[a_r:a_x], Q[b_x:b_r:-1]):
                yield Rr if not with_anchors else (Rr, frozenset(restore_anchors(P, Q, a_r, b_r, a_r, b_r)))

        elif l_alpha < b_r - b_x + 1 and a_r < a_x and b_x < b_r:
            a_y, b_y = a_r - l_alpha + 1, b_r - l_alpha + 1

            # the second characteristic anchor
            for a_s, b_s in J:
                if not (a_s <= a_y and b_x < b_s <= b_y):
                    continue
                if not joinable(P[:a_s], Q[b_s:]):
                    continue
                Rs = P[:a_s] ** Q[b_s:]

                PA, PB, QA, QB = P[:a_s:-1], P[a_s:a_y], Q[b_s:b_y], Q[b_x:b_s:-1]

                if LL(PA, QA) > 1 or LL(PA, QB) > 1:
                    continue

                l_beta = LL(PB, QB)
                if (not eqint(PB, QA)) or (l_beta > 1 and l_beta == min(len(PB), len(QB))):
                    continue
                assert eqint(PB, QA) or 1 < l_beta < min(len(PB), len(QB))

                a_z, b_z = a_s + l_beta - 1, b_s - l_beta + 1
                # the third characteristic anchor
                for a_t, b_t in J:
                    if not (a_r < a_t <= a_x and b_x <= b_t < b_z):
                        continue
                    if not joinable(P[:a_s], Q[b_t:b_s:-1], P[a_r:a_t:-1], Q[b_r:]):
                        continue
                    Rt = P[:a_s] ** Q[b_t:b_s:-1] ** P[a_r:a_t:-1] ** Q[b_r:]
                    for R in {Rr, Rs, Rt}:
                        yield R if not with_anchors else (R, frozenset(restore_anchors(P, Q, a_r, b_r, a_s, b_s, a_t, b_t)))
