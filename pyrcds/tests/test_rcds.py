from itertools import combinations

import numpy as np

from pyrcds.domain import generate_schema, RelationalSchema, Cardinality, RelationshipClass, EntityClass
from pyrcds.graphs import PDAG
from pyrcds.model import generate_rcm, RelationalPath, RelationalVariable, SymTriple, UndirectedRDep, is_valid_rpath, enumerate_rpaths, \
    enumerate_rdeps, enumerate_rvars
from pyrcds.rcds import interner, extend, UnvisitedQueue, AbstractGroundGraph, sound_rules, completes, \
    d_separated, \
    RpCD, markov_equivalence, CITester, CITestResult, new_extend, intersectable, co_intersectable, CIQuery
from pyrcds.tests.testing_utils import company_rcm, company_schema, EPBDF


def test_something():
    schema = company_schema()
    rcm = company_rcm()

    rpaths = set(enumerate_rpaths(schema, 4))
    assert len(rpaths) == 43
    for rpath in enumerate_rpaths(schema, 4):
        assert is_valid_rpath(list(rpath))

    assert rcm.directed_dependencies <= set(enumerate_rdeps(schema, rcm.max_hop))
    assert 22 == len(set(enumerate_rdeps(schema, 4)))
    assert 162 == len(set(enumerate_rdeps(schema, 16)))
    assert 22 == len(set(enumerate_rdeps(schema, 4)))
    assert 162 == len(set(enumerate_rdeps(schema, 16)))


def test_enumerate_rpaths():
    M = company_rcm()
    assert {d.cause.rpath for d in M.directed_dependencies} <= set(enumerate_rpaths(M.schema, M.max_hop))

    schema = company_schema()
    assert len(set(enumerate_rpaths(schema, 0))) == len(schema.item_classes)

    expected = iter([5, 13, 22, 32, 43, 55, 69, 85, 103, 123])
    for i in range(10):
        x = list(enumerate_rpaths(schema, i))
        y = set(x)
        assert len(x) == len(y)
        assert len(y) == next(expected)


def test_enumerate_rvars():
    schema = company_schema()
    for i in range(10):
        l = list(enumerate_rvars(schema, i))
        s = set(l)
        assert len(s) == len(l)
        assert len(l) == sum(
            len(p.terminal.attrs) for p in enumerate_rpaths(schema, i))


def test_interner():
    xx = interner()
    x = (1, 2, 3)
    y = (1, 2, 3)
    assert id(x) != id(y)
    z1 = xx[x]
    z2 = xx[y]
    assert id(z1) == id(z2) == id(x)


def test_enumerate_rdeps():
    schema = company_schema()
    rcm = company_rcm()
    assert rcm.directed_dependencies <= set(enumerate_rdeps(schema, rcm.max_hop))
    expected = [4, 4, 12, 12, 22, 22, 34, 34, 48, 48]
    for h in range(10):
        assert expected[h] == len(set(enumerate_rdeps(schema, h)))


def test_extend():
    E, P, B, D, F = EPBDF()
    actual = set(extend(RelationalPath([E, D, P, F, B]), RelationalPath([B, F, P, D, E])))
    expected = {RelationalPath([E, D, P, F, B, F, P, D, E]),
                RelationalPath([E, D, P, D, E]),
                RelationalPath([E])}
    assert actual == expected
    actual = set(extend(RelationalPath([E, D, P, F, B, F, P]), RelationalPath([P, F, B])))
    expected = {RelationalPath([E, D, P, F, B]), }
    assert actual == expected


def test_intersectible():
    E, P, B, D, F = EPBDF()

    assert intersectable(RelationalPath([E, D, P, F, B, F, P, D, E]), RelationalPath([E, D, P, D, E]))
    assert not intersectable(RelationalPath([E, D, P, F, B, F, P]), RelationalPath([E, D, P, D, E]))
    assert not intersectable(RelationalPath([D, P, F, B, F, P]), RelationalPath([E, D, P, D, E]))
    assert not intersectable(RelationalPath([E]), RelationalPath([E, D, P, D, E]))


def test_UnvisitedQueue():
    q = UnvisitedQueue()
    q.puts([1, 2, 3, 4, 1, 2, 3])
    assert len(q) == 4
    while q:
        q.pop()
    assert len(q) == 0
    q.puts([1, 2, 3, 4, 1, 2, 3])
    assert len(q) == 0


def test_d_separated():
    g = PDAG()
    g.add_path([1, 2, 3])
    g.add_path([5, 4, 3])
    g.add_path([3, 6, 7, 8])
    g = g.as_networkx_dag()
    assert d_separated(g, 1, 5)
    assert not d_separated(g, 1, 2)
    assert not d_separated(g, 1, 3)
    assert not d_separated(g, 2, 3)
    assert not d_separated(g, 5, 3)
    assert not d_separated(g, 3, 1)
    assert d_separated(g, 1, 5)
    assert d_separated(g, 1, 4)
    assert d_separated(g, 2, 4)
    assert not d_separated(g, 2, 4, {3})
    assert not d_separated(g, 1, 5, {3})
    assert not d_separated(g, 1, 4, {3})
    assert not d_separated(g, 2, 4, {6})
    assert not d_separated(g, 1, 5, {7})
    assert not d_separated(g, 1, 4, {8})
    assert not d_separated(g, 1, 8)
    assert not d_separated(g, 5, 8)
    assert d_separated(g, 1, 8, {3})
    assert d_separated(g, 1, 8, {3, 5})
    assert d_separated(g, 1, 8, {3, 6})
    assert not d_separated(g, 1, 3, {8})
    assert not d_separated(g, 1, 3, {8, 6})


def test_company_agg_size():
    rcm = company_rcm()
    agg2h = AbstractGroundGraph(rcm, 2 * rcm.max_hop)
    assert agg2h.agg.number_of_nodes() == 158
    # assert agg2h.agg.number_of_edges() == 444
    assert agg2h.agg.number_of_edges() == 487  # new agg


def test_company_fig_4_4_maier():
    rcm = company_rcm()
    agg2h = AbstractGroundGraph(rcm, 2 * rcm.max_hop)
    # Fig 4.4 in Maier's Thesis
    E, P, B, D, F = EPBDF()
    rvs = [RelationalVariable([E, D, P, F, B, F, P], "Success"),
           RelationalVariable(E, "Competence"),
           RelationalVariable([E, D, P], "Success"),
           RelationalVariable([E, D, P, F, B], "Revenue"),
           RelationalVariable([E, D, P, D, E], "Competence"),
           RelationalVariable([E, D, P, D, E, D, P], "Success")]

    iv = frozenset((rvs[-1], rvs[0]))
    vs = set(rvs) | {iv}
    sub = agg2h.agg.subgraph(list(vs))
    # assert len(sub.edges()) == 7
    assert len(sub.edges()) == 10  # new agg now has 10 not 7
    assert {(rvs[0], rvs[3]),
            (rvs[1], rvs[2]),
            (rvs[2], rvs[3]),
            (rvs[4], rvs[2]),
            (rvs[4], iv),
            (rvs[4], rvs[5]),
            (iv, rvs[3]),
            # newly added three edges, due to path semantics
            (rvs[1], rvs[0]),
            (rvs[1], iv),
            (rvs[1], rvs[5])} == set(sub.edges())


def test_sound_rules():
    g = PDAG()
    g.add_undirected_path([1, 2, 3, 4, 5])
    sound_rules(g)
    assert not g.oriented()
    assert len(g.unoriented()) == 4

    g.orient(1, 2)
    nc = {SymTriple(1, 2, 3)}
    sound_rules(g, nc)
    assert g.oriented() == {(1, 2), (2, 3)}


def test_evidence_together():
    for _ in range(100):
        g = PDAG()
        vs = np.random.permutation(np.random.randint(20) + 1)
        for x, y in combinations(vs, 2):
            if np.random.rand() < 0.2:
                g.add_edge(x, y)

        # unshielded colliders
        nc = set()
        for y in vs:
            for x, z in combinations(g.pa(y), 2):
                if not g.is_adj(x, z):
                    nc.add(SymTriple(x, y, z))

        flag = bool(np.random.randint(2))
        if flag:
            sound_rules(g, nc, bool(np.random.randint(2)))
        else:
            completes(g, nc)
        current = g.oriented()
        if not flag:
            sound_rules(g, nc, bool(np.random.randint(2)))
        else:
            completes(g, nc)
        post = g.oriented()
        assert current == post


def test_evidence_completes():
    for _ in range(100):
        g = PDAG()
        vs = np.random.permutation(np.random.randint(20) + 1)
        for x, y in combinations(vs, 2):
            if np.random.rand() < 0.2:
                g.add_edge(x, y)

        # unshielded colliders
        nc = set()
        for y in vs:
            for x, z in combinations(g.pa(y), 2):
                if not g.is_adj(x, z):
                    nc.add(SymTriple(x, y, z))

        completes(g, nc)
        current = g.oriented()
        sound_rules(g, nc, bool(np.random.randint(2)))
        post = g.oriented()
        assert current == post


def test_company():
    rcm = company_rcm()
    agg = AbstractGroundGraph(rcm, rcm.max_hop * 2)
    rpcd = RpCD(rcm.schema, rcm.max_hop, agg)
    rpcd.phase_I()
    assert rpcd.prcm.undirected_dependencies == {UndirectedRDep(d) for d in rcm.directed_dependencies}
    rpcd.phase_II()
    assert rpcd.prcm.directed_dependencies == rcm.directed_dependencies

    assert markov_equivalence(rcm).directed_dependencies == rpcd.prcm.directed_dependencies


def test_company_markov_equivalence():
    rcm = company_rcm()
    markov_equivalence(rcm, verbose=True)


def test_company_fixer():
    rcm = company_rcm()

    class DumbCITester(CITester):
        def ci_test_data(self, x, y, zs, attrs, base_items, query_name='') -> CITestResult:
            raise NotImplementedError()

        def ci_test(self, x: RelationalVariable, y: RelationalVariable, zs=tuple(), **options) -> CITestResult:
            query = CIQuery(x, y, zs)
            return CITestResult(query, np.random.randint(10) == 0)

        @property
        def is_p_value_available(self):
            return False

    rpcd = RpCD(rcm.schema, rcm.max_hop, DumbCITester(), verbose=True)
    rpcd.phase_I(rcm)
    print(rpcd.prcm.undirected_dependencies)
    assert rpcd.prcm.undirected_dependencies == {UndirectedRDep(d) for d in rcm.directed_dependencies}


def test_rpcd_markov_equivalence():
    """An empirical test for Markov equivalence"""
    np.random.seed(0)
    for _ in range(3):
        schema = generate_schema()
        rcm = generate_rcm(schema, max_hop=2)
        agg = AbstractGroundGraph(rcm, rcm.max_hop * 2)
        rpcd = RpCD(rcm.schema, rcm.max_hop, agg)
        rpcd.phase_I()
        to_uds = {UndirectedRDep(d) for d in rcm.directed_dependencies}
        phase_i_uds = rpcd.prcm.undirected_dependencies
        assert phase_i_uds == to_uds
        rpcd.phase_II()
        assert markov_equivalence(rcm).directed_dependencies == rpcd.prcm.directed_dependencies


def test_extend_example():
    print()
    E1 = EntityClass('E1')
    E2 = EntityClass('E2')
    E3 = EntityClass('E3')
    E4 = EntityClass('E4')
    Ra = RelationshipClass('Ra', set(), {E1: Cardinality.many, E2: Cardinality.many})
    Rb = RelationshipClass('Rb', set(), {E2: Cardinality.many, E3: Cardinality.many})
    Rc = RelationshipClass('Rc', set(), {E2: Cardinality.many, E4: Cardinality.many})

    P = RelationalPath([E1, Ra, E2, Rb, E3])
    Q = RelationalPath([E3, Rb, E2, Rc, E4])
    extend1 = RelationalPath([E1, Ra, E2, Rb, E3, Rb, E2, Rc, E4])
    extend3 = RelationalPath([E1, Ra, E2, Rc, E4])
    assert {extend1, extend3} == set(extend(P, Q))

    print()
    Rb = RelationshipClass('Rb', set(), {E2: Cardinality.many, E3: Cardinality.one})
    P = RelationalPath([E1, Ra, E2, Rb, E3])
    Q = RelationalPath([E3, Rb, E2, Rc, E4])
    assert {extend3} == set(extend(P, Q))


def test_co_intersectable_example():
    print()
    Ij, Ik, B, E1, E2, E3 = entities = [EntityClass(name) for name in ['Ij', 'Ik', 'B', 'E1', 'E2', 'E3']]
    R1 = RelationshipClass('R1', set(), {B: Cardinality.one, E1: Cardinality.one})
    R2 = RelationshipClass('R2', set(), {E1: Cardinality.one, E3: Cardinality.one})
    R3 = RelationshipClass('R3', set(), {E1: Cardinality.one, E2: Cardinality.one})
    R4 = RelationshipClass('R4', set(), {E2: Cardinality.one, E3: Cardinality.one, Ik: Cardinality.one})
    R5 = RelationshipClass('R5', set(), {Ik: Cardinality.one, Ij: Cardinality.one})

    Q = RelationalPath([B, R1, E1, R2, E3, R4, Ik, R5, Ij])
    R = RelationalPath([Ij, R5, Ik, R4, E3, R2, E1, R3, E2, R4, Ik])
    P = RelationalPath([B, R1, E1, R3, E2, R4, Ik])
    P_prime = RelationalPath([B, R1, E1, R2, E3, R4, Ik])
    schema = RelationalSchema(entities, [R1, R2, R3, R4, R5])
    assert P in set(extend(Q, R))
    assert P in set(new_extend(Q, R))
    assert intersectable(P, P_prime)
    assert not co_intersectable(Q, R, P, P_prime, schema)
