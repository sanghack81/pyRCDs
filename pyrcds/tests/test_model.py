from typing import Union, List

import numpy as np
import pytest
from joblib import Parallel, delayed
from networkx import is_directed_acyclic_graph
from tqdm import trange

from pyrcds.domain import AttributeClass, RelationalSkeleton, SkItem, generate_schema, generate_skeleton, EntityClass, RelationshipClass, Cardinality, RelationalSchema
from pyrcds.model import RelationalPath, llrsp, RelationalVariable, terminal_set, PRCM, UndirectedRDep, RCM, GroundGraph, generate_rcm, \
    canonical_rvars, linear_gaussians_rcm, generate_values_for_skeleton, flatten, generate_rpath, is_valid_rpath, RelationalDependency
from pyrcds.rcds import intersectable
from pyrcds.tests.testing_utils import EPBDF, company_schema, company_rcm, company_skeleton, company_deps
from pyrcds.utils import between_sampler


def test_rpath_0():
    with pytest.raises(Exception):
        E, P, B, D, F = EPBDF()
        RelationalPath([E, D, E])


def test_rpath_1():
    with pytest.raises(Exception):
        E, P, B, D, F = EPBDF()
        RelationalPath([E, E])


def test_rpath_2():
    with pytest.raises(Exception):
        E, P, B, D, F = EPBDF()
        RelationalPath([D, D])


def test_rpath_3():
    with pytest.raises(Exception):
        E, P, B, D, F = EPBDF()
        RelationalPath([E, D, P, F, B, F, P, F])


def test_rpath_6():
    with pytest.raises(Exception):
        E, P, B, D, F = EPBDF()
        RelationalPath([E, P, F, B])


def test_rpath_5():
    with pytest.raises(Exception):
        E, P, B, D, F = EPBDF()
        RelationalPath([E, F, B])


def test_rpath_4():
    E, P, B, D, F = EPBDF()
    RelationalPath([E])
    RelationalPath(E)
    RelationalPath([E, D])
    RelationalPath([E, D, P])
    RelationalPath([E, D, P, D])
    RelationalPath([E, D, P, D, E])
    RelationalPath([E, D, P, F, B])
    RelationalPath([E, D, P, F, B, F])
    RelationalPath([E, D, P, F, B, F, P])
    RelationalPath([E, D, P, F, B, F, P, D])
    RelationalPath([E, D, P, F, B, F, P, D, E])
    assert RelationalPath([E, D, P, F, B, F, P, D, E]).joinable(RelationalPath([E, D, P]))
    assert RelationalPath([E, D, P, F, B, F, P, D, E]).join(RelationalPath([E, D, P])) == RelationalPath([E, D, P, F, B, F, P, D, E, D, P])
    assert RelationalPath([E, D, P, F, B, F, P, D, E]).joinable(RelationalPath(E))
    assert RelationalPath(E).joinable(RelationalPath(E))
    assert not RelationalPath(E).joinable(RelationalPath(D))
    assert RelationalPath(E).join(RelationalPath(E)) == RelationalPath(E)
    assert not RelationalPath([E, D, P, F, B, F, P]).joinable(RelationalPath([P, F, B]))
    RelationalPath([D])
    RelationalPath(D)
    assert RelationalPath(D).is_canonical
    RelationalPath([D, P])
    assert not RelationalPath([D, P]).is_canonical
    RelationalPath([D, P, D])
    RelationalPath([D, P, D, E])
    assert E == RelationalPath([D, P, D, E]).terminal
    assert D == RelationalPath([D, P, D, E]).base
    assert P == RelationalPath([D, P, D, E])[1]
    assert RelationalPath([D, P, D, E]).reverse() == RelationalPath(tuple(reversed([D, P, D, E]))) == RelationalPath([E, D, P, D])
    RelationalPath([D, P, F, B])
    RelationalPath([D, P, F, B, F])
    RelationalPath([D, P, F, B, F, P])
    assert RelationalPath([D, P, F, B, F, P])[2:5] == RelationalPath((F, B, F, P))
    assert RelationalPath([D, P, F, B, F, P]).subpath(2, 5) == RelationalPath([F, B, F])
    RelationalPath([D, P, F, B, F, P, D])
    RelationalPath([D, P, F, B, F, P, D, E])
    assert not RelationalPath([D, P, F, B, F, P, D, E]).is_canonical
    assert RelationalPath(D) == RelationalPath([D, ])
    assert len({RelationalPath([D, P, F, B, F, P, D, E]), RelationalPath([D, P, F, B, F, P, D]), RelationalPath([D, P, F, B, F, P]),
                RelationalPath([D, P, F, B, F, P])}) == 3
    assert all(i == j for i, j in zip(RelationalPath([D, P, F, B, F, P, D, E]), [D, P, F, B, F, P, D, E]))
    assert llrsp(RelationalPath([P, F, B]), RelationalPath([P, F, B])) == 3
    assert llrsp(RelationalPath([E, D, P, F, B]), RelationalPath([E, D, P, D, E])) == 1
    assert llrsp(RelationalPath([P, F, B, F, P]), RelationalPath([P, F, B, F, P])) == 3
    assert intersectable(RelationalPath([P, F, B, F, P]), RelationalPath([P, F, B, F, P]))
    assert intersectable(RelationalPath([E, D, P, D, E]), RelationalPath([E, D, P, D, E, D, P, D, E]))
    assert intersectable(RelationalPath([E, D, P, D, E]), RelationalPath([E, D, P, D, E, D, P, D, E]))
    assert RelationalVariable(RelationalPath([E, D, P, D, E]), AttributeClass('Salary')) == RelationalVariable(RelationalPath([E, D, P, D, E]), AttributeClass('Salary'))
    assert not RelationalVariable(RelationalPath([E, D, P, D, E]), AttributeClass('Salary')).is_canonical
    assert RelationalVariable(RelationalPath(E), AttributeClass('Salary')).is_canonical
    assert len(RelationalVariable(RelationalPath(E), AttributeClass('Salary'))) == 1
    assert len(RelationalVariable(RelationalPath([E, D, P, D, E]), AttributeClass('Salary'))) == 5
    assert len(
        {RelationalVariable(RelationalPath([E, D, P, D, E]), AttributeClass('Salary')), RelationalVariable(RelationalPath([E, D, P, D, E]), AttributeClass('Salary'))}) == 1


def test_skeleton():
    E, P, B, D, F = EPBDF()
    entities = ['Paul', 'Roger', 'Quinn', 'Sally', 'Thomas',
                'Case', 'Adapter', 'Laptop', 'Tablet', 'Smartphone',
                'Accessories', 'Devices']

    entity_types = {'Paul': E, 'Roger': E, 'Quinn': E, 'Sally': E, 'Thomas': E,
                    'Case': P, 'Adapter': P, 'Laptop': P, 'Tablet': P, 'Smartphone': P,
                    'Accessories': B, 'Devices': B}
    skeleton = RelationalSkeleton(company_schema(), True)
    p, r, q, s, t, c, a, l, ta, sm, ac, d = ents = tuple([SkItem(e, entity_types[e]) for e in entities])
    skeleton.add_entities(*ents)
    for emp, prods in ((p, {c, }), (q, {c, a, l}), (s, {l, ta}), (t, {sm, ta}), (r, {l, })):
        for prod in prods:
            skeleton.add_relationship(SkItem(emp.name + '-' + prod.name, D), {emp, prod})
    for biz, prods in ((ac, {c, a}), (d, {l, ta, sm})):
        for prod in prods:
            skeleton.add_relationship(SkItem(biz.name + '-' + prod.name, F), {biz, prod})

    assert terminal_set(skeleton, RelationalPath([E, D, P, F, B]), p) == {ac, }
    assert terminal_set(skeleton, RelationalPath([E, D, P, F, B]), q) == {ac, d}
    assert terminal_set(skeleton, RelationalPath([E, D, P, D, E]), r) == {q, s}
    assert terminal_set(skeleton, RelationalPath([E, D, P, D, E, D, P]), r) == {c, a, ta}

    assert terminal_set(skeleton, RelationalPath([E, D, P, F, B]), p, 'bridge-burning') == {ac, }
    assert terminal_set(skeleton, RelationalPath([E, D, P, F, B]), q, 'bridge-burning') == {ac, d}
    assert terminal_set(skeleton, RelationalPath([E, D, P, D, E]), r, 'bridge-burning') == {q, s}
    assert terminal_set(skeleton, RelationalPath([E, D, P, D, E, D, P]), r, 'bridge-burning') == {c, a, ta}

    assert terminal_set(skeleton, RelationalPath([E, D, P, D, E]), q) == {p, r, s}
    assert terminal_set(skeleton, RelationalPath([E, D, P, D, E]), q, 'bridge-burning') == {p, r, s}
    assert terminal_set(skeleton, RelationalPath([E, D, P, D, E, D, P, D, E]), q) == {t}
    assert terminal_set(skeleton, RelationalPath([E, D, P, D, E, D, P, D, E]), q, 'bridge-burning') == {t}

    assert terminal_set(skeleton, RelationalPath([P, D, E, D, P]), l) == {c, a, ta}
    assert terminal_set(skeleton, RelationalPath([P, D, E, D, P]), l, 'bridge-burning') == {c, a, ta}
    assert terminal_set(skeleton, RelationalPath([P, D, E, D, P, F, B, F, P]), l) == {sm, c, a}
    assert terminal_set(skeleton, RelationalPath([P, D, E, D, P, F, B, F, P]), l, 'bridge-burning') == {sm}


def test_sub():
    E, P, B, D, F = EPBDF()
    p = RelationalPath([E, D, P, F, B])
    assert p[:] == RelationalPath([E, D, P, F, B])
    assert p[:2] == RelationalPath([E, D, P])
    assert p[1:] == RelationalPath([D, P, F, B])
    assert p[1:2] == RelationalPath([D, P])
    assert p[::-1] == RelationalPath([B, F, P, D, E])
    assert p[:2:-1] == RelationalPath([P, D, E])
    assert p[1::-1] == RelationalPath([B, F, P, D])
    assert p[1:2:-1] == RelationalPath([P, D])


def test_rcm():
    deps = company_deps()
    rcm = company_rcm()
    urcm = PRCM(company_schema(), {UndirectedRDep(d) for d in rcm.directed_dependencies})
    skeleton = company_skeleton()

    gg = GroundGraph(rcm, skeleton)
    directed_uts = gg.unshielded_triples()
    assert len(directed_uts) == 57
    gg.as_networkx_dag()

    # TODO any test?

    ugg = GroundGraph(urcm, skeleton)
    undirected_uts = ugg.unshielded_triples()
    assert directed_uts == undirected_uts

    assert rcm.max_hop == 4

    urcm = PRCM(company_schema(), {UndirectedRDep(d) for d in deps})
    assert urcm.max_hop == 4
    ucdg = urcm.class_dependency_graph
    assert not ucdg.pa(AttributeClass('Salary'))
    assert not ucdg.ch(AttributeClass('Salary'))
    assert ucdg.ne(AttributeClass('Salary')) == {AttributeClass('Budget'), AttributeClass('Competence')}
    assert ucdg.adj(AttributeClass('Salary')) == {AttributeClass('Budget'), AttributeClass('Competence')}
    for d in deps:
        urcm.orient_as(d)
    cdg = urcm.class_dependency_graph
    assert cdg.adj(AttributeClass('Revenue')) == {AttributeClass('Success'), AttributeClass('Budget')}
    assert cdg.ne(AttributeClass('Revenue')) == set()
    assert cdg.pa(AttributeClass('Revenue')) == {AttributeClass('Success'), }
    assert cdg.ch(AttributeClass('Revenue')) == {AttributeClass('Budget'), }

    rcm = RCM(company_schema(), [])
    assert rcm.max_hop == -1


def test_crv():
    rvars = canonical_rvars(company_schema())
    E, P, B, D, F = EPBDF()
    assert rvars == {RelationalVariable(RelationalPath(E), 'Competence'), RelationalVariable(RelationalPath(E), 'Salary'), RelationalVariable(RelationalPath(B), 'Revenue'),
                     RelationalVariable(RelationalPath(B), 'Budget'), RelationalVariable(RelationalPath(P), 'Success')}


def test_rcm_orient_0():
    deps = company_deps()
    urcm = PRCM(company_schema(), {UndirectedRDep(d) for d in deps})
    for d in deps:
        urcm.orient_as(d)
    with pytest.raises(Exception):
        for d in deps:
            urcm.orient_as(reversed(d))


def test_rcm_orient_1():
    deps = company_deps()
    urcm = PRCM(company_schema(), {UndirectedRDep(d) for d in deps})
    with pytest.raises(Exception):
        urcm.add(deps)


def test_rcm_orient_2():
    deps = company_deps()
    rcm = company_rcm()
    with pytest.raises(Exception):
        rcm.add({UndirectedRDep(d) for d in deps})


def test_rcm_orient_3():
    deps = company_deps()
    rcm = company_rcm()
    with pytest.raises(Exception):
        rcm.add({reversed(d) for d in deps})


def test_to_code():
    print(company_rcm().to_code())
    BizUnit = EntityClass('BizUnit', (AttributeClass('Budget'), AttributeClass('Revenue')))
    Employee = EntityClass('Employee', (AttributeClass('Competence'), AttributeClass('Salary')))
    Product = EntityClass('Product', (AttributeClass('Success'),))
    Develops = RelationshipClass('Develops', (), {Employee: Cardinality.many, Product: Cardinality.many})
    Funds = RelationshipClass('Funds', (), {Product: Cardinality.one, BizUnit: Cardinality.many})
    entities = {BizUnit, Employee, Product}
    relationships = {Develops, Funds}
    schema = RelationalSchema(entities, relationships)

    rcm = RCM(schema, {RelationalDependency(RelationalVariable([Employee, Develops, Product, Funds, BizUnit], 'Budget'), RelationalVariable([Employee], 'Salary')),
                       RelationalDependency(RelationalVariable([Product, Develops, Employee], 'Competence'), RelationalVariable([Product], 'Success')),
                       RelationalDependency(RelationalVariable([Employee], 'Competence'), RelationalVariable([Employee], 'Salary')),
                       RelationalDependency(RelationalVariable([BizUnit, Funds, Product], 'Success'), RelationalVariable([BizUnit], 'Revenue')),
                       RelationalDependency(RelationalVariable([BizUnit], 'Revenue'), RelationalVariable([BizUnit], 'Budget'))})

    assert rcm.directed_dependencies == company_rcm().directed_dependencies


def test_rcm_gen():
    s1 = between_sampler(1, 10)
    s2 = between_sampler(1, 20)
    s3 = between_sampler(1, 5)
    for _ in trange(200):
        schema = generate_schema()
        max_hop = s1.sample()
        max_degree = s3.sample()
        num_dependencies = s2.sample()
        rcm = generate_rcm(schema, num_dependencies, max_degree, max_hop)
        assert rcm.max_hop <= max_hop
        assert len(rcm.directed_dependencies) <= num_dependencies
        assert not rcm.undirected_dependencies
        assert is_directed_acyclic_graph(rcm.class_dependency_graph.as_networkx_dag())
        effects = {e for c, e in rcm.directed_dependencies}
        assert all(len(rcm.pa(e)) <= max_degree for e in effects)


def test_generate_rpaths():
    for _ in range(100):
        schema = generate_schema()
        for __ in range(100):
            rpath = generate_rpath(schema, length=np.random.randint(1, 15))
            assert is_valid_rpath([i for i in rpath])


def random_seeds(n=None) -> Union[int, List[int]]:
    """Random seeds of given size or a random seed if n is None"""
    if n is None:
        return np.random.randint(np.iinfo(np.int32).max)
    else:
        return [np.random.randint(np.iinfo(np.int32).max) for _ in range(n)]


def test_data_gen():
    np.random.seed(0)
    n = 10
    Parallel(-1)(delayed(_test_gen_inner)(seed) for seed in random_seeds(n))


def _test_gen_inner(seed):
    np.random.seed(seed)
    schema = generate_schema()
    rcm = generate_rcm(schema, 20, 5, 4)
    skeleton = generate_skeleton(schema)
    lg_rcm = linear_gaussians_rcm(rcm)
    generate_values_for_skeleton(lg_rcm, skeleton)
    causes = {c for c, e in rcm.directed_dependencies}
    one_cause = next(iter(causes))
    causes_of_the_base = list(filter(lambda c: c.base == one_cause.base, causes))
    data_frame0 = flatten(skeleton, causes_of_the_base, False, False)
    data_frame1 = flatten(skeleton, causes_of_the_base, True, True)
    data_frame2 = flatten(skeleton, causes_of_the_base, False, True)
    data_frame3 = flatten(skeleton, causes_of_the_base, True, False)
    # base items
    assert np.all((np.equal(data_frame3[:, 0], data_frame1[:, 0])).ravel())
    assert np.all((np.equal(data_frame1[:, 1], data_frame2[:, 0])).ravel())
    assert np.all((np.equal(data_frame0[:, 0], data_frame3[:, 1])).ravel())
    assert sorted(sorted(val for item, val in ivs) for ivs in data_frame0[:, 0]) == sorted(
        sorted(v) for v in data_frame1[:, 1])
