from itertools import combinations

import numpy as np

from pyrcds.domain import generate_schema, cardinality_sampler, RelationshipClass, Cardinality, EntityClass, RelationalSchema, AttributeClass
from pyrcds.model import enumerate_rpaths, RelationalPath, RelationalVariable, RelationalDependency, RCM
from pyrcds.rcds import new_extend, intersectible, co_intersectible, extend, AbstractGroundGraph


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


def test_extend_newextend_subsume_evidence():
    for _ in range(100):
        schema = generate_schema()
        ics = list(schema.item_classes)
        base = ics[np.random.randint(len(ics))]
        rpaths_of_base = list(enumerate_rpaths(schema, np.random.randint(6), base))
        if len(rpaths_of_base) > 50:
            rpaths_of_base = np.random.choice(rpaths_of_base, 50, replace=False)
        for Q, R in combinations(rpaths_of_base, 2):
            s = set(extend(reversed(Q), R))
            set1 = set(new_extend(reversed(Q), R))
            assert s <= set1


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
    assert intersectible(P, P_prime)
    assert not co_intersectible(Q, R, P, P_prime, schema)


def test_co_intersectable_working():
    for _ in range(30):
        print(_)
        schema = generate_schema(cardinality_distr=cardinality_sampler(0.5))
        ics = list(schema.item_classes)
        if len(ics) < 3:
            continue
        ic_P, ic_R, ic_terminal = np.random.choice(ics, 3)
        rpaths = list(enumerate_rpaths(schema, 8))
        P_candidates = [rpath for rpath in rpaths if rpath.base == ic_P and rpath.terminal == ic_terminal]
        Q_candidates = [rpath for rpath in rpaths if rpath.base == ic_P and rpath.terminal == ic_R]
        R_candidates = [rpath for rpath in rpaths if rpath.base == ic_R and rpath.terminal == ic_terminal]
        np.random.shuffle(P_candidates)
        np.random.shuffle(Q_candidates)
        np.random.shuffle(R_candidates)

        p_pairs = set()
        combis = list(combinations(P_candidates, 2))
        np.random.shuffle(combis)
        for P, P_prime in combis:
            if intersectible(P, P_prime):
                p_pairs.add((P, P_prime))
                if len(p_pairs) > 5:
                    break

        for Q in Q_candidates[:5]:
            for R in R_candidates[:5]:
                QR_extended = list(new_extend(Q, R))
                np.random.shuffle(QR_extended)
                for P in QR_extended[:5]:
                    np.random.shuffle(P_candidates)
                    passed = 0
                    for P_prime in P_candidates:
                        if P == P_prime:
                            continue
                        if intersectible(P, P_prime):
                            co_intersectible(Q, R, P, P_prime, schema)
                            passed += 1
                        if passed > 5:
                            break


def test_agg_non_completeness():
    """ Check the non-completeness of revised AGG for relational d-separation """
    X, Y, Z = [AttributeClass(_) for _ in ('X', 'Y', 'Z')]
    E1 = EntityClass('E1')
    E2 = EntityClass('E2', Y)
    E3 = EntityClass('E3', X)
    E4 = EntityClass('E4')
    E5 = EntityClass('E5', Z)
    R1 = RelationshipClass('R1', set(), {e: Cardinality.one for e in {E1, E2, E4}})
    R2 = RelationshipClass('R2', set(), {e: Cardinality.one for e in {E2, E3}})
    R3 = RelationshipClass('R3', set(), {e: Cardinality.one for e in {E3, E4, E5}})

    schema = RelationalSchema([E1, E2, E3, E4, E5], [R1, R2, R3])

    D1 = RelationalPath([E2, R2, E3, R3, E4, R1, E2, R2, E3])
    D2 = RelationalPath([E2, R2, E3, R3, E5])
    P = RelationalPath([E1, R1, E2, R2, E3])
    Q = RelationalPath([E1, R1, E4, R3, E3, R2, E2])
    S = RelationalPath([E1, R1, E4, R3, E5])
    S_prime = RelationalPath([E1, R1, E2, R2, E3, R3, E5])

    assert intersectible(S, S_prime)
    assert co_intersectible(Q, D2, S, S_prime)

    P_X = RelationalVariable(P, X)
    S_Z = RelationalVariable(S, Z)
    S_prime_Z = RelationalVariable(S_prime, Z)
    Q_Y = RelationalVariable(Q, Y)

    dep_D1 = RelationalDependency(RelationalVariable(D1, X), RelationalVariable(E2, Y))
    dep_D2 = RelationalDependency(RelationalVariable(D2, Z), RelationalVariable(E2, Y))
    DD = {dep_D1, dep_D2}

    rcm = RCM(schema, DD)

    agg = AbstractGroundGraph(rcm, h=10)
    assert (P_X, Q_Y) in agg.RVEs
    assert (frozenset((S_Z, S_prime_Z)), Q_Y) in agg.IVEs
    assert not agg.ci_test(P_X, S_prime_Z, frozenset({Q_Y}))
