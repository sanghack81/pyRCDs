import numpy as np
import pytest

from pyrcds.domain import SchemaElement, EntityClass, AttributeClass, Cardinality, RelationshipClass, RelationalSchema, generate_schema, \
    generate_skeleton, ImmutableRSkeleton, repeat_skeleton
from pyrcds.tests.testing_utils import company_skeleton, EPBDF, company_schema


def test_SchemaElement_fail_empty():
    with pytest.raises(Exception):
        SchemaElement('')


def test_SchemaElement():
    x, y = SchemaElement('asd'), SchemaElement('asd')
    assert len({x, y}) == 1
    assert x.name == 'asd'

    x, y = SchemaElement('asd'), SchemaElement('abc')
    assert len({x, y}) == 2


def test_item_classes():
    xx1 = EntityClass('x', ['y', 'z'])
    xx2 = EntityClass('x', 'y')
    xx3 = EntityClass('x', AttributeClass('y'))
    xx4 = EntityClass('x', [AttributeClass('y'), AttributeClass('z')])
    assert xx2 == xx3
    assert xx1 == xx4


def test_EDPDE():
    xx1 = EntityClass('x', ['y', 'z'])
    E, P, B, _, F = EPBDF()
    D = RelationshipClass('Develops', (AttributeClass('dummy1'), AttributeClass('dummy2')), {E: Cardinality.many, P: Cardinality.many})

    assert F[P] == Cardinality.one
    assert F[B] == Cardinality.many
    assert F.is_many(B)
    assert not F.is_many(P)
    assert P in F
    assert B in F
    assert P in D
    assert E in D
    assert E not in F
    assert D not in F
    assert F not in F
    schema = RelationalSchema({E, P, B}, {D, F})
    assert schema.item_class_of(AttributeClass('Salary')) == E
    assert schema.item_class_of(AttributeClass('Revenue')) == B
    assert AttributeClass('Salary') in schema
    assert E in schema
    assert xx1 not in schema
    assert AttributeClass('celery') not in schema
    assert str(schema) == 'RSchema(BizUnit, Develops, Employee, Funds, Product)'
    assert repr(
        schema) == "RSchema(Entity classes: [" \
                   "BizUnit(Budget, Revenue), " \
                   "Employee(Competence, Salary), " \
                   "Product(Success)" \
                   "], " \
                   "Relationship classes: [" \
                   "Develops(dummy1, dummy2, {Employee: many, Product: many}), " \
                   "Funds((), {BizUnit: many, Product: one})" \
                   "])"

    assert schema.relateds(B) == {F}
    assert schema.relateds(P) == {D, F}
    assert schema.relateds(E) == {D}


def test_skeleton():
    company_skeleton()


def test_to_code():
    print()
    print(company_schema().to_code())

    BizUnit = EntityClass('BizUnit', (AttributeClass('Budget'), AttributeClass('Revenue')))
    Employee = EntityClass('Employee', (AttributeClass('Competence'), AttributeClass('Salary')))
    Product = EntityClass('Product', (AttributeClass('Success'),))
    Develops = RelationshipClass('Develops', (), {Employee: Cardinality.many, Product: Cardinality.many})
    Funds = RelationshipClass('Funds', (), {Product: Cardinality.one, BizUnit: Cardinality.many})
    entities = {BizUnit, Employee, Product}
    relationships = {Develops, Funds}
    schema = RelationalSchema(entities, relationships)

    assert schema.to_code() == company_schema().to_code()


def test_skeleton_attribute():
    schema = generate_schema()
    skeleton = generate_skeleton(schema)
    skeleton = repeat_skeleton(skeleton, 2)
    for item in skeleton.items():
        skeleton[(item, 0)] = '1'
    for item in skeleton.items():
        for ne_item in skeleton.neighbors(item):
            assert skeleton[(ne_item, 0)] == '1'
            assert ne_item[0] == '1'


def test_skeleton_gen():
    for i in range(3):
        schema = generate_schema()
        skeleton = generate_skeleton(schema)
        iskeleton = ImmutableRSkeleton(skeleton)
        for R in schema.relationships:
            for E in R.entities:
                if not R.is_many(E):
                    ents = skeleton.items(E)
                    assert iskeleton.items(E) == ents
                    for e in ents:
                        must_be_one = skeleton.neighbors(e, R)
                        assert len(iskeleton.neighbors(e, R)) <= 1
                        assert len(must_be_one) <= 1
                        if 'key' not in e:
                            assert skeleton[(e, 'key')] is None
                        v = np.random.randint(100)
                        skeleton[(e, 'key')] = v
                        assert 'key' in e
                        assert v == skeleton[(e, 'key')]
                        assert v == iskeleton[(e, 'key')]
                        assert v == e['key']
