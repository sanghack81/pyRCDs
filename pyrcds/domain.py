import functools
import itertools
from collections import defaultdict, Counter
from enum import IntEnum
from functools import total_ordering
from itertools import chain
from typing import Set

import networkx as nx
from numpy.random.mtrand import random_sample, choice, randint

from pyrcds.utils import between_sampler


class Cardinality(IntEnum):
    """Define constants for how many relationships an entity can participate in"""
    one = 1
    many = 2

    def __str__(self):
        return type(self).__name__ + '.' + self.name

    def __repr__(self):
        return type(self).__name__ + '.' + self.name

    @staticmethod
    def from_int(v: int):
        if v == 1:
            return Cardinality.one
        elif v == 2:
            return Cardinality.many
        raise ValueError('unknown Cardinality: {}'.format(v))


def _names(ys) -> Set[str]:
    return {y.name for y in ys}


def _is_unique(ys) -> bool:
    return len(set(ys)) == len(ys)


@total_ordering
class SchemaElement:
    """An abstract class for item classes and attribute classes"""

    def __init__(self, name: str):
        if not isinstance(name, str) or not name:
            raise ValueError('A name must be a non-empty string')

        self.name = name
        self.__h = hash(self.name)

    def __hash__(self):
        return self.__h

    def __eq__(self, other):
        return isinstance(other, SchemaElement) and self.name == other.name

    def __str__(self):
        return self.name

    def __lt__(self, other):
        return self.name < other.name


class ItemClass(SchemaElement):
    """An item class of a relational schema"""

    def __init__(self, name: str, attrs=()):
        """

        Parameters
        ----------
        name : str
            The name of item class
        attrs : AttributeClass or str or iterable of A_Class or str
            Attribute classes of the item class
        """
        if isinstance(attrs, AttributeClass):
            attrs = {attrs, }
        elif isinstance(attrs, str):
            attrs = {AttributeClass(attrs), }
        attrs = {AttributeClass(a) if isinstance(a, str) else a for a in attrs}
        assert all(isinstance(a, AttributeClass) for a in attrs)
        assert name not in _names(attrs)

        super().__init__(name)
        self.attrs = frozenset(attrs)
        self.attribute_classes = self.attrs

    @property
    def is_entity_class(self):
        raise AssertionError('abstract')

    @property
    def is_relationship_class(self):
        raise AssertionError('abstract')


class EntityClass(ItemClass):
    """An entity class of a relational schema"""

    def __init__(self, name, attrs=()):
        super().__init__(name, attrs)

    def __repr__(self):
        attr_part = ', '.join(str(a) for a in sorted(self.attrs)) if self.attrs else ''
        return self.name + "(" + attr_part + ")"

    def to_dict(self):
        return {'name': self.name,
                'attrs': [a.name for a in self.attrs]}

    @staticmethod
    def from_dict(dic):
        return EntityClass(dic['name'], dic['attrs'])

    @property
    def is_entity_class(self):
        return True

    @property
    def is_relationship_class(self):
        return False


class RelationshipClass(ItemClass):
    """A relationship class of a relational schema"""

    def __init__(self, name, attrs, cards: dict):
        super().__init__(name, attrs)
        self.__cards = cards.copy()
        self.entities = frozenset(self.__cards.keys())
        self.entity_classes = self.entities

    def __contains__(self, item):
        """Whether an entity class participates in this relationship class"""
        return item in self.__cards

    def __getitem__(self, entity):
        """Cardinality of a participating entity class"""
        return self.__cards[entity]

    def is_many(self, entity):
        """Whether the cardinality of a given participating entity class is many"""
        return self.__cards[entity] == Cardinality.many

    def __repr__(self):
        attr_part = ', '.join(str(a) for a in sorted(self.attrs)) if self.attrs else '()'
        card_part = '{' + (
            ', '.join(
                [str(e) + ': ' + ('many' if self.is_many(e) else 'one') for e in sorted(self.__cards.keys())])) + '}'
        return self.name + "(" + attr_part + ", " + card_part + ")"

    def to_dict(self):
        return {'name': self.name,
                'attrs': [a.name for a in self.attrs],
                'cards': {e.name: self.__cards[e] for e in self.__cards}}

    @staticmethod
    def from_dict(dic, es):
        es = {e.name: e for e in es}
        return RelationshipClass(dic['name'],
                                 dic['attrs'],
                                 {es[e]: Cardinality.from_int(dic['cards'][e]) for e in dic['cards']}
                                 )

    @property
    def is_entity_class(self):
        return False

    @property
    def is_relationship_class(self):
        return True


class AttributeClass(SchemaElement):
    """Attribute class"""

    def __init__(self, name):
        super().__init__(name)

    def __repr__(self):
        return 'A_Class(' + repr(self.name) + ')'

    def to_dict(self):
        return {'name': self.name}


class RelationalSchema:
    """Relational schema"""

    def __init__(self, entities, relationships):
        """Initialize a relational schema with given entity and relationship classes"""
        assert all(e.is_entity_class for e in entities)
        assert all(r.is_relationship_class for r in relationships)
        assert _is_unique((*_names(entities), *_names(relationships),
                           *[attr.name for item in (set(entities) | set(relationships)) for attr in item.attrs]))

        self.entities = frozenset(entities)
        self.relationships = frozenset(relationships)
        self.entity_classes = self.entities
        self.relationship_classes = self.relationships
        self.item_classes = self.entities | self.relationships
        self.attrs = frozenset(chain(*[i.attrs for i in chain(entities, relationships)]))

        __i2i = defaultdict(set)
        for r in relationships:
            __i2i[r] = r.entities
            for e in r.entities:
                __i2i[e].add(r)
        for e in self.entities - __i2i.keys():
            __i2i[e] = set()

        self.__i2i = {i: frozenset(__i2i[i]) for i in __i2i}
        self.attr2item_class = dict()
        for item_class in self.entities | self.relationships:
            for attr in item_class.attrs:
                self.attr2item_class[attr] = item_class

        self.elements = {e.name: e for e in self.entities | self.relationships | self.attrs}

    def __eq__(self, other):
        return isinstance(other, RelationalSchema) and \
               self.entities == other.entities and self.relationships == other.relationships

    def __getitem__(self, name) -> SchemaElement:
        """Returns a schema element given its name"""
        return self.elements[name]

    def item_class_of(self, attr) -> ItemClass:
        """an item class of the given attribute class"""
        return self.attr2item_class[attr]

    def __contains__(self, item):
        """Whether the given schema element is in this relational schema"""
        return item in self.entities or item in self.relationships or item in self.attrs

    def __str__(self):
        return "RSchema(" + ', '.join(e.name for e in sorted(self.entities | self.relationships)) + ")"

    def __repr__(self):
        return "RSchema(Entity classes: " + repr(sorted(self.entities)) + ", Relationship classes: " + repr(
            sorted(self.relationships)) + ")"

    def relateds(self, item_class: ItemClass) -> frozenset:
        """Neighboring item classes of the given item class"""
        return self.__i2i[item_class]

    def as_networkx_ug(self, with_attribute_classes=False) -> nx.Graph:
        """An undirected graph representation (networkx.Graph) of the relational skeleton."""
        g = nx.Graph()
        g.add_nodes_from(self.entities)
        g.add_nodes_from(self.relationships)
        for r in self.relationships:
            g.add_edges_from([(e, r) for e in r.entities])
        if with_attribute_classes:
            g.add_nodes_from(self.attrs)
            for attr in self.attrs:
                g.add_edge(self.item_class_of(attr), attr)
        return g

    def to_dict(self):
        return {'entities': [e.to_dict() for e in self.entities],
                'relationships': [r.to_dict() for r in self.relationships]}

    @staticmethod
    def from_dict(dic):
        es = [EntityClass.from_dict(e) for e in dic['entities']]
        return RelationalSchema(es,
                                [RelationshipClass.from_dict(r, es) for r in dic['relationships']])


@functools.total_ordering
class SkItem:
    """Item -- an instance (i.e., entity or relationship) in a relational skeleton"""

    def __init__(self, name, item_class: ItemClass, values: dict = None):
        self.name = name
        self.item_class = item_class
        self.__values = values.copy() if values is not None else dict()
        self.__h = hash(self.name)

    def __eq__(self, other):
        return isinstance(other, SkItem) and self.name == other.name

    def __hash__(self):
        return self.__h

    def __le__(self, other):
        return self.name <= other.name

    def __contains__(self, k):
        """Whether the value of given attribute is set"""
        return k in self.__values

    def __getitem__(self, item):
        """The value of the given attribute"""
        if item not in self.__values:
            return None
        return self.__values[item]

    def __setitem__(self, item, value):
        """Set a value for the given attribute"""
        self.__values[item] = value

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class RelationalSkeleton:
    """Relational skeleton"""

    def __init__(self, schema: RelationalSchema, strict=False):
        self.schema = schema
        self._G = nx.Graph()
        self._nodes_by_type = defaultdict(set)
        self.__strict = strict

    def __setitem__(self, key, value):
        """Set the value for the item's attribute (`key`)"""
        item, attr = key
        assert isinstance(item, SkItem)
        item[attr] = value

    def __getitem__(self, key):
        """The value for the item's attribute (`key`)"""
        item, attr = key
        if attr not in item:
            return None
        return item[attr]

    def add_entities(self, *args):
        for x in args:
            self.add_entity(x)

    def add_entity(self, item: SkItem):
        assert item.item_class.is_entity_class
        assert item not in self._G

        self._nodes_by_type[item.item_class].add(item)
        self._G.add_node(item, item_class=item.item_class.name)

    def add_relationship(self, rel: SkItem, entities):
        """Add a relationship item with specifying its participating entities.

        Notes
        -----
        Currently, entities must be added before the relationship is added.
        """
        assert rel.item_class.is_relationship_class
        assert all(e.item_class.is_entity_class for e in entities)
        assert rel not in self._G
        assert all(e in self._G for e in entities)

        for e in entities:
            if not rel.item_class.is_many(e.item_class):
                assert len(self.neighbors(e, rel.item_class)) == 0

        if self.__strict:
            assert set(rel.item_class.entities) == {e.item_class for e in entities}

        self._nodes_by_type[rel.item_class].add(rel)
        self._G.add_node(rel, item_class=rel.item_class.name)
        self._G.add_edges_from((rel, e) for e in entities)

    def items(self, filter_type: ItemClass = None):
        """Items of the given item class, if provided"""
        if filter_type is not None:
            return frozenset(self._nodes_by_type[filter_type])
        return frozenset(self._G)

    def neighbors(self, x, filter_type: ItemClass = None):
        """`x`'s neighboring items of the given item class, if provided"""
        if filter_type is None:
            return frozenset(self._G[x])
        else:
            return frozenset(filter(lambda y: y.item_class == filter_type, self._G[x]))

    def __str__(self):
        return str(self._G)

    def as_networkx_ug(self) -> nx.Graph:
        return self._G.copy()

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def remove(self, item):
        self._G.remove_node(item)
        if item in self._nodes_by_type[item.item_class]:
            self._nodes_by_type[item.item_class].remove(item)


def ImmutableRSkeleton(skeleton):
    return ClassImmutableRSkeleton(skeleton) if not isinstance(skeleton, ClassImmutableRSkeleton) else skeleton


class ClassImmutableRSkeleton(RelationalSkeleton):
    def __init__(self, skeleton: RelationalSkeleton):
        # TODO does not inherit correctly
        self.schema = skeleton.schema
        self._nodes = frozenset(skeleton._G.nodes_iter())
        self._nodes_of = defaultdict(frozenset)
        self._nodes_of.update({k: frozenset(vs) for k, vs in skeleton._nodes_by_type.items()})
        self._ne = defaultdict(frozenset)
        for v in self._nodes:
            neighbors = skeleton.neighbors(v)
            self._ne[(v, None)] = frozenset(neighbors)
            for k, g in itertools.groupby(sorted(neighbors, key=lambda x: x.item_class),
                                          key=lambda x: x.item_class):
                self._ne[(v, k)] = frozenset(g)
        self._G = nx.freeze(skeleton.as_networkx_ug())

    def __setitem__(self, key, value):
        raise AssertionError('not allowed to modify')

    def __getitem__(self, key):
        item, attr = key
        return item[attr]

    def add_entities(self, *args):
        raise AssertionError('not allowed to modify')

    def add_entity(self, item: SkItem):
        raise AssertionError('not allowed to modify')

    def add_relationship(self, rel: SkItem, entities):
        raise AssertionError('not allowed to modify')

    def items(self, filter_type: ItemClass = None):
        if filter_type is not None:
            return self._nodes_of[filter_type]
        return self._nodes

    def neighbors(self, x, filter_type: ItemClass = None):
        return self._ne[(x, filter_type)]

    def __str__(self):
        return str(self._G)

    def as_networkx_ug(self):
        return self._G.copy()

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


class cardinality_sampler:
    def __init__(self, p_many=0.5):
        assert 0.0 <= p_many <= 1.0
        self.p_many = p_many

    def sample(self):
        if random_sample() <= self.p_many:
            return Cardinality.many
        else:
            return Cardinality.one


def generate_schema(num_ent_classes_distr=between_sampler(2, 5),
                    num_rel_classes_distr=between_sampler(2, 5),
                    num_ent_classes_per_rel_class_distr=between_sampler(2, 3),
                    num_attr_classes_per_ent_class_distr=between_sampler(2, 3),
                    num_attr_classes_per_rel_class_distr=between_sampler(0, 0),
                    cardinality_distr=cardinality_sampler(0.75)  # Cardinality sampler
                    ) -> RelationalSchema:
    """A random relational schema.

    Notes
    -----
    Guarantees reproducibility
    """
    ent_classes = []
    rel_classes = []
    attr_count = itertools.count(1)

    num_ent_classes = num_ent_classes_distr.sample()
    if num_ent_classes < 2:
        num_rel_classes = 0
    else:
        num_rel_classes = num_rel_classes_distr.sample()

    for i in range(1, num_ent_classes + 1):
        n_attr = num_attr_classes_per_ent_class_distr.sample()
        attrs = (AttributeClass("A" + str(next(attr_count))) for _ in range(n_attr))
        ent_classes.append(EntityClass("E" + str(i), attrs))
    assert len(ent_classes) == num_ent_classes

    for i in range(1, num_rel_classes + 1):
        n_e_r = num_ent_classes_per_rel_class_distr.sample()
        n_e_r = max(min(n_e_r, num_ent_classes), 2)
        cards = {ent_classes[i]: cardinality_distr.sample() for i in choice(num_ent_classes, n_e_r, replace=False)}
        n_attr = num_attr_classes_per_rel_class_distr.sample()
        attrs = (AttributeClass("A" + str(next(attr_count))) for _ in range(n_attr))
        rel_classes.append(RelationshipClass("R" + str(i), attrs, cards))
    assert len(rel_classes) == num_rel_classes

    return RelationalSchema(ent_classes, rel_classes)


def generate_skeleton(schema: RelationalSchema, min_items_per_class=300, max_degree=3) -> RelationalSkeleton:
    """A random relational skeleton

    Parameters
    ----------
    schema : RelationalSchema
        a base relational schema.
    min_items_per_class : int
        a minimum number of items per item class of the resulting relational skeleton.
    max_degree : int
        a maximum number of relationships of the same type for an entity in the resulting skeleton.

    Notes
    -----
    Guarantees reproducibility
    """
    c = itertools.count()

    def entity_generator(E):
        while True:
            yield SkItem('e' + str(next(c)), E)

    def rel_generator(R):
        while True:
            yield SkItem('r' + str(next(c)), R)

    gens = {**{E: entity_generator(E) for E in schema.entities}, **{R: rel_generator(R) for R in schema.relationships}}
    n_nodes = {i: randint(min_items_per_class, round(1.2 * min_items_per_class))
               for i in sorted(schema.item_classes)}

    for R in sorted(schema.relationships):
        for E in sorted(R.entities):
            if not R.is_many(E) and n_nodes[E] < n_nodes[R]:
                n_nodes[R] = n_nodes[E]

    overage = min(n_nodes.values()) - min_items_per_class
    n_nodes = {k: v - overage for k, v in n_nodes.items()}

    nodes = defaultdict(list)
    g = nx.Graph()
    for I in sorted(schema.item_classes):
        for _ in range(n_nodes[I]):
            i = next(gens[I])
            nodes[I].append(i)
            g.add_node(i)

    for R in sorted(schema.relationships):
        for E in sorted(R.entities):
            g.add_edges_from(zip(nodes[R], choice(nodes[E], len(nodes[R]), replace=R.is_many(E))))

    for E in sorted(schema.entities):
        for e in nodes[E]:
            counted = Counter([ne.item_class for ne in g.neighbors(e)])
            for R, count in counted.most_common():
                if count > max_degree:
                    neighbors_of_R = list(filter(lambda ne: ne.item_class == R, sorted(g.neighbors(e))))
                    to_remove = choice(neighbors_of_R, count - max_degree, replace=False)
                    g.remove_nodes_from(to_remove)
                    for r in to_remove:
                        nodes[R].remove(r)
                else:
                    break
    skeleton = RelationalSkeleton(schema, strict=True)
    for E in sorted(schema.entities):
        for e in nodes[E]:
            skeleton.add_entity(e)
    for R in sorted(schema.relationships):
        for r in nodes[R]:
            skeleton.add_relationship(r, g.neighbors(r))
    return skeleton


def remove_lone_entities(skeleton: RelationalSkeleton):
    """Removes every entitiy which is not connected to other entities."""
    to_remove = list()
    for e in skeleton.schema.entities:
        for e_item in skeleton.items(e):
            if not skeleton.neighbors(e_item):
                to_remove.append(e_item)

    for item in to_remove:
        skeleton.remove(item)


def repeat_skeleton(skeleton: RelationalSkeleton, times) -> RelationalSkeleton:
    """A relational skeleton repeating the given relational skeleton given times."""
    new_skeleton = RelationalSkeleton(skeleton.schema, True)
    for i in range(times):
        @functools.lru_cache(maxsize=None)
        def ify(item0: SkItem):
            return SkItem(str(i) + '_' + item0.name, item0.item_class)

        for item in sorted(skeleton.items()):
            if item.item_class.is_entity_class:
                new_skeleton.add_entity(ify(item))
        for item in sorted(skeleton.items()):
            if item.item_class.is_relationship_class:
                new_skeleton.add_relationship(ify(item), [ify(ne) for ne in skeleton.neighbors(item)])

    return new_skeleton


def skeleton_to_entities_only_nx_graph(skeleton: RelationalSkeleton) -> nx.Graph:
    """Returns a skeleton to an entity-only undirected graph"""
    entities_ug = nx.Graph()
    for e_cl in skeleton.schema.entities:
        for ent in skeleton.items(e_cl):
            entities_ug.add_node(ent, item_class=ent.item_class.name)  # attribute

    for r_cl in skeleton.schema.relationships:
        for rel in skeleton.items(r_cl):
            two_ents = skeleton.neighbors(rel)
            assert len(two_ents) == 2
            u, v = two_ents
            entities_ug.add_edge(u, v)
    return entities_ug
