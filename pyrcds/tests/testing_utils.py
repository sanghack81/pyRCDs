import functools
import multiprocessing
from os import makedirs
from os.path import expanduser

from pyrcds.domain import RelationshipClass, EntityClass, AttributeClass, Cardinality, RelationalSchema, RelationalSkeleton, SkItem, generate_skeleton, \
    ImmutableRSkeleton, remove_lone_entities
from pyrcds.model import RelationalDependency, RelationalPath, ParamRCM, generate_values_for_skeleton, normalize_skeleton, RCM
from pyrcds.model import RelationalVariable
from pyrcds.utils import linear_gaussian, average_agg, normal_sampler, xors


@functools.lru_cache(1)
def EPBDF():
    E = EntityClass('Employee', ('Salary', 'Competence'))
    P = EntityClass('Product', (AttributeClass('Success'),))
    B = EntityClass('BizUnit', (AttributeClass('Revenue'), AttributeClass('Budget')))
    D = RelationshipClass('Develops', tuple(), {E: Cardinality.many, P: Cardinality.many})
    F = RelationshipClass('Funds', tuple(), {P: Cardinality.one, B: Cardinality.many})

    return E, P, B, D, F


@functools.lru_cache(1)
def company_deps():
    E, P, B, D, F = EPBDF()
    deps = (RelationalDependency(RelationalVariable(E, 'Competence'), RelationalVariable(E, 'Salary')),
            RelationalDependency(RelationalVariable([E, D, P, F, B], 'Budget'), RelationalVariable(E, 'Salary')),
            RelationalDependency(RelationalVariable([P, D, E], 'Competence'), RelationalVariable(P, 'Success')),
            RelationalDependency(RelationalVariable([B, F, P], 'Success'), RelationalVariable(B, 'Revenue')),
            RelationalDependency(RelationalVariable(B, 'Revenue'), RelationalVariable(B, 'Budget')))
    return deps


@functools.lru_cache(1)
def company_schema() -> RelationalSchema:
    """A sample relational schema. See [1]

    References
    ----------
    [1] Marc Maier (2014),
        Causal Discovery for Relational Domains:
        Representation, Reasoning, and Learning,
        Ph.D. Dissertation, University of Massachusetts, Amherst

    """

    E, P, B, D, F = EPBDF()
    return RelationalSchema({E, P, B}, {D, F})


@functools.lru_cache(1)
def company_skeleton():
    """A sample relational skeleton. See [1]

    References
    ----------
    [1] Marc Maier (2014),
        Causal Discovery for Relational Domains:
        Representation, Reasoning, and Learning,
        Ph.D. Dissertation, University of Massachusetts, Amherst

    """

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

    return skeleton


@functools.lru_cache(1)
def company_rcm() -> RCM:
    """A sample relational causal model. See [1]

    References
    ----------
    [1] Marc Maier (2014),
        Causal Discovery for Relational Domains:
        Representation, Reasoning, and Learning,
        Ph.D. Dissertation, University of Massachusetts, Amherst

    """
    return RCM(company_schema(), company_deps())


def gen_company_data(n, max_degree=2, stdev=0.3, xor=False, flip=0.15, remove_lone=False, schema=company_schema(),
                     rcm=company_rcm()):
    functions = dict()
    effects = {RelationalVariable(RelationalPath(rcm.schema.item_class_of(attr)), attr) for attr in rcm.schema.attrs}
    skeleton = generate_skeleton(schema, n, max_degree=max_degree)
    if remove_lone:
        remove_lone_entities(skeleton)

    for e in effects:
        parameters = {cause: 1.0 for cause in rcm.pa(e)}
        if xor:
            # stdev will be used as xor flip probability
            functions[e] = xors(parameters, flip)
        else:
            functions[e] = linear_gaussian(parameters, average_agg(), normal_sampler(0, stdev))

    rcm = ParamRCM(rcm.schema, rcm.directed_dependencies, functions)
    generate_values_for_skeleton(rcm, skeleton)
    normalize_skeleton(skeleton)

    skeleton = ImmutableRSkeleton(skeleton)
    return rcm, skeleton


def print_both(msg, file=None, **options):
    print(msg)
    if file is not None:
        print(msg, file=file, **options)


def default_kcipt_options(**override):
    vals = {'n_jobs': multiprocessing.cpu_count(),
            'alpha': 0.05,
            'B': 128,
            'b': 128,
            'to_record': {'inner_null', 'mmds'},
            'p_value_method': 'normal'
            }
    for k, v in override.items():
        vals[k] = v
    return vals


def project_dir():
    home = expanduser("~")
    return home + '/Dropbox/research/2014 rcm/workspace/python/pyRCDs/'


def project_temp_dir(name=None):
    if name is None:
        return project_dir() + 'temporary/'
    else:
        d = project_dir() + 'temporary/' + str(name) + '/'
        makedirs(d, exist_ok=True)
        return d
