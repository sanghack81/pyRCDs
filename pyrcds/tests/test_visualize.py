from pyrcds.domain import generate_skeleton, remove_lone_entities
from pyrcds.model import GroundGraph
from pyrcds.tests.testing_utils import company_schema, company_rcm
from pyrcds.visualize import visualize_schema, visualize_ground_graph, visualize_skeleton


def test_all_three():
    schema = company_schema()
    visualize_schema(company_schema(), 'company_schema.pdf')

    sk = generate_skeleton(schema, 20)
    options = visualize_skeleton(sk, 'company_skeleton_before.pdf')
    remove_lone_entities(sk)
    visualize_skeleton(sk, 'company_skeleton_after.pdf', **options)

    rcm = company_rcm()
    for prog in ['neato', 'dot', 'twopi', 'circo']:  # , 'fdp', , 'nop'
        visualize_ground_graph(GroundGraph(rcm, sk), f'company_gg_{prog}.pdf', prog=prog)
