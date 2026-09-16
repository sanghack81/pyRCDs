"""Mock-based tests for visualize.py that run without pygraphviz."""

import os
import tempfile
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

from pyrcds.domain import generate_skeleton, remove_lone_entities
from pyrcds.model import GroundGraph
from pyrcds.tests.testing_utils import company_schema, company_rcm
from pyrcds.visualize import visualize_schema, visualize_ground_graph, visualize_skeleton


def _fake_layout(G, **kwargs):
    """Return a deterministic layout: each node gets a unique (x, y) position."""
    return {node: (float(i), float(i)) for i, node in enumerate(G.nodes())}


@patch("pyrcds.visualize.nx.nx_agraph.graphviz_layout", side_effect=_fake_layout)
def test_visualize_schema_returns_expected_keys(mock_layout):
    schema = company_schema()
    with tempfile.TemporaryDirectory() as tmpdir:
        outfile = os.path.join(tmpdir, "schema.pdf")
        result = visualize_schema(schema, outfile, title="test schema")

    assert isinstance(result, dict)
    for key in ("pos", "pal", "factor", "xmin", "xmax", "ymin", "ymax"):
        assert key in result, f"Missing key: {key}"
    assert isinstance(result["pos"], dict)
    assert result["factor"] > 0
    mock_layout.assert_called_once()


@patch("pyrcds.visualize.nx.nx_agraph.graphviz_layout", side_effect=_fake_layout)
def test_visualize_skeleton_returns_expected_keys(mock_layout):
    schema = company_schema()
    skeleton = generate_skeleton(schema, 20)
    remove_lone_entities(skeleton)
    with tempfile.TemporaryDirectory() as tmpdir:
        outfile = os.path.join(tmpdir, "skeleton.pdf")
        result = visualize_skeleton(skeleton, outfile, title="test skeleton")

    assert isinstance(result, dict)
    for key in ("pos", "pal", "factor", "xmin", "xmax", "ymin", "ymax"):
        assert key in result, f"Missing key: {key}"
    assert isinstance(result["pos"], dict)
    mock_layout.assert_called_once()


@patch("pyrcds.visualize.nx.nx_agraph.graphviz_layout", side_effect=_fake_layout)
def test_visualize_skeleton_reuses_options(mock_layout):
    schema = company_schema()
    skeleton = generate_skeleton(schema, 20)
    remove_lone_entities(skeleton)
    with tempfile.TemporaryDirectory() as tmpdir:
        outfile1 = os.path.join(tmpdir, "skeleton1.pdf")
        options = visualize_skeleton(skeleton, outfile1)
        outfile2 = os.path.join(tmpdir, "skeleton2.pdf")
        result2 = visualize_skeleton(skeleton, outfile2, **options)

    # When options are passed in, the returned values should be consistent
    assert result2["factor"] == options["factor"]


@patch("pyrcds.visualize.nx.nx_agraph.graphviz_layout", side_effect=_fake_layout)
def test_visualize_ground_graph_returns_expected_keys(mock_layout):
    schema = company_schema()
    skeleton = generate_skeleton(schema, 20)
    remove_lone_entities(skeleton)
    rcm = company_rcm()
    gg = GroundGraph(rcm, skeleton)
    with tempfile.TemporaryDirectory() as tmpdir:
        outfile = os.path.join(tmpdir, "gg.pdf")
        result = visualize_ground_graph(gg, outfile, title="test gg")

    assert isinstance(result, dict)
    for key in ("pos", "pal", "factor", "xmin", "xmax", "ymin", "ymax"):
        assert key in result, f"Missing key: {key}"
    mock_layout.assert_called_once()


@patch("pyrcds.visualize.nx.nx_agraph.graphviz_layout", side_effect=_fake_layout)
def test_visualize_ground_graph_with_edgelist(mock_layout):
    schema = company_schema()
    skeleton = generate_skeleton(schema, 20)
    remove_lone_entities(skeleton)
    rcm = company_rcm()
    gg = GroundGraph(rcm, skeleton)
    dag = gg.as_networkx_dag()
    edgelist = list(dag.edges())[:3] if dag.edges() else []
    with tempfile.TemporaryDirectory() as tmpdir:
        outfile = os.path.join(tmpdir, "gg_edges.pdf")
        result = visualize_ground_graph(gg, outfile, edgelist=edgelist)

    assert isinstance(result, dict)


@patch("pyrcds.visualize.nx.nx_agraph.graphviz_layout", side_effect=_fake_layout)
def test_visualize_schema_creates_file(mock_layout):
    schema = company_schema()
    with tempfile.TemporaryDirectory() as tmpdir:
        outfile = os.path.join(tmpdir, "schema_file.pdf")
        visualize_schema(schema, outfile)
        assert os.path.exists(outfile), "Output file should be created"
        assert os.path.getsize(outfile) > 0, "Output file should not be empty"
