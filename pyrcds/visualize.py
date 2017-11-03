import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from pyrcds.domain import RelationalSkeleton, EntityClass, RelationalSchema, RelationshipClass
from pyrcds.model import GroundGraph
from pyrcds.utils import group_by


def visualize_schema(schema: RelationalSchema, filename, title='untitled relational schema', **options):
    sns.set()
    sns.set_style(style='white')

    fig, ax = plt.subplots(figsize=(options.get('figure_width', 6), options.get('figure_height', 6)))
    G = schema.as_networkx_ug(with_attribute_classes=True)
    pos = options.get('pos', nx.nx_agraph.graphviz_layout(G))
    pal = options.get('pal', sns.color_palette(options.get('palette', "Paired"), 3))

    num_to_display = len(schema.item_classes) + len(schema.attrs)
    factor = options.get('factor', max(1, np.math.log((num_to_display) / np.math.log(8))) / 2)

    def marker_of(node):
        if isinstance(node, EntityClass):
            return 's'
        elif isinstance(node, RelationshipClass):
            return 'D'
        else:
            return 'o'

    for i, nodes in enumerate([schema.entity_classes, schema.relationship_classes, schema.attrs]):
        for v in nodes:
            ax.scatter([pos[v][0]], [pos[v][1]],
                       s=300 / factor,
                       c=pal[i],
                       marker=marker_of(v),
                       alpha=1,
                       linewidths=0
                       ).set_zorder(2)

    nx.draw_networkx_edges(G, pos, arrows=False, ax=ax, width=0.5)
    nx.draw_networkx_labels(G, pos,
                            labels=dict((schema_element, schema_element.name) for schema_element in G.nodes),
                            font_size=9 - factor, ax=ax, alpha=1)

    plt.title(title)
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        left='off',
        labelleft='off',
        labelbottom='off')  # labels along the bottom edge are off
    fig.savefig(filename, transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    return pos, pal, factor


def visualize_ground_graph(gg: GroundGraph, filename, title='undirected Ground Graph', **options):
    sns.set()
    sns.set_style(style='white')
    schema = gg.schema

    if options.get('prog', 'dot') == 'dot':
        fig, ax = plt.subplots(figsize=(options.get('figure_width', 2 * 6), options.get('figure_height', 6 // 2)))
    else:
        fig, ax = plt.subplots(figsize=(options.get('figure_width', 6), options.get('figure_height', 6)))
    G = gg.as_networkx_dag()
    pos = options.get('pos', nx.nx_agraph.graphviz_layout(G, prog=options.get('prog', 'neato')))
    pal = options.get('pal', sns.color_palette(options.get('palette', "Paired"), len(schema.attrs)))

    all_item_attributes = G.nodes()
    factor = options.get('factor', 2 * max(1, np.math.log(len(all_item_attributes) / np.math.log(8))))

    for i, (attr, item_attributes) in enumerate(group_by(all_item_attributes, lambda ia: ia[1])):
        xy = np.asarray([pos[v] for v in item_attributes])
        node_collection = ax.scatter(xy[:, 0], xy[:, 1],
                                     s=300 / factor,
                                     c=pal[i],
                                     alpha=1,
                                     linewidths=0,
                                     label=attr)
        node_collection.set_zorder(2)

    nx.draw_networkx_edges(G, pos, arrows=True, ax=ax, width=0.5)
    plt.title(title)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        left='off',
        labelleft='off',
        labelbottom='off')  # labels along the bottom edge are off
    fig.savefig(filename, transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    return pos, pal, factor


def visualize_skeleton(skeleton: RelationalSkeleton, filename, title='relational skeleton', **options):
    sns.set()
    sns.set_style(style='white')

    fig, ax = plt.subplots(figsize=(options.get('figure_width', 6), options.get('figure_height', 6)))
    G = skeleton.as_networkx_ug()
    pos = options.get('pos', nx.nx_agraph.graphviz_layout(G, prog=options.get('prog', 'neato')))
    pal = options.get('pal', sns.color_palette(options.get('palette', "Paired"), len(skeleton.schema.item_classes)))

    nodelist = list(G.nodes)
    factor = options.get('factor', max(1, np.math.log(len(nodelist) / np.math.log(8))))

    for i, (item_class, nodes) in enumerate(group_by(nodelist, lambda node: node.item_class)):
        xy = np.asarray([pos[v] for v in nodes])
        node_collection = ax.scatter(xy[:, 0], xy[:, 1],
                                     s=(300 if isinstance(item_class, EntityClass) else 100) / factor,
                                     c=pal[i],
                                     marker='s' if isinstance(item_class, EntityClass) else 'D',
                                     alpha=1,
                                     linewidths=0,
                                     label=item_class)
        node_collection.set_zorder(2)

    nx.draw_networkx_edges(G, pos, arrows=False, ax=ax, width=0.5)
    plt.title(title)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        left='off',
        labelleft='off',
        labelbottom='off')  # labels along the bottom edge are off
    fig.savefig(filename, transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    return pos, pal, factor
