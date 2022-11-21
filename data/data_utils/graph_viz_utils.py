import numpy as np
import os
import sys
import pickle
import networkx as nx
import collections
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import rankdata

def get_spectral_graph_positions(num_nodes, layout="spring"):
    """
    Get positions of EEG electrodes for visualizations
    """

    graph = nx.Graph()
    node_id_label = collections.defaultdict()

    for i in range(num_nodes):
        graph.add_node(i)

    # Add pseudo edges
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:   # do no include self-edge in visualization
                graph.add_edge(i, j)

    if layout == "spring":
        pos = nx.spring_layout(graph)
    elif layout == "spectral":
        pos = nx.spectral_layout(graph)
    else:
        raise NotImplementedError

    # keep the nice shape of the electronodes on the scalp
    pos_spec = {node: (y, -x) for (node, (x, y)) in pos.items()}

    return pos_spec

def get_ecg_graph_positions(lead_vector, node_id2ind):
    """
    Get positions of EEG electrodes for visualizations
    """

    graph = nx.Graph()
    num_nodes = len(lead_vector)

    for i in range(num_nodes):
        graph.add_node(i)
    
    # Add pseudo edges
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:   # do no include self-edge in visualization
                graph.add_edge(i, j)
        
    # keep the nice shape of the electronodes on the scalp
    pos_spec = {node_id2ind[node]: (x, y, z) for (node, (x, y, z)) in lead_vector.items()}

    return pos_spec

def draw_graph_weighted_edge(
        adj_mx,
        channel_names,
        pos_spec,
        is_directed,
        title='',
        save_dir=None,
        fig_size=(
            12,
            8),
        node_color='Red',
        font_size=20,
        node_size=250,
        plot_colorbar=False):
    """
    Draw a graph with weighted edges
    Args:
        adj_mx: Adjacency matrix for the graph, shape (num_nodes, num_nodes
        node_id_dict: dict, key is node name, value is node index
        pos_spec: Graph node position specs from function get_spectral_graph_positions
        is_directed: If True, draw directed graphs
        title: str, title of the figure
        save_dir: Dir to save the plot
        fig_size: figure size
    """
    graph = nx.DiGraph() if is_directed else nx.Graph()
    node_id_label = collections.defaultdict()

    for i in range(adj_mx.shape[0]):
        graph.add_node(i)

    for i, lab in enumerate(channel_names):
        node_id_label[i] = lab.split('BSK ')[-1]

    # Add edges
    for i in range(adj_mx.shape[0]):
        for j in range(adj_mx.shape[1]):  # since it's now directed
            if i != j and adj_mx[i, j] > 0:
                graph.add_edge(i, j, weight=adj_mx[i, j])

    edges, weights = zip(*nx.get_edge_attributes(graph, 'weight').items())

    # Change the color scales below
    k = 3
    cmap = plt.cm.Greys(np.linspace(0, 1, (k + 1) * len(weights)))
    cmap = matplotlib.colors.ListedColormap(cmap[len(weights):-1:(k - 1)])

    plt.figure(figsize=fig_size)
    nx.draw_networkx(graph, pos_spec, labels=node_id_label, with_labels=True,
                     edgelist=edges, edge_color=rankdata(weights),
                     width=fig_size[1] / 2, edge_cmap=cmap, font_weight='bold',
                     node_color=node_color,
                     node_size=node_size,
                     font_color='white',
                     font_size=font_size)
    plt.title(title, fontsize=font_size)
    plt.axis('off')
    if plot_colorbar:
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(
                vmin=0, vmax=1))
        sm.set_array([])
        plt.colorbar(sm)
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir, dpi=300)

    plt.show()
