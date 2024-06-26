import networkx as nx
import numpy as np
import os
import pyvista

import torch
import torch_geometric
from matplotlib import pyplot as plt


def process_edge_and_rad(edges, edge_attr):
    """
    Converts the edge and edge indices into dense adjacency matrix.
    Next, picks the upper traingular matrix and returns the directed edges using torch.nonzero()
    :param edges: torch.tensor(2, E)
    :param edge_attr: torch.tensor(E, )
    :return: torch.tensor(2, E), torch.tensor(E, )
    """
    # print(f"Old edges = {edges}")
    dense_adj = torch_geometric.utils.to_dense_adj(edge_index=edges, edge_attr=edge_attr)
    # An extra batch dimension added by pytorch geometric. Squeezing it else leads to invalid results
    dense_adj = dense_adj.squeeze(0)
    directed_adj = torch.triu(dense_adj)
    edge_indices = torch.nonzero(directed_adj, as_tuple=False).t()
    edge_attr = directed_adj[edge_indices[0], edge_indices[1]]
    # print(f"New edges = {edge_indices}")
    return edge_indices, edge_attr


def plot_networkx_graphs(idx, base_dir, orig_nodes, orig_edges, gen_nodes=None, gen_edges=None):
    save_dir = os.path.join(base_dir, "nx_samples")
    os.makedirs(save_dir, exist_ok=True)
    # Converting numpy tensors into simple python list
    orig_edges = orig_edges.T.cpu().numpy().tolist()

    # The `*_nodes` contain position coordinates. We do not need it.
    # We just need to have a list of nodes
    orig_nodes = list(range(orig_nodes.size(0)))

    # Create NetworkX graphs from the edges
    graph1 = nx.Graph()

    # Add the nodes. This is to make sure that the graphs have the same number of nodes.
    # The number of nodes can be different if we just use the `add_edges_from()` method
    graph1.add_nodes_from(orig_nodes)

    # Now add the edges

    graph1.add_edges_from(orig_edges)
    # Create a Matplotlib figure with two subplots
    num_subplots = 2 if gen_edges is not None else 1
    fig, axes = plt.subplots(1, num_subplots, figsize=(10, 4))
    axis_1 = axes[0] if num_subplots == 2 else axes
    # Plot the first graph on the first subplot
    pos1 = nx.spring_layout(graph1)
    nx.draw(graph1, pos=pos1, with_labels=True, labels={node: node for node in orig_nodes}, ax=axis_1)
    axis_1.set_title('Graph 1')

    if gen_nodes is not None and gen_edges is not None:
        plt.subplot(1, 2, 2)
        gen_edges = gen_edges.T.cpu().numpy().tolist()
        gen_nodes = list(range(gen_nodes.size(0)))
        graph2 = nx.Graph()
        graph2.add_nodes_from(gen_nodes)
        graph2.add_edges_from(gen_edges)

        # Plot the second graph on the second subplot
        pos2 = nx.spring_layout(graph2)
        nx.draw(graph2, pos=pos2, with_labels=True, labels={node: node for node in gen_nodes}, ax=axes[1])
        axes[1].set_title('Graph 2')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{idx}.png"))
    plt.show()


def save_as_vtp(base_dir, nodes, edge_index, edge_attr, idx):
    mesh_edge = np.concatenate((np.int32(2 * np.ones((1, edge_index.shape[1]))), edge_index), 0)
    mesh = pyvista.UnstructuredGrid(mesh_edge.T, np.array([4] * len(edge_attr)), nodes.numpy())
    mesh.cell_data['radius'] = edge_attr.numpy()
    mesh_structured = mesh.extract_surface()
    # mesh.save(save_path + 'vtp/sample_' + str(idx).zfill(6) + '_graph.vtp')
    recons_path = os.path.join(base_dir,  "synthetic_data", "vtp")
    os.makedirs(recons_path, exist_ok=True)
    # sample_043309_graph.vtp
    mesh_structured.save(recons_path + f'/sample_graph_{idx}.vtp')