import os
import torch
from torch_geometric.utils import remove_self_loops

from midi.datasets.dataset_utils import load_pickle
from post_process.post_process_utils import process_edge_and_rad, plot_networkx_graphs, save_as_vtp


def load_graph(data_dir):
    pyG_list_original = load_pickle(path=os.path.join(data_dir, 'original_test_samples.pkl'))
    pyG_list_generated = load_pickle(path=os.path.join(data_dir, 'interpolated_test_samples.pkl'))
    for idx, (pyG_orig_dataset, pyG_generated_dataset) in enumerate(zip(pyG_list_original, pyG_list_generated)):
        orig_nodes, orig_edges, orig_attr = pyG_orig_dataset.x, pyG_orig_dataset.edge_index, pyG_orig_dataset.edge_attr
        gen_nodes, gen_edges, gen_attr = pyG_generated_dataset.x, pyG_generated_dataset.edge_index, pyG_generated_dataset.edge_attr
        # Remove the self loops if any
        orig_edges, orig_attr = remove_self_loops(edge_index=orig_edges, edge_attr=orig_attr)
        gen_edges, gen_attr = remove_self_loops(edge_index=gen_edges, edge_attr=gen_attr)
        orig_edge_rad_range = [(max(x - 1, 0), x) for x in orig_attr]
        gen_edge_rad_range = [(max(x - 1, 0), x) for x in gen_attr]
        orig_random_edge_rad = torch.cat([torch.FloatTensor(1).uniform_(a, b) for (a, b) in orig_edge_rad_range], dim=0)
        gen_random_edge_rad = torch.cat([torch.FloatTensor(1).uniform_(a, b) for (a, b) in gen_edge_rad_range], dim=0)
        # We need to ensure symmetry.
        # Of the two edges that are generated (since bidirectional), we would just drop the smaller radius one
        orig_selected_edge_attr, orig_selected_edges = process_edge_and_rad(edges=orig_edges,
                                                                            random_edge_rad=orig_random_edge_rad)
        gen_selected_edge_attr, gen_selected_edges = process_edge_and_rad(edges=gen_edges,
                                                                          random_edge_rad=gen_random_edge_rad)

        save_as_vtp(base_dir=data_dir, nodes=orig_nodes, edge_index=orig_selected_edges,
                    edge_attr=orig_selected_edge_attr, idx=f"{idx}_orig")
        # Now saving the generated file as well for comparison
        save_as_vtp(base_dir=data_dir, nodes=gen_nodes, edge_index=gen_selected_edges,
                    edge_attr=gen_selected_edge_attr, idx=f"{idx}_gen")
        if idx <= 5:
            # Also plot a few images for visualization
            plot_networkx_graphs(idx=idx, base_dir=data_dir, orig_nodes=orig_nodes, orig_edges=orig_selected_edges,
                                 gen_nodes=gen_nodes, gen_edges=gen_selected_edges)


if __name__ == '__main__':
    data_dir = '/home/chinmayp/workspace/MiDi/outputs/2023-09-20/18-13-46-graph-vessel-model'
    load_graph(data_dir=data_dir)
