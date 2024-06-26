import glob
import io
import os
import shutil

import numpy as np
import pyvista
import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops
from tqdm import tqdm

from bootstrap.raw_data_to_midi_utils.convert_files import remove_self_loops_and_save_as_pt
from environment_setup import PROJECT_ROOT_DIR, time_logging


def load_vtp_graph(filename):
    vtk_data = pyvista.read(filename)
    nodes = torch.tensor(np.float32(np.asarray(vtk_data.points)), dtype=torch.float)
    edges = torch.tensor(np.asarray(vtk_data.lines.reshape(-1, 3)), dtype=torch.int64)[:, 1:]
    radius = torch.tensor(np.asarray(vtk_data.cell_data['radius']), dtype=torch.float)
    edges = edges.T
    edges, radius = remove_self_loops(edge_index=edges, edge_attr=radius)
    degree = torch.bincount(edges[0, :], minlength=len(nodes)) + \
             torch.bincount(edges[1, :], minlength=len(nodes))
    graph_data = Data(x=nodes, edge_index=edges, edge_attr=radius)
    return graph_data, degree


def save_updated_vtp(pyG_data, filename, split_save_path):
    nodes, edge_index, edge_attr = pyG_data.x, pyG_data.edge_index, pyG_data.edge_attr
    mesh_edge = np.concatenate((np.int32(2 * np.ones((1, edge_index.shape[1]))), edge_index), 0)
    mesh = pyvista.UnstructuredGrid(mesh_edge.T, np.array([4] * len(edge_attr)), nodes.numpy())
    mesh.cell_data['radius'] = edge_attr.numpy()
    mesh_structured = mesh.extract_surface()
    base_filename = os.path.basename(filename)
    base_filename = base_filename.replace("sample", "processed")
    mesh_structured.save(os.path.join(split_save_path, base_filename))


def _remove_neg_radius_edges(pyGdata, filename, string_buffer):
    mask = pyGdata.edge_attr > 0
    edge_index_pruned = pyGdata.edge_index.T[mask]
    pyGdata.edge_index = edge_index_pruned.T
    pyGdata.edge_attr = pyGdata.edge_attr[mask]
    if mask.sum() != mask.shape[0]:
        # At least one edge has a negative radius
        string_buffer.write(f"{filename=} has negative edge. Removing these edges\n")


def remove_neg_radius_edges_for_one_graph(filename, string_buffer, split_save_path):
    pyGdata, degree = load_vtp_graph(filename)
    # Remove the negative edges
    _remove_neg_radius_edges(pyGdata=pyGdata, filename=filename, string_buffer=string_buffer)
    save_updated_vtp(pyG_data=pyGdata, filename=filename, split_save_path=split_save_path)


def remove_neg_rad_edges_for_all_graphs(split_path, split_save_path, non_neg_rad_graph_loc, split):
    # Initialize an in-memory buffer
    buffer = io.StringIO()
    for filename in tqdm(glob.glob(f"{split_path}/*.vtp")):
        # We only remove the negative edges. Everything else remains.
        remove_neg_radius_edges_for_one_graph(filename, string_buffer=buffer, split_save_path=split_save_path)
    # Now let us save the information
    final_content = buffer.getvalue()
    # Close the buffer (not necessary for io.StringIO)
    buffer.close()
    # Write the final content to a text file
    with open(os.path.join(non_neg_rad_graph_loc, f"{split}_changelog.txt"), "w") as file:
        file.write(final_content)


@time_logging
def preprocess_graph_to_remove_neg_edges(unprocessed_vessap_path, non_neg_rad_graph_loc):
    if os.path.exists(non_neg_rad_graph_loc):
        print("non-neg rad folder exists. Cleaning it and re-generating")
        shutil.rmtree(non_neg_rad_graph_loc)
        os.makedirs(non_neg_rad_graph_loc)
    for split in ['train_data', 'test_data']:
        # Make all the directories before processing
        os.makedirs(os.path.join(non_neg_rad_graph_loc, split), exist_ok=True)
    for split in ['train_data', 'test_data']:
        print(f"Processing {split=}")
        split_path = os.path.join(unprocessed_vessap_path, split, 'vtp')
        split_save_path = os.path.join(non_neg_rad_graph_loc, split)
        remove_neg_rad_edges_for_all_graphs(split_path=split_path, split_save_path=split_save_path,
                                            non_neg_rad_graph_loc=non_neg_rad_graph_loc, split=split)


@time_logging
def add_psuedo_conn_and_create_dataset(non_neg_rad_graph_loc, make_undirected,
                                       add_node_diameter, include_cont_radii):
    append_str = "" if make_undirected else 'directed_'
    midi_graph_path = os.path.join(PROJECT_ROOT_DIR, 'data', 'vessap', f'converted_{append_str}pt_files')
    if os.path.exists(midi_graph_path):
        print("Existing data folder found. Cleaning it now.")
        shutil.rmtree(midi_graph_path)
    os.makedirs(midi_graph_path)
    # Create the string buffer object
    buffer = io.StringIO()
    idx = 0

    print("Generating undirected graphs. This is done by using pyG utility function")
    print("--------------")
    add_node_coords = True
    print(f"{make_undirected=}, {add_node_coords=}, {add_node_diameter=}")
    print("--------------")
    try:
        # We would be processing only the training split. The test split is untouched.
        for filename in tqdm(glob.glob(f"{non_neg_rad_graph_loc}/train_data/*.vtp")):
            remove_self_loops_and_save_as_pt(filename=filename,
                                             idx=idx,
                                             buffer=buffer, save_path=midi_graph_path,
                                             make_undirected=make_undirected,
                                             add_node_diameter=add_node_diameter,
                                             include_cont_radii=include_cont_radii
                                             )
            idx += 1
    finally:
        final_content = buffer.getvalue()
        buffer.close()
        # Write the final content to a text file
        with open(os.path.join(non_neg_rad_graph_loc, f"{append_str}updated_graph_changelog.txt"), "w") as file:
            file.write(final_content)


@time_logging
def generate_data(unprocessed_vessap_path, non_neg_rad_graph_loc, make_undirected,
                  add_node_diameter, include_cont_radii):
    preprocess_graph_to_remove_neg_edges(unprocessed_vessap_path=unprocessed_vessap_path,
                                         non_neg_rad_graph_loc=non_neg_rad_graph_loc)
    add_psuedo_conn_and_create_dataset(non_neg_rad_graph_loc=non_neg_rad_graph_loc,
                                       make_undirected=make_undirected,
                                       add_node_diameter=add_node_diameter, include_cont_radii=include_cont_radii)


if __name__ == '__main__':
    unprocessed_vessap_graph_path = '/mnt/elephant/chinmay/vessap_new/'
    non_neg_rad_graph_loc = '/mnt/elephant/chinmay/midi_vessap/non_neg_rad_graph'
    if os.path.exists(non_neg_rad_graph_loc):
        print("Cleaning existing destination path")
        folder_contents = os.listdir(non_neg_rad_graph_loc)
        # Retaining the changelog files since they might come in handy
        candidates = [x for x in folder_contents if 'changelog' not in x]
        for x in candidates:
            shutil.rmtree(os.path.join(non_neg_rad_graph_loc, x))
    generate_data(unprocessed_vessap_path=unprocessed_vessap_graph_path,
                  non_neg_rad_graph_loc=non_neg_rad_graph_loc,
                  make_undirected=True,
                  add_node_diameter=False, include_cont_radii=True)
