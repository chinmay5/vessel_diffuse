import glob
import math
import os
from collections import defaultdict

import numpy as np
import pyvista
import torch
import torch_geometric.utils
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm

from environment_setup import time_logging

REAL_VESSAP = '/mnt/elephant/chinmay/midi_vessap/non_neg_rad_graph/train_data'
REAL_CoW = '/mnt/elephant/chinmay/COWN_plus_top_CoW/all_vtps'


def _read_vtp_file(filename):
    vtk_data = pyvista.read(filename)
    nodes = torch.tensor(np.float32(np.asarray(vtk_data.points)), dtype=torch.float)
    edges = torch.tensor(np.asarray(vtk_data.lines.reshape(-1, 3)), dtype=torch.int64)[:, 1:]
    edges = edges.T
    if 'radius' in vtk_data.cell_data:
        radius = torch.tensor(np.asarray(vtk_data.cell_data['radius']), dtype=torch.float)
    else:
        radius = torch.tensor(np.asarray(vtk_data.cell_data['avg_radius']), dtype=torch.float)
    return nodes, edges, radius


def _save_vtp_file(nodes, edges, radius, filename, save_dir):
    mesh_edge = np.concatenate((np.int32(2 * np.ones((1, edges.shape[0]))), edges.T), 0)
    mesh = pyvista.UnstructuredGrid(mesh_edge.T, np.array([4] * len(radius)), nodes.numpy())
    mesh.cell_data['radius'] = radius.numpy()
    mesh_structured = mesh.extract_surface()
    mesh_structured.save(os.path.join(save_dir, filename))

@time_logging
def plot_num_edges_kde(folders, make_undirected=False):
    colors = ['r', 'g']
    labels = ['real', 'synth']
    plt.figure(figsize=(8, 6))
    num_edges_list_all = []
    for idx, folder in enumerate(folders):
        kde, num_edges = load_edges_kde(folder=folder, make_undirected=make_undirected)
        num_edges_list_all.append(num_edges)
        print(f"{min(num_edges)=}")
        # Generate a range of values for the PDF
        if idx == 0:
            x = np.linspace(min(num_edges), max(num_edges), 1000)
        plt.plot(x, kde(x), label=f"KDE_num_edges_{labels[idx]}", color=colors[idx], alpha=0.5 ** idx)
        plt.xlabel("Integer Values")
        plt.ylabel("Probability Density")
    plt.title("KDE for Number of edges")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # Adjust subplots for better spacing
    plt.show()
    print(
        f"KL divergence between num edges {kl_divergence_between_lists(num_edges_list_all[0], num_edges_list_all[1])}")


@time_logging
def plot_rad_kde(folders):
    plt.figure(figsize=(8, 6))
    labels = ['real', 'synth']
    colors = ['r', 'g']
    for idx, folder in enumerate(folders):
        rad_info = _load_rad_info(folder)
        kde = gaussian_kde(rad_info)
        # Generate a range of values for the PDF
        # min_rad=0, max_rad=23
        x = np.linspace(0, 23, 1000)
        plt.plot(x, kde(x), label=f"KDE_{labels[idx]}", color=colors[idx], alpha=0.5 ** idx)
    plt.xlabel("Integer Values")
    plt.ylabel("Probability Density")
    plt.title("KDE for radius")
    plt.legend()
    plt.grid(True)
    plt.show()


def load_kde(folder):
    num_nodes_arr = _load_num_nodes(folder=folder)
    # Calculate the PMF
    unique_values, counts = np.unique(num_nodes_arr, return_counts=True)
    pmf = counts / len(num_nodes_arr)
    # Step 2: Create the PDF using KDE
    kde = gaussian_kde(num_nodes_arr)
    return kde, num_nodes_arr


def load_edges_kde(folder, make_undirected=False):
    num_edges_arr = _load_num_edges(folder=folder, make_undirected=make_undirected)
    kde = gaussian_kde(num_edges_arr)
    return kde, num_edges_arr


def _load_num_nodes(folder):
    num_nodes_arr = []
    for filename in tqdm(glob.glob(f"{folder}/*.vtp")):
        nodes, edges, radius = _read_vtp_file(filename)
        num_nodes_arr.append(nodes.size(0))
    return num_nodes_arr


def _load_num_edges(folder, make_undirected):
    num_edges_arr = []
    for filename in tqdm(glob.glob(f"{folder}/*.vtp")):
        nodes, edges, radius = _read_vtp_file(filename)
        if make_undirected:
            edges = torch_geometric.utils.to_undirected(edge_index=edges)
        num_edges_arr.append(edges.size(1))
    return num_edges_arr


def _load_rad_info(folder):
    rad_arr = []
    ctr = 0
    min_rad, max_rad = float("inf"), -float("inf")
    for filename in tqdm(glob.glob(f"{folder}/*.vtp")):
        nodes, edges, radius = _read_vtp_file(filename)
        orig_radius = radius.clone()
        rad_arr.extend(radius.tolist())
        ctr += 1
        max_rad = max(torch.max(orig_radius).item(), max_rad)
        min_rad = min(torch.min(orig_radius).item(), min_rad)
    print(f"{min_rad=}")
    print(f"{max_rad=}")
    return rad_arr


def read_raw_coords(folder, mean_adjust, read_like_midi_data):
    all_coords = []
    max_coord, min_coord = -float("inf"), float("inf")
    for filename in tqdm(glob.glob(f"{folder}/*.vtp")):
        if read_like_midi_data:
            coords, _, _ = _read_vtp_file(filename)
            coords = coords - torch.mean(coords, dim=0, keepdim=True)
            max_distance = torch.max(torch.linalg.norm(coords, axis=1))
            coords = coords / max_distance
            all_coords.append(coords)
        else:
            coords, _, _ = _read_vtp_file(filename)
            if mean_adjust:
                coords = coords - torch.mean(coords, dim=0, keepdim=True)
            if torch.all(torch.max(coords, dim=0)[0] > max_coord):
                max_coord = torch.max(coords, dim=0)[0]
            if torch.all(torch.min(coords, dim=0)[0] < min_coord):
                min_coord = torch.min(coords, dim=0)[0]
            all_coords.append(coords)
    # print(f"{max_coord=}\n{min_coord=}")
    return torch.cat(all_coords)


def perform_coord_wise_kde(folders, min, max, read_like_midi_data):
    colors = ['r', 'g']
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    labels = ['real', 'synth']
    list_x, list_y, list_z = [], [], []
    for idx, folder in enumerate(folders):
        coords = read_raw_coords(folder=folder, mean_adjust=False,
                                 read_like_midi_data=read_like_midi_data)
        x_coords, y_coords, z_coords = coords[:, 0], coords[:, 1], coords[:, 2]
        x_coord_kde = gaussian_kde(x_coords.tolist())
        y_coord_kde = gaussian_kde(y_coords.tolist())
        z_coord_kde = gaussian_kde(z_coords.tolist())
        list_x.append(x_coords.tolist())
        list_y.append(y_coords.tolist())
        list_z.append(z_coords.tolist())
        pts = np.linspace(min, max, 2000)
        # pts = np.linspace(-400, 400, 2000)
        axes[0].plot(pts, x_coord_kde(pts), label=f"KDE_x_coords_{labels[idx]}", color=colors[idx], alpha=0.5 ** idx)
        axes[0].legend()
        axes[1].plot(pts, y_coord_kde(pts), label=f"KDE_y_coords_{labels[idx]}", color=colors[idx], alpha=0.5 ** idx)
        axes[1].legend()
        axes[2].plot(pts, z_coord_kde(pts), label=f"KDE_z_coords_{labels[idx]}", color=colors[idx], alpha=0.5 ** idx)
        axes[2].legend()
    plt.tight_layout()
    plt.show()
    coord_kl_div = (kl_divergence_between_lists(list_x[0], list_x[1]) + kl_divergence_between_lists(list_y[0],
                                                                                                    list_y[1])
                    + kl_divergence_between_lists(list_z[0], list_z[1])) / 3
    print(f"Coord kl divergence = {coord_kl_div}")


def read_edges(folder):
    edge_count = []
    for filename in tqdm(glob.glob(f"{folder}/*.vtp")):
        nodes, edges, radius = _read_vtp_file(filename)
        edge_count.append(edges.size(1))
        if 'synth' not in folder:
            if torch_geometric.utils.contains_isolated_nodes(edge_index=edges, num_nodes=nodes.size(0)):
                print(f"Isolated node in the ground truth {filename=}")
    return edge_count


def read_edge_types(folder):
    edge_types = []
    for filename in tqdm(glob.glob(f"{folder}/*.vtp")):
        nodes, edges, radius = _read_vtp_file(filename)
        radius = torch.ceil(radius)
        radius[radius >= 9] = 9
        edge_types.extend(radius.tolist())
    return edge_types


def plot_edge_kde(folders):
    colors = ['r', 'g']
    labels = ['real', 'synth']
    edge_max = 0
    for idx, folder in enumerate(folders):
        edge_counts_for_graph = read_edges(folder=folder)
        kde = gaussian_kde(edge_counts_for_graph)
        # Generate a range of values for the PDF
        if idx == 0:
            edge_max = max(edge_counts_for_graph)
        x = np.linspace(0, edge_max, 1000)
        plt.plot(x, kde(x), label=f"KDE_{labels[idx]}", color=colors[idx], alpha=0.5 ** idx)
    plt.xlabel("Num edges")
    plt.ylabel("Probability Density")
    plt.title("KDE for edge count")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_edge_type_kde(folders):
    colors = ['r', 'g']
    labels = ['real', 'synth']
    edge_max = 0
    for idx, folder in enumerate(folders):
        edge_types_for_graph = read_edge_types(folder=folder)
        kde = gaussian_kde(edge_types_for_graph)
        # Generate a range of values for the PDF
        if idx == 0:
            edge_max = max(edge_types_for_graph)
        x = np.linspace(0, edge_max, 100)
        plt.plot(x, kde(x), label=f"KDE_{labels[idx]}", color=colors[idx], alpha=0.5 ** idx)
    plt.xlabel("Num edges")
    plt.ylabel("Probability Density")
    plt.title("KDE for edge types")
    plt.legend()
    plt.grid(True)
    plt.show()


def read_edge_angles(folder, make_undirected):
    edge_angles = []
    for filename in tqdm(glob.glob(f"{folder}/*.vtp")):
        nodes, edges, radius = _read_vtp_file(filename)
        if make_undirected:
            edges, radius = torch_geometric.utils.to_undirected(edge_index=edges, edge_attr=radius)
        for idx in range(len(nodes)):
            p_a = nodes[idx]
            neighbors, _, _, _ = k_hop_subgraph(node_idx=idx, num_hops=1, edge_index=edges, relabel_nodes=False,
                                                flow='target_to_source')
            for i in neighbors:
                p_i = nodes[i]
                for j in neighbors:
                    if j == idx or i == j or i == idx:
                        continue
                    p_j = nodes[j]
                    v1 = p_i - p_a
                    v2 = p_j - p_a
                    assert not torch.isnan(v1).any()
                    assert not torch.isnan(v2).any()
                    prod = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-6)
                    if prod > 1:
                        print(f"Invalid angle {i} {j} -- {prod} -- {v1 / (torch.norm(v1) + 1e-6)} --"
                              f" {v2 / (torch.norm(v2) + 1e-6)}")
                    prod.clamp(min=0, max=1)
                    angle = torch.acos(prod)
                    if torch.isnan(angle).any():
                        print(f"Nan obtained in angle {i} {j} -- {prod} -- {v1 / (torch.norm(v1) + 1e-6)} --"
                              f" {v2 / (torch.norm(v2) + 1e-6)}")
                    else:
                        angle = angle * 180 / math.pi
                        edge_angles.append(angle)
    return edge_angles


def read_edge_lengths(folder, read_like_midi_data, make_undirected=False):
    edge_lengths = []
    for filename in tqdm(glob.glob(f"{folder}/*.vtp")):
        nodes, edges, radius = _read_vtp_file(filename)
        if make_undirected:
            edges = torch_geometric.utils.to_undirected(edge_index=edges)
        if read_like_midi_data:
            nodes = nodes - torch.mean(nodes, dim=0, keepdim=True)
            max_distance = torch.max(torch.linalg.norm(nodes, axis=1))
            nodes = nodes / max_distance
        edge_lengths.extend(torch.linalg.norm(nodes[edges[0]] - nodes[edges[1]], dim=1).tolist())
    return edge_lengths


def plot_edge_length_kde(folders, read_like_midi_data, make_undirected=False):
    colors = ['r', 'g']
    labels = ['real', 'synth']
    edge_length_arr = []
    for idx, folder in enumerate(folders):
        edge_length_array = read_edge_lengths(folder=folder, make_undirected=make_undirected,
                                              read_like_midi_data=read_like_midi_data)
        kde = gaussian_kde(edge_length_array)
        # Generate a range of values for the PDF
        edge_length_arr.append(edge_length_array)
        edge_max = max(edge_length_array)
        edge_min = min(edge_length_array)
        if idx == 0:
            kde_plot_points = edge_max
        x = np.linspace(0, kde_plot_points, 1000)
        plt.plot(x, kde(x), label=f"KDE_{labels[idx]}", color=colors[idx], alpha=0.5 ** idx)
    plt.xlabel("Edge length")
    plt.ylabel("Probability Density")
    plt.title("KDE for edge lengths")
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f"kl divergence between edge lengths {kl_divergence_between_lists(edge_length_arr[0], edge_length_arr[1])}")


def plot_edge_angles_kde(folders, make_undirected=False):
    colors = ['r', 'g']
    labels = ['real', 'synth']
    edge_angle_max = 0
    kl_div_lists = []
    for idx, folder in enumerate(folders):
        edge_angles_for_graph = read_edge_angles(folder=folder, make_undirected=make_undirected)
        kl_div_lists.append(edge_angles_for_graph)
        kde = gaussian_kde(edge_angles_for_graph)
        if idx == 0:
            edge_angle_max = max(edge_angles_for_graph)
            # print(f"{edge_angle_max=}")
        x = np.linspace(0, edge_angle_max, 1000)
        plt.plot(x, kde(x), label=f"KDE_{labels[idx]}", color=colors[idx], alpha=0.5 ** idx)
    plt.xlabel("Edge angles")
    plt.ylabel("Probability Density")
    plt.title("KDE for edge angles")
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f"Edge angles KL Div. {kl_divergence_between_lists(kl_div_lists[0], kl_div_lists[1])}")


def read_edge_angles_along_axes(folder, make_undirected=False):
    angles_with_axes = []
    for filename in tqdm(glob.glob(f"{folder}/*.vtp")):
        nodes, edges, radius = _read_vtp_file(filename)
        if make_undirected:
            edges, radius = torch_geometric.utils.to_undirected(edge_index=edges, edge_attr=radius)
        # Step 2: Identify pairs of nodes connected by each edge
        edge_pairs = edges.t()
        edge_vectors = nodes[edge_pairs[:, 1]] - nodes[edge_pairs[:, 0]]
        # Step 3: Compute vectors corresponding to each edge
        for vector in edge_vectors:
            for axis in torch.eye(3):
                dot_product = torch.dot(vector, axis)
                norm_vector = torch.norm(vector)
                norm_axis = torch.norm(axis)

                cosine_similarity = dot_product / (norm_vector * norm_axis)
                angle = torch.acos(torch.clamp(cosine_similarity, -1.0, 1.0))

                # Convert angle to degrees
                angle_degrees = np.degrees(angle.item())
                angles_with_axes.append(angle_degrees)
    # Reshape the angles for each axis
    angles_with_axes = torch.tensor(angles_with_axes).reshape(-1, 3)
    return angles_with_axes


def percentile(t, qs):
    resulting_values = []
    for q in qs:
        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        resulting_values.append(t.view(-1).kthvalue(k).values.item())
    return resulting_values


def _get_more_stats(value_list):
    if not isinstance(value_list, torch.Tensor):
        value_list = torch.as_tensor(value_list)
    # Calculate statistics
    mean_angle = torch.mean(value_list)
    median_angle = torch.median(value_list)
    std_dev_angle = torch.std(value_list)
    min_angle = torch.min(value_list)
    max_angle = torch.max(value_list)
    q25, q50, q75 = percentile(value_list, [25, 50, 75])
    return f"mean={mean_angle:.3f}; median={median_angle:.3f}; std_dev= {std_dev_angle:.3f};" \
           f" {q25=:.3f}; {q50=:.3f}; {q75=:.3f}; min={min_angle:.3f}; max={max_angle:.3f}"


def plot_edge_angles_along_axes_kde(folders, make_undirected=False):
    """
    Computes edge angles along coordinate axes for only the graphs with degree 2
    :param folders: list[folder]
    :param make_undirected: Make graph undirected. Default: False
    :return: None
    """
    colors = ['r', 'g']
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    labels = ['real', 'synth']
    max_angle = 0
    angle_lists1, angle_lists2, angle_lists3 = [], [], []
    for idx, folder in enumerate(folders):
        edge_angles_for_graph = read_edge_angles_along_axes(folder=folder, make_undirected=make_undirected)
        # We will get three kde plots here since we are working along the three axes
        x_axis_angles, y_axis_angles, z_axis_angles = edge_angles_for_graph[:, 0], edge_angles_for_graph[:,
                                                                                   1], edge_angles_for_graph[:, 2]
        x_axis_kde = gaussian_kde(x_axis_angles.tolist())
        y_axis_kde = gaussian_kde(y_axis_angles.tolist())
        z_axis_kde = gaussian_kde(z_axis_angles.tolist())
        angle_lists1.append(x_axis_angles)
        angle_lists2.append(y_axis_angles)
        angle_lists3.append(z_axis_angles)
        if idx == 0:
            max_angle = torch.max(edge_angles_for_graph).item()
            print(f"{max_angle=}")
        pts = np.linspace(0, max_angle, 1000)
        axes[0].plot(pts, x_axis_kde(pts), label=f"KDE_x_angles_{labels[idx]}", color=colors[idx], alpha=0.5 ** idx)
        axes[0].legend()
        axes[1].plot(pts, y_axis_kde(pts), label=f"KDE_y_angles_{labels[idx]}", color=colors[idx], alpha=0.5 ** idx)
        axes[1].legend()
        axes[2].plot(pts, z_axis_kde(pts), label=f"KDE_z_angles_{labels[idx]}", color=colors[idx], alpha=0.5 ** idx)
        axes[2].legend()
    plt.tight_layout()
    plt.show()
    # Calling kl divergence on all the three lists
    angle_kl = ((kl_divergence_between_lists(angle_lists1[0], angle_lists1[1]) +
                 kl_divergence_between_lists(angle_lists2[0], angle_lists2[1])
                 + kl_divergence_between_lists(angle_lists3[0], angle_lists3[1]))) / 3
    print(f"kl divergence for edge angles along axes {angle_kl}")


def compute_average_sparsity(folders):
    sample_type = ['real', 'synth']
    for idx, folder in enumerate(folders):
        avg_sparsity = ctr = 0
        for filename in tqdm(glob.glob(f"{folder}/*.vtp")):
            nodes, edges, radius = _read_vtp_file(filename)
            sparsity = edges.size(1) / (nodes.size(0) * nodes.size(0))
            avg_sparsity += sparsity
            ctr += 1
        print(f"{sample_type[idx]} has average sparsity = {avg_sparsity / ctr}")


def plot_node_length_and_angle_kde(folders, read_like_midi_data):
    colors = ['r', 'g']
    labels = ['real', 'synth']
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    for idx, folder in enumerate(folders):
        coords = read_raw_coords(folder=folder, mean_adjust=False,
                                 read_like_midi_data=read_like_midi_data)
        R = torch.linalg.norm(coords, dim=1)
        azimuths = torch.atan2(coords[:, 1], coords[:, 0])
        Phi = torch.acos(coords[:, 2] / R)
        r_coord_kde = gaussian_kde(R.tolist())
        azimuth_coord_kde = gaussian_kde(azimuths.tolist())
        phi_kde = gaussian_kde(Phi.tolist())
        pts = np.linspace(-1, 3, 1000)
        axes[0].plot(pts, r_coord_kde(pts), label=f"r_{labels[idx]}", color=colors[idx], alpha=0.5 ** idx)
        axes[0].legend()
        axes[1].plot(pts, azimuth_coord_kde(pts), label=f"azimuth_{labels[idx]}", color=colors[idx], alpha=0.5 ** idx)
        axes[1].legend()
        axes[2].plot(pts, phi_kde(pts), label=f"phi_{labels[idx]}", color=colors[idx], alpha=0.5 ** idx)
        axes[2].legend()
    plt.tight_layout()
    plt.show()


def compute_angles(coords, edges):
    # coordinates of interest are only the edge end points
    edge_vectors = coords[edges[:, 1]] - coords[edges[:, 0]]
    # compute the length of the edges
    R = torch.linalg.norm(edge_vectors, dim=1)
    azimuths = torch.atan2(edge_vectors[:, 1], edge_vectors[:, 0])
    phi = torch.acos(edge_vectors[:, 2] / R)
    return azimuths, phi


def compute_histogram(data, num_bins, bin_min, bin_max):
    histogram = torch.histc(data, bins=num_bins, min=bin_min, max=bin_max)
    return histogram / histogram.sum()


def kl_divergence_between_lists(data1, data2, num_bins=10, bin_min=None, bin_max=None):
    """
    Compute the KL divergence between two lists of unequal lengths using PyTorch.
    """
    print("Make sure first list is the GT")
    # Determine bins based on the combined range of both lists
    data1_tensor = torch.tensor(data1, dtype=torch.float)
    data2_tensor = torch.tensor(data2, dtype=torch.float)

    if bin_min is None and bin_max is None:
        bin_min, bin_max = data1_tensor.min(), data1_tensor.max()
    # Compute histograms
    histogram1 = compute_histogram(data1_tensor, num_bins, bin_min, bin_max)
    histogram2 = compute_histogram(data2_tensor, num_bins, bin_min, bin_max)

    # Compute KL divergence
    # Second term is the ground truth and the first term is the prediction
    # Adding a very small constant to make sure we do not get nan
    kl_div = torch.nn.functional.kl_div(torch.log(histogram2 + 1e-15), histogram1)
    return kl_div.item()  # Convert to Python scalar


def compute_betti_vals(edge_index):
    from midi.analysis.topological_analysis import SimplicialComplex
    edges = edge_index.T
    edge_list = edges.numpy().tolist()
    betti_val_info = SimplicialComplex(edge_list)
    bettis = defaultdict(list)
    for betti_number in [0, 1]:
        # A counter is internally a dictionary
        val = betti_val_info.betti_number(betti_number)
        bettis[betti_number].append(val)
    return bettis


def betti_kl_divergence(folders):
    print("Computation made on undirected graphs")
    betti_0_real, betti_1_real, betti_0_fake, betti_1_fake = [], [], [], []
    for idx, folder in enumerate(folders):
        for filename in tqdm(glob.glob(f"{folder}/*.vtp")):
            nodes, edges, radius = _read_vtp_file(filename)
            edges = torch_geometric.utils.to_undirected(edge_index=edges)
            bettis = compute_betti_vals(edge_index=edges)
            if idx == 0:
                betti_0_real.extend(bettis[0])
                x = [val for val in bettis[1] if val <= 2]
                # betti_1_real.extend(bettis[1])
                betti_1_real.extend(x)
            else:
                betti_0_fake.extend(bettis[0])
                betti_1_fake.extend(bettis[1])
    print(f"{betti_0_real=}")
    print(f"{betti_1_real=}")
    # print("Generated values")
    print(f"{betti_0_fake=}")
    print(f"{betti_1_fake=}")
    # Now compute the kl divergence
    print(
        f"Betti 0 KL divergence {kl_divergence_between_lists(betti_0_real, betti_0_fake, num_bins=5, bin_min=0.5, bin_max=5.5)}")
    print(
        f"Betti 1 KL divergence {kl_divergence_between_lists(betti_1_real, betti_1_fake, num_bins=3, bin_min=0.5, bin_max=3.5)}")


def load_cow_filenames(folder, save_dir='/mnt/elephant/chinmay/COW/all_vtps', degrees=0, rotate=False):
    all_graph_filenames = []
    for base, _, filename_list in os.walk(folder):
        for filename in filename_list:
            if filename.endswith('multi_met_graph.vtp'):
                all_graph_filenames.append(os.path.join(folder, base, filename))

    os.makedirs(save_dir, exist_ok=True)
    for filename in all_graph_filenames:
        from midi.datasets.cow_dataset import load_cow_file
        graph_data = load_cow_file(filename, directed=False)
        if rotate:
            # Rotate
            rotation_angle_x = torch.tensor(degrees)
            rotation_angle_x = torch.deg2rad(rotation_angle_x)
            rotation_matrix_x = torch.tensor([
                [1, 0, 0],
                [0, torch.cos(rotation_angle_x), -torch.sin(rotation_angle_x)],
                [0, torch.sin(rotation_angle_x), torch.cos(rotation_angle_x)],
            ], dtype=torch.float32)
            graph_data.pos = torch.matmul(graph_data.pos, rotation_matrix_x)
            # Add the mean back
        append_str = "Flip_" if "flip" in filename else ""
        dest_filename = append_str + os.path.split(filename)[1]
        _save_vtp_file(graph_data.pos, graph_data.edge_index.T, graph_data.edge_attr,
                       filename=os.path.join(save_dir, dest_filename))
        # shutil.copy2(filename, os.path.join(LOC_FOLDER, dest_filename))
    print(f"Filenames copied to {save_dir}")


def _load_node_degrees(folder, make_undirected=False):
    all_degree_list = []
    for filename in tqdm(glob.glob(f"{folder}/*.vtp")):
        nodes, edges, radius = _read_vtp_file(filename)
        if make_undirected:
            edges = torch_geometric.utils.to_undirected(edge_index=edges)
        dense_adj = torch_geometric.utils.to_dense_adj(edges, max_num_nodes=nodes.size(0)).squeeze(0)
        # degree = torch_geometric.utils.degree(index=edges[0], num_nodes=len(nodes))
        degree = dense_adj.sum(dim=1) + dense_adj.sum(dim=0)
        all_degree_list.extend(degree.tolist())
    return all_degree_list


def perform_node_degree_kde(folders, make_undirected=False):
    print("Using dense adj for undirected graph degree computation")
    colors = ['r', 'g']
    labels = ['real', 'synth']
    degrees = []
    for idx, folder in enumerate(folders):
        degree_list = _load_node_degrees(folder=folder, make_undirected=make_undirected)
        kde = gaussian_kde(degree_list)
        degrees.append(degree_list)
        x = np.linspace(0, 10, 1000)
        plt.plot(x, kde(x), label=f"KDE_{labels[idx]}", color=colors[idx], alpha=0.5 ** idx)
    plt.xlabel("Degree")
    plt.ylabel("Probability Density")
    plt.title("KDE for node degrees")
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f"kl divergence between node degrees {kl_divergence_between_lists(degrees[0], degrees[1])}")


def cow_stats(synth_folder):
    read_like_midi_data = True
    perform_coord_wise_kde(folders=[REAL_CoW, synth_folder], min=-0.5, max=0.5,
                           read_like_midi_data=read_like_midi_data)
    perform_node_degree_kde(folders=[REAL_CoW, synth_folder], make_undirected=True)
    plot_num_edges_kde(folders=[REAL_CoW, synth_folder])
    plot_edge_length_kde(folders=[REAL_CoW, synth_folder], read_like_midi_data=read_like_midi_data)
    plot_edge_angles_kde(folders=[REAL_CoW, synth_folder], make_undirected=True)
    plot_edge_angles_along_axes_kde(folders=[REAL_CoW, synth_folder], make_undirected=True)
    betti_kl_divergence(folders=[REAL_CoW, synth_folder])


def vessap_stats():
    print("Checking vessap stats")
    synth_folder = '/home/chinmayp/workspace/MiDi/outputs/midi_best_2024-03-06/18-56-40-graph-vessel-model' \
                   '/test_generated_samples_0/synthetic_data/vtp'
    read_like_midi_data = False
    perform_coord_wise_kde(folders=[REAL_VESSAP, synth_folder], min=0, max=1, read_like_midi_data=read_like_midi_data)
    perform_node_degree_kde(folders=[REAL_VESSAP, synth_folder], make_undirected=True)
    plot_num_edges_kde(folders=[REAL_VESSAP, synth_folder], make_undirected=True)
    plot_edge_length_kde(folders=[REAL_VESSAP, synth_folder], make_undirected=True,
                         read_like_midi_data=read_like_midi_data)
    plot_edge_angles_kde(folders=[REAL_VESSAP, synth_folder], make_undirected=True)
    plot_edge_angles_along_axes_kde(folders=[REAL_VESSAP, synth_folder], make_undirected=True)

    betti_kl_divergence(folders=[REAL_VESSAP, synth_folder])


def get_cow_stats():
    print("Checking cow stats")
    synth_folder = ('/home/chinmayp/workspace/MiDi_camera_ready/outputs/cow_multi_more_data2024-06-26/14-58-19-graph-vessel-model'
                    '/test_generated_samples_0/synthetic_data/vtp')
    cow_stats(synth_folder=synth_folder)


if __name__ == '__main__':
    get_cow_stats()
    # vessap_stats()
