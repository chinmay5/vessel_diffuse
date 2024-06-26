import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
import torch_geometric

from scipy.stats import gaussian_kde
from tqdm import tqdm

from post_process.stats_comparison import read_raw_coords, read_edge_lengths, read_edge_angles, \
    read_edge_angles_along_axes, _load_node_degrees, load_edges_kde, compute_betti_vals, _read_vtp_file

FONTSIZE = 20


def betti_0_and_1_plots(folders, axes, is_vessap=False):
    print("Computation made on undirected graphs")
    colors = ['tab:purple', 'tab:orange', 'tab:green']
    for idx, folder in enumerate(folders):
        betti_0, betti_1 = load_betti_numbers(folder, is_vessap=is_vessap)
        if idx == 0:
            alpha = 0.6
        elif idx == 1:
            alpha = 0.5
        else:
            alpha = 0.4
        # Define the bin ranges
        bins_betti_0 = np.arange(0, 20) - 0.5 if is_vessap else np.arange(0, 8) - 0.5
        bins_betti_1 = np.arange(0, 20) - 0.5 if is_vessap else np.arange(0, 4) - 0.5
        sns.histplot(betti_0, bins=bins_betti_0, ax=axes[2, 2], stat='density', kde=False,
                     color=colors[idx], alpha=alpha, edgecolor='none')
        sns.histplot(betti_1, bins=bins_betti_1, ax=axes[2, 3], stat='density', kde=False,
                     color=colors[idx], alpha=alpha, edgecolor='none')
    axes[2, 2].grid(True)
    axes[2, 3].grid(True)
    axes[2, 2].set_xlabel(r'$\it{\beta_0}$', fontsize=FONTSIZE)
    axes[2, 3].set_xlabel(r'$\it{\beta_1}$', fontsize=FONTSIZE)
    axes[2, 2].set_ylabel('')
    axes[2, 3].set_ylabel('')
    # Let us ensure that the x-ticks are always integer values
    # axes[2, 2].set_xticks(np.arange(0, max(betti_0), 2))
    # axes[2, 3].set_xticks(np.arange(0, max(betti_1), 2))


def load_betti_numbers(folder, is_vessap):
    betti_0, betti_1 = [], []
    for filename in tqdm(glob.glob(f"{folder}/*.vtp")):
        nodes, edges, radius = _read_vtp_file(filename)
        edges = torch_geometric.utils.to_undirected(edge_index=edges)
        bettis = compute_betti_vals(edge_index=edges)
        betti_0.extend(bettis[0])
        x = bettis[1] if is_vessap else [val for val in bettis[1] if val <= 2]
        # betti_1_real.extend(bettis[1])
        betti_1.extend(x)
    return betti_0, betti_1


def plot_node_degree_staircase(folders, axes, make_undirected=False, is_vessap=False):
    print("Using dense adj for undirected graph degree computation")
    colors = ['tab:purple', 'tab:orange', 'tab:green']
    labels = ['real', 'synth']
    print("Reducing degree list by a factor of 2")
    for idx, folder in enumerate(folders):
        degree_list = _load_node_degrees(folder=folder, make_undirected=make_undirected)
        # Reducing the degree list
        degree_list = [x / 2 for x in degree_list]
        if idx == 0:
            alpha = 0.6
        elif idx == 1:
            alpha = 0.5
        else:
            alpha = 0.4
        # Define the bin range
        bins = np.arange(8) - 0.5 if is_vessap else np.arange(6) - 0.5
        sns.histplot(degree_list, bins=bins, ax=axes[2, 0], stat='density', kde=False, color=colors[idx],
                     alpha=alpha, edgecolor='none')
    axes[2, 0].grid(True)
    axes[2, 0].set_ylabel('')
    axes[2, 0].set_xlabel(r'deg($\mathcal{V}$)', fontsize=FONTSIZE)


def plot_num_edges_staircase(folders, axes, make_undirected=False, is_vessap=False):
    colors = ['tab:purple', 'tab:orange', 'tab:green']
    labels = ['real', 'synth']
    print("reducing num edge by a factor of 2")
    for idx, folder in enumerate(folders):
        _, num_edges = load_edges_kde(folder=folder, make_undirected=make_undirected)
        num_edges = [x / 2 for x in num_edges]
        # Generate a range of values for the PDF
        # h, edges = np.histogram(num_edges, bins=np.linspace(0, 20, 10))
        # axes[2, 1].stairs(h, edges, color=colors[idx])
        if idx == 0:
            alpha = 0.6
        elif idx == 1:
            alpha = 0.5
        else:
            alpha = 0.4
        bins = np.arange(1, 80) - 0.5 if is_vessap else np.arange(5, 25) - 0.5
        sns.histplot(num_edges, bins=bins, ax=axes[2, 1], stat='density', kde=False,
                     color=colors[idx], alpha=alpha, edgecolor='none')
    axes[2, 1].grid(True)
    axes[2, 1].set_xlabel(r'|$\mathcal{E}$|', fontsize=FONTSIZE)
    axes[2, 1].set_ylabel('')


def plot_edge_angles_kde(folders, axes, make_undirected=True):
    colors = ['tab:purple', 'tab:orange', 'tab:green']
    labels = ['Real', 'MiDi', 'Ours']
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
        # Putting label here for eventual plot
        axes[1, 3].plot(x, kde(x), color=colors[idx], label=f"{labels[idx]}")
    axes[1, 3].grid(True)
    axes[1, 3].set_xlabel(r'$\mathcal{E}~\angle$', fontsize=FONTSIZE)


def plot_edge_length_kde(folders, axes, make_undirected=True):
    colors = ['tab:purple', 'tab:orange', 'tab:green']
    labels = ['real', 'midi', 'ours']
    edge_length_arr = []
    for idx, folder in enumerate(folders):
        edge_length_array = read_edge_lengths(folder=folder, make_undirected=make_undirected, read_like_midi_data=False)
        kde = gaussian_kde(edge_length_array)
        # Generate a range of values for the PDF
        edge_length_arr.append(edge_length_array)
        edge_max = max(edge_length_array)
        if idx == 0:
            kde_plot_points = edge_max
        x = np.linspace(0, kde_plot_points, 1000)
        axes[0, 3].plot(x, kde(x), color=colors[idx])
    axes[0, 3].grid(True)
    axes[0, 3].set_xlabel(r'$l_\mathcal{E}$', fontsize=FONTSIZE)


def perform_coord_wise_kde(folders, axes, is_vessap):
    colors = ['tab:purple', 'tab:orange', 'tab:green']
    labels = ['real', 'midi', 'ours']
    list_x, list_y, list_z = [], [], []
    for idx, folder in enumerate(folders):
        coords = read_raw_coords(folder=folder, mean_adjust=False, read_like_midi_data=False)
        x_coords, y_coords, z_coords = coords[:, 0], coords[:, 1], coords[:, 2]
        if is_vessap:
            # We normalize the coordinates so that they lie between -0.5 to 0.5
            x_coords, y_coords, z_coords = x_coords - 0.5, y_coords - 0.5, z_coords - 0.5
        x_coord_kde = gaussian_kde(x_coords.tolist())
        y_coord_kde = gaussian_kde(y_coords.tolist())
        z_coord_kde = gaussian_kde(z_coords.tolist())
        list_x.append(x_coords.tolist())
        list_y.append(y_coords.tolist())
        list_z.append(z_coords.tolist())
        pts = np.linspace(-0.5, 0.5, 1000)
        axes[0, 0].plot(pts, x_coord_kde(pts), color=colors[idx], label=f"{labels[idx]}")
        axes[0, 1].plot(pts, y_coord_kde(pts), color=colors[idx])
        axes[0, 2].plot(pts, z_coord_kde(pts), color=colors[idx])
    # Turn on the grids
    axes[0, 0].grid(True)
    axes[0, 1].grid(True)
    axes[0, 2].grid(True)
    # Give the labels
    axes[0, 0].set_xlabel(r'$x$', fontsize=FONTSIZE)
    axes[0, 1].set_xlabel(r'$y$', fontsize=FONTSIZE)
    axes[0, 2].set_xlabel(r'$z$', fontsize=FONTSIZE)
    # plt.show()


def plot_edge_angles_along_axes_kde(folders, axes, make_undirected=True):
    """
    Computes edge angles along coordinate axes for only the graphs with degree 2
    :param folders: list[folder]
    :param make_undirected: Make graph undirected. Default: False
    :return: None
    """
    colors = ['tab:purple', 'tab:orange', 'tab:green']
    labels = ['real', 'midi', 'ours']
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
        axes[1, 0].plot(pts, x_axis_kde(pts), color=colors[idx])
        axes[1, 1].plot(pts, y_axis_kde(pts), color=colors[idx])
        axes[1, 2].plot(pts, z_axis_kde(pts), color=colors[idx])
    # Turning on the grid
    axes[1, 0].grid(True)
    axes[1, 1].grid(True)
    axes[1, 2].grid(True)
    # We give the labels
    axes[1, 0].set_xlabel(r'$\theta$', fontsize=FONTSIZE)
    axes[1, 1].set_xlabel(r'$\phi$', fontsize=FONTSIZE)
    axes[1, 2].set_xlabel(r'$\psi$', fontsize=FONTSIZE)



def cow_plots():
    ours = '/home/chinmayp/workspace/MiDi/outputs/crown_more_data2024-04-09/midi_and_ours/ours/vtp'
    midi = '/home/chinmayp/workspace/MiDi/outputs/crown_more_data2024-04-09/midi_and_ours/midi/vtp'
    gt = '/mnt/elephant/chinmay/COWN_plus_top_CoW/all_vtps'
    # &\textit{$x,y,z$} &\textit{deg($\mathcal{V}$)}  & |$\mathcal{E}$| & $l_\mathcal{E}$ & $\mathcal{E} ~\angle$& \textit{$\theta,\phi,\psi$}&\textit{$\beta_0$} &\textit{$\beta_1$}\\
    fig, axes = plt.subplots(3, 4, figsize=(16, 8))
    perform_coord_wise_kde([gt, midi, ours], axes, is_vessap=False)
    plot_edge_angles_along_axes_kde([gt, midi, ours], axes)
    plot_edge_angles_kde([gt, midi, ours], axes)
    plot_edge_length_kde([gt, midi, ours], axes)
    # The discrete features
    plot_node_degree_staircase([gt, midi, ours], axes, make_undirected=False, is_vessap=False)
    plot_num_edges_staircase([gt, midi, ours], axes, make_undirected=False, is_vessap=False)
    betti_0_and_1_plots([gt, midi, ours], axes, is_vessap=False)
    # Get the handle and the final figure
    # Add legend
    handles, labels = axes[1, 3].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper right')
    plt.legend(handles, labels, bbox_to_anchor=(1.05, 1.05), loc='upper left', borderaxespad=0., fontsize=16)
    plt.tight_layout()
    plt.savefig('cow.jpg', format='jpg', dpi=250)
    plt.show()


def vessap_plots():
    ours = '/home/chinmayp/workspace/MiDi/outputs/our_best_2024-02-17/11-19-54-graph-vessel-model'\
            '/test_generated_samples_0/synthetic_data/vtp'
    midi = '/home/chinmayp/workspace/MiDi/outputs/midi_best_2024-03-06/18-56-40-graph-vessel-model' \
           '/test_generated_samples_0/synthetic_data/vtp'
    gt = '/mnt/elephant/chinmay/midi_vessap/non_neg_rad_graph/train_data'
    fig, axes = plt.subplots(3, 4, figsize=(16, 8))
    perform_coord_wise_kde([gt, midi, ours], axes, is_vessap=True)
    plot_edge_angles_along_axes_kde([gt, midi, ours], axes)
    plot_edge_angles_kde([gt, midi, ours], axes)
    plot_edge_length_kde([gt, midi, ours], axes)
    # The discrete features
    plot_node_degree_staircase([gt, midi, ours], axes, make_undirected=True, is_vessap=True)
    plot_num_edges_staircase([gt, midi, ours], axes, make_undirected=True, is_vessap=True)
    betti_0_and_1_plots([gt, midi, ours], axes, is_vessap=True)
    # Get the handle and the final figure
    # Add legend
    handles, labels = axes[1, 3].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper right')
    plt.legend(handles, labels, bbox_to_anchor=(1.05, 1.05), loc='upper left', borderaxespad=0., fontsize=16)
    plt.tight_layout()
    plt.savefig('vessap.jpg', format='jpg', dpi=250)
    plt.show()


if __name__ == "__main__":
    cow_plots()
    # vessap_plots()
