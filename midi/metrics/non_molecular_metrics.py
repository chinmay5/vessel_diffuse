from collections import Counter

import hydra
import math
import networkx as nx
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from matplotlib import pyplot as plt
from torch import Tensor
from torch_geometric.utils import dense_to_sparse
from torchmetrics import MeanMetric
from tqdm import tqdm

from midi.analysis.topological_analysis import SimplicialComplex
from midi.datasets.dataset_utils import plot_list_of_dict_as_hist, plot_list_as_hist, load_pickle, plot_counter_as_hist
from midi.metrics.metrics_utils import counter_to_tensor, wasserstein1d, total_variation1d
from midi.utils import NoSyncMAE as MeanAbsoluteError
from midi.utils import NoSyncMetric as Metric


class NonMolecularSamplingMetrics(nn.Module):
    def __init__(self, dataset_infos, test):
        super().__init__()
        self.dataset_infos = dataset_infos
        self.atom_decoder = dataset_infos.atom_decoder

        self.test = test
        self.num_nodes_w1 = MeanMetric()
        self.atom_types_tv = MeanMetric()
        self.edge_types_tv = MeanMetric()
        self.bond_lengths_w1 = MeanMetric()
        self.angles_w1 = MeanMetric()

    def reset(self):
        for metric in [self.num_nodes_w1,
                       self.atom_types_tv, self.edge_types_tv,
                       self.bond_lengths_w1, self.angles_w1]:
            metric.reset()

    def __call__(self, generated_graph_info_list: list, name, current_epoch, local_rank):
        uniqueness = eval_fraction_unique(fake_graph_info_list=generated_graph_info_list)
        os.makedirs('graphs', exist_ok=True)
        stat = self.dataset_infos.statistics['test'] if self.test else self.dataset_infos.statistics['val']

        self.num_nodes_w1(number_nodes_distance(generated_graph_info_list, stat.num_nodes))

        atom_types_tv, atom_tv_per_class = node_types_distance(generated_graph_info_list, stat.atom_types,
                                                               save_histogram=self.test)
        self.atom_types_tv(atom_types_tv)
        edge_types_tv, bond_tv_per_class = edge_types_distance(generated_graph_info_list,
                                                               stat.bond_types,
                                                               save_histogram=self.test)
        self.edge_types_tv(edge_types_tv)

        bond_lengths_w1, bond_lengths_w1_per_type = edge_length_distance(generated_graph_info_list, stat.bond_lengths,
                                                                         stat.bond_types,
                                                                         num_edge_types=self.dataset_infos.edge_types.size(
                                                                             0))
        self.bond_lengths_w1(bond_lengths_w1)

        angles_w1, angles_w1_per_type = angle_distance(generated_graph_info_list, stat.bond_angles, stat.atom_types,
                                                       atom_decoder=self.dataset_infos.atom_decoder,
                                                       save_histogram=self.test)
        self.angles_w1(angles_w1)
        # Compute the topological metric but only for test split to save time.
        betti_diffs = None
        if stat.betti_vals is not None and self.test:
            betti_diffs = betti_val_distance(generated_graph_info_list, stat.betti_vals)
        to_log = {'sampling/NumNodesW1': self.num_nodes_w1.compute(),
                  'sampling/AtomTypesTV': self.atom_types_tv.compute(),
                  'sampling/EdgeTypesTV': self.edge_types_tv.compute(),
                  'sampling/BondLengthsW1': self.bond_lengths_w1.compute(),
                  'sampling/AnglesW1': self.angles_w1.compute(),
                  'sampling/Uniqueness': uniqueness * 100}
        if betti_diffs is not None:
            for betti_number, diff in betti_diffs.items():
                to_log[f'sampling/betti_{betti_number}'] = diff
        if local_rank == 0:
            print(f"Sampling metrics", {key: round(val.item(), 3) for key, val in to_log.items()})

        for i, atom_type in enumerate(self.dataset_infos.atom_decoder):
            to_log[f'sampling_per_class/{atom_type}_TV'] = atom_tv_per_class[i].item()
            to_log[f'sampling_per_class/{atom_type}_BondAnglesW1'] = angles_w1_per_type[i].item() \
                if angles_w1_per_type[i] != -1 else -1

        # for j, bond_type in enumerate(['No bond', 'Single', 'Double', 'Triple', 'Aromatic']):
        for j, bond_type in enumerate(torch.arange(self.dataset_infos.edge_types.size(0))):  # Since last term is excluded
            to_log[f'sampling_per_class/{bond_type}_TV'] = bond_tv_per_class[j].item()
            if j > 0:
                to_log[f'sampling_per_class/{bond_type}_BondLengthsW1'] = bond_lengths_w1_per_type[j - 1].item()

        if wandb.run:
            wandb.log(to_log, commit=False)
        if local_rank == 0:
            print(f"Sampling metrics done.")
        self.reset()
        plt.close()


def plot_num_node_diff(reference_n, generated_n):
    reference_n = reference_n / reference_n.sum()
    generated_n = generated_n / generated_n.sum()
    max_len = max(len(generated_n), len(reference_n))
    generated_n = F.pad(generated_n, (0, max_len - len(generated_n)))
    reference_n = F.pad(reference_n, (0, max_len - len(reference_n)))
    plot_list_as_hist(data=reference_n.numpy().tolist(), x_label='Num nodes', second_list=generated_n.numpy().tolist(),
                      name='Num_nodes_diff')


def number_nodes_distance(generated_graph_info_list, dataset_counts):
    max_number_nodes = max(dataset_counts.keys())
    reference_n = torch.zeros(max_number_nodes + 1)
    for n, count in dataset_counts.items():
        reference_n[n] = count

    c = Counter()
    for graph_info in generated_graph_info_list:
        nodes, _, _, _ = graph_info
        c[len(nodes)] += 1

    generated_n = counter_to_tensor(c)
    plot_num_node_diff(reference_n, generated_n)
    return wasserstein1d(generated_n, reference_n)


def plot_node_type_diff(generated_distribution, target):
    generated_dist = generated_distribution / generated_distribution.sum()
    target_dist = target / target.sum()
    plot_list_as_hist(data=target_dist.numpy().tolist(), x_label='Num nodes',
                      second_list=generated_dist.numpy().tolist(),
                      name='Node_type_diff')


def node_types_distance(generated_graph_info_list, target, save_histogram=False):
    generated_distribution = torch.zeros_like(target)
    for graph_info in generated_graph_info_list:
        nodes, edges, _, _ = graph_info
        for node_type in nodes.cpu().numpy().tolist():
            generated_distribution[node_type] += 1
    plot_node_type_diff(generated_distribution, target)
    if save_histogram:
        np.save('generated_atom_types.npy', generated_distribution.cpu().numpy())
    return total_variation1d(generated_distribution, target)


def plot_edge_type_diff(target, generated_distribution):
    generated_dist = generated_distribution / generated_distribution.sum()
    target_dist = target / target.sum()
    plot_list_as_hist(data=target_dist.cpu().numpy().tolist(), x_label='Num nodes',
                      second_list=generated_dist.cpu().numpy().tolist(),
                      name='Edge_type_diff')


def edge_types_distance(generated_graph_info_list, target, save_histogram=False):
    # nodes, edges, pos, num_atom_types
    device = generated_graph_info_list[0][0].device
    generated_distribution = torch.zeros_like(target).to(device)
    for graph_info in generated_graph_info_list:
        _, edges, _, _ = graph_info
        mask = torch.ones_like(edges)
        mask = torch.triu(mask, diagonal=1).bool()
        generated_edges = edges[mask]
        valid_edges = generated_edges[generated_edges > 0]
        generated_distribution[valid_edges] += 1
        # unique_edge_types, counts = torch.unique(edge_type, return_counts=True)
        # for type, count in zip(unique_edge_types, counts):
        #     generated_distribution[type] += count
        # count edge types
    if save_histogram:
        np.save('generated_bond_types.npy', generated_distribution.cpu().numpy())
    plot_edge_type_diff(target, generated_distribution)
    tv, tv_per_class = total_variation1d(generated_distribution, target.to(device))
    return tv, tv_per_class


def plot_bond_lengths(target_bond_lengths, generated_bond_lengths):
    target_bond_lengths_dict_of_list = {key: target_bond_lengths[key].cpu().numpy().tolist() for key in
                                        range(len(target_bond_lengths))}
    generated_bond_lengths_dict_of_list = {key: generated_bond_lengths[key].cpu().numpy().tolist() for key in
                                           range(len(generated_bond_lengths))}
    plot_list_of_dict_as_hist(data_dict=target_bond_lengths_dict_of_list,
                              second_dict=generated_bond_lengths_dict_of_list, name='bond_length_plot')


def edge_length_distance(generated_graph_info_list, target, bond_types_probabilities, num_edge_types):
    generated_edge_lenghts = {x: Counter() for x in range(1, num_edge_types)}  # Should be edge_attr
    for graph_info in generated_graph_info_list:
        node, edge_adj, pos, _ = graph_info
        cdists = torch.cdist(pos.unsqueeze(0),
                             pos.unsqueeze(0)).squeeze(0)
        for edge_type in range(1, num_edge_types):
            edges = torch.nonzero(edge_adj == edge_type)
            edge_distances = cdists[edges[:, 0], edges[:, 1]]
            distances_to_consider = torch.round(edge_distances, decimals=2)
            for d in distances_to_consider:
                generated_edge_lenghts[edge_type][d.item()] += 1

    # Normalizing the bond lenghts
    for edge_type in generated_edge_lenghts.keys():
        s = sum(generated_edge_lenghts[edge_type].values())
        if s == 0:
            s = 1
        for d, count in generated_edge_lenghts[edge_type].items():
            generated_edge_lenghts[edge_type][d] = count / s

    # Convert both dictionaries to tensors
    min_generated_length = min(min(d.keys()) if len(d) > 0 else 1e4 for d in generated_edge_lenghts.values())
    min_target_length = min(min(d.keys()) if len(d) > 0 else 1e4 for d in target.values())
    min_length = min(min_generated_length, min_target_length)

    max_generated_length = max(max(bl.keys()) if len(bl) > 0 else -1 for bl in generated_edge_lenghts.values())
    max_target_length = max(max(bl.keys()) if len(bl) > 0 else -1 for bl in target.values())
    max_length = max(max_generated_length, max_target_length)

    num_bins = int((max_length - min_length) * 100) + 1
    # Since we are ignoring edge type 0 (i.e. non-existing edges)
    generated_bond_lengths = torch.zeros(num_edge_types - 1, num_bins)
    target_bond_lengths = torch.zeros(num_edge_types - 1, num_bins)

    for edge_type in generated_edge_lenghts.keys():
        for d, count in generated_edge_lenghts[edge_type].items():
            bin = int((d - min_length) * 100)
            generated_bond_lengths[edge_type - 1, bin] = count
        for d, count in target[edge_type].items():
            bin = int((d - min_length) * 100)
            target_bond_lengths[edge_type - 1, bin] = count
    # Let us plot the results as well
    # plot_bond_lengths(target_bond_lengths, generated_bond_lengths)
    cs_generated = torch.cumsum(generated_bond_lengths, dim=1)
    cs_target = torch.cumsum(target_bond_lengths, dim=1)

    w1_per_class = torch.sum(torch.abs(cs_generated - cs_target), dim=1) / 100  # 100 because of bin size
    weighted = w1_per_class * bond_types_probabilities[1:]
    return torch.sum(weighted).item(), w1_per_class


def betti_val_distance(generated_graph_info_list, betti_vals):
    sampling_betti_val_dict = {idx: Counter() for idx in range(3)}
    for graph_info in tqdm(generated_graph_info_list):
        nodes, edges, pos, _ = graph_info
        edges = edges.cpu()
        edges, _ = dense_to_sparse(edges)
        edge_list = edges.T.numpy().tolist()
        # The class expects input in the shape [E, 2] and not dense adjacency
        betti_val_info = SimplicialComplex(edge_list)
        for betti_number in [0, 1, 2]:
            # A counter is internally a dictionary
            val = betti_val_info.betti_number(betti_number)
            sampling_betti_val_dict[betti_number][val] += 1
    # The betti numbers are computed. Now, we normalize them to convert it into a discrete pdf
    for betti_number in [0, 1, 2]:
        s = sum(sampling_betti_val_dict[betti_number].values())
        for component, count in sampling_betti_val_dict[betti_number].items():
            sampling_betti_val_dict[betti_number][component] = count / s
    # The two sets of pdf are available. Final step is to compute the Wasserestein distance
    # We can also plot the obtained distributions as images for convenience
    for idx in [0, 1]:
        plot_counter_as_hist(data_counter=betti_vals[idx], second_counter=sampling_betti_val_dict[idx],
                             name=f'betti_{idx}')
    betti_diffs = {}
    for betti_number in [0, 1, 2]:
        betti_compute = []
        test_counter, orig_dist_counter = sampling_betti_val_dict[betti_number], betti_vals[betti_number]
        for number in range(max(max(test_counter), max(orig_dist_counter))):
            betti_compute.append(abs(test_counter.get(number, 0) - orig_dist_counter.get(number, 0)))
        betti_diffs[betti_number] = torch.tensor(sum(betti_compute))
    return betti_diffs


def plot_bond_angles(target_angles, generated_angles):
    target_bond_angles_dict_of_list = {key: target_angles[key].tolist() for key in
                                       range(len(target_angles))}
    generated_bond_angles_dict_of_list = {key: generated_angles[key].tolist() for key in
                                          range(len(generated_angles))}
    plot_list_of_dict_as_hist(data_dict=target_bond_angles_dict_of_list,
                              second_dict=generated_bond_angles_dict_of_list, name='bond_angles_plot')


def angle_distance(generated_graph_info_list, target_angles, atom_types_probabilities, atom_decoder,
                   save_histogram: bool):
    num_atom_types = len(atom_types_probabilities)
    generated_angles = torch.zeros(num_atom_types, 180 * 10 + 1)
    for graph_info in generated_graph_info_list:
        node_types, adj, pos, _ = graph_info
        for node in range(adj.shape[0]):
            p_a = pos[node]
            neighbors = torch.nonzero(adj[node]).squeeze(1)
            for i in neighbors:
                p_i = pos[i]
                for j in neighbors:
                    if j == node or i == j or i == node:
                        continue
                    p_j = pos[j]
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
                        bin = int(torch.round(angle, decimals=1) * 10)
                        generated_angles[node_types[node].item(), bin] += 1

    s = torch.sum(generated_angles, dim=1, keepdim=True)
    s[s == 0] = 1
    generated_angles = generated_angles / s
    if save_histogram:
        np.save('generated_angles_historgram.npy', generated_angles.numpy())
    # Let us plot the bond angles as well
    # plot_bond_angles(target_angles, generated_angles)
    if type(target_angles) in [np.array, np.ndarray]:
        target_angles = torch.from_numpy(target_angles).float()

    cs_generated = torch.cumsum(generated_angles, dim=1)
    cs_target = torch.cumsum(target_angles, dim=1)

    w1_per_type = torch.sum(torch.abs(cs_generated - cs_target), dim=1) / 10

    weighted = w1_per_type * atom_types_probabilities
    return (torch.sum(weighted) / (torch.sum(atom_types_probabilities) + 1e-5)).item(), w1_per_type


# Some extra metric useful for non molecular graphs
def eval_fraction_isomorphic(fake_graphs, train_graphs):
    count = 0
    for fake_g in fake_graphs:
        for train_g in train_graphs:
            if nx.faster_could_be_isomorphic(fake_g, train_g):
                if nx.is_isomorphic(fake_g, train_g):
                    count += 1
                    break
    return count / float(len(fake_graphs))


def eval_fraction_unique(fake_graph_info_list, precise=False):
    count_non_unique = 0
    fake_evaluated = []
    for fake_g_info in fake_graph_info_list:
        nodes, edges, pos, _ = fake_g_info
        unique = True
        if not len(nodes) == 0:
            fake_g = to_networkx(node_list=nodes.cpu(), adjacency_matrix=edges.cpu())
            for fake_old in fake_evaluated:
                if precise:
                    if nx.faster_could_be_isomorphic(fake_g, fake_old):
                        if nx.is_isomorphic(fake_g, fake_old):
                            count_non_unique += 1
                            unique = False
                            break
                else:
                    if nx.faster_could_be_isomorphic(fake_g, fake_old):
                        if nx.could_be_isomorphic(fake_g, fake_old):
                            count_non_unique += 1
                            unique = False
                            break
            if unique:
                fake_evaluated.append(fake_g)

    frac_unique = (float(len(fake_graph_info_list)) - count_non_unique) / float(
        len(fake_graph_info_list))  # Fraction of distinct isomorphism classes in the fake graphs

    return torch.as_tensor(frac_unique)


def eval_fraction_unique_non_isomorphic_valid(fake_graphs, train_graphs, validity_func=(lambda x: True)):
    count_valid = 0
    count_isomorphic = 0
    count_non_unique = 0
    fake_evaluated = []
    for fake_g in fake_graphs:
        unique = True

        for fake_old in fake_evaluated:
            if nx.faster_could_be_isomorphic(fake_g, fake_old):
                if nx.is_isomorphic(fake_g, fake_old):
                    count_non_unique += 1
                    unique = False
                    break
        if unique:
            fake_evaluated.append(fake_g)
            non_isomorphic = True
            for train_g in train_graphs:
                if nx.faster_could_be_isomorphic(fake_g, train_g):
                    if nx.is_isomorphic(fake_g, train_g):
                        count_isomorphic += 1
                        non_isomorphic = False
                        break
            if non_isomorphic:
                if validity_func(fake_g):
                    count_valid += 1

    frac_unique = (float(len(fake_graphs)) - count_non_unique) / float(
        len(fake_graphs))  # Fraction of distinct isomorphism classes in the fake graphs
    frac_unique_non_isomorphic = (float(len(fake_graphs)) - count_non_unique - count_isomorphic) / float(
        len(fake_graphs))  # Fraction of distinct isomorphism classes in the fake graphs that are not in the training set
    frac_unique_non_isomorphic_valid = count_valid / float(
        len(fake_graphs))  # Fraction of distinct isomorphism classes in the fake graphs that are not in the training set and are valid
    return frac_unique, frac_unique_non_isomorphic, frac_unique_non_isomorphic_valid


class HistogramsMAE(MeanAbsoluteError):
    def __init__(self, target_histogram, **kwargs):
        """ Compute the distance between histograms. """
        super().__init__(**kwargs)
        assert (target_histogram.sum() - 1).abs() < 1e-3
        self.target_histogram = target_histogram

    def update(self, pred):
        pred = pred / pred.sum()
        target = self.target_histogram.to(pred.device)
        super().update(pred, target)


class CEPerClass(Metric):
    full_state_update = True

    def __init__(self, class_id):
        super().__init__()
        self.class_id = class_id
        self.add_state('total_ce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.softmax = torch.nn.Softmax(dim=-1)
        self.binary_cross_entropy = torch.nn.BCELoss(reduction='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.
        Args:
            preds: Predictions from model   (bs, n, d) or (bs, n, n, d)
            target: Ground truth values     (bs, n, d) or (bs, n, n, d)
        """
        target = target.reshape(-1, target.shape[-1])
        mask = (target != 0.).any(dim=-1)

        prob = self.softmax(preds)[..., self.class_id]
        prob = prob.flatten()[mask]

        target = target[:, self.class_id]
        target = target[mask]

        output = self.binary_cross_entropy(prob, target)
        self.total_ce += output
        self.total_samples += prob.numel()

    def compute(self):
        return self.total_ce / self.total_samples


class MeanNumberEdge(Metric):
    full_state_update = True

    def __init__(self):
        super().__init__()
        self.add_state('total_edge', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, molecules, weight=1.0) -> None:
        for molecule in molecules:
            _, edge_types = molecule
            triu_edge_types = torch.triu(edge_types, diagonal=1)
            bonds = torch.nonzero(triu_edge_types)
            self.total_edge += len(bonds)
        self.total_samples += len(molecules)

    def compute(self):
        return self.total_edge / self.total_samples


class TrainNonMolecularMetrics(nn.Module):
    def __init__(self, dataset_infos):
        # self.train_atom_metrics = AtomMetricsCE(dataset_infos=dataset_infos)
        # self.train_bond_metrics = BondMetricsCE()
        super().__init__()

    def forward(self, masked_pred, masked_true, log: bool):
        return None
        self.train_atom_metrics(masked_pred.X, masked_true.X)
        self.train_bond_metrics(masked_pred.E, masked_true.E)
        if not log:
            return

        to_log = {}
        for key, val in self.train_atom_metrics.compute().items():
            to_log['train/' + key] = val.item()
        for key, val in self.train_bond_metrics.compute().items():
            to_log['train/' + key] = val.item()
        if wandb.run:
            wandb.log(to_log, commit=False)
        return to_log

    def reset(self):
        return
        for metric in [self.train_atom_metrics, self.train_bond_metrics]:
            metric.reset()

    def log_epoch_metrics(self, current_epoch, local_rank):
        return
        epoch_atom_metrics = self.train_atom_metrics.compute()
        epoch_bond_metrics = self.train_bond_metrics.compute()

        to_log = {}
        for key, val in epoch_atom_metrics.items():
            to_log['train_epoch/' + key] = val.item()
        for key, val in epoch_bond_metrics.items():
            to_log['train_epoch/' + key] = val.item()

        if wandb.run:
            wandb.log(to_log, commit=False)

        for key, val in epoch_atom_metrics.items():
            epoch_atom_metrics[key] = round(val.item(), 3)
        for key, val in epoch_bond_metrics.items():
            epoch_bond_metrics[key] = round(val.item(), 3)
        print(f"Epoch {current_epoch} on rank {local_rank}: {epoch_atom_metrics} -- {epoch_bond_metrics}")

        return to_log


def to_networkx(node_list, adjacency_matrix):
    """
    Convert graphs to networkx graphs
    node_list: the nodes of a batch of nodes (bs x n)
    adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
    """
    graph = nx.Graph()

    for i in range(len(node_list)):
        if node_list[i] == -1:
            continue
        graph.add_node(i, number=i, symbol=node_list[i], color_val=node_list[i])

    rows, cols = np.where(adjacency_matrix >= 1)
    edges = zip(rows.tolist(), cols.tolist())
    for edge in edges:
        edge_type = adjacency_matrix[edge[0]][edge[1]]
        graph.add_edge(edge[0], edge[1], color=float(edge_type), weight=3 * edge_type)

    return graph


def test_metric(cfg):
    from midi.datasets.vessap_dataset import VessapGraphDataset, VessapGraphDataModule, VessapDatasetInfos
    from torch_geometric.utils import to_dense_adj
    datamodule = VessapGraphDataModule(cfg)
    dataset_config = cfg["dataset"]
    dataset_infos = VessapDatasetInfos(datamodule, dataset_config)
    metric = NonMolecularSamplingMetrics(dataset_infos, test=True)
    save_dir = '/home/chinmayp/workspace/MiDi/outputs/2023-09-20/18-13-46-graph-vessel-model'
    graph_list_filename = 'interpolated_test_samples.pkl'
    pyG_list = load_pickle(path=os.path.join(save_dir, graph_list_filename))
    # converting the pyg dataobjects into a list
    generated_graph_info_list = []
    for pyG_data in pyG_list:
        pos = pyG_data.x
        nodes = torch.randint(low=1, high=len(dataset_infos.atom_decoder), size=(pyG_data.x.size(0),))
        edges = to_dense_adj(edge_index=pyG_data.edge_index, edge_attr=pyG_data.edge_attr).squeeze(0)
        num_node_types = len(dataset_infos.atom_decoder)
        generated_graph_info_list.append((nodes, edges, pos, num_node_types))
    metric(generated_graph_info_list=generated_graph_info_list, name='sample_test', current_epoch=0, local_rank=0)


@hydra.main(version_base='1.3', config_path='../../configs', config_name='config')
def compute_distr_diff_on_splits(cfg):
    from midi.datasets.vessap_dataset import VessapGraphDataset, VessapGraphDataModule, VessapDatasetInfos
    from environment_setup import PROJECT_ROOT_DIR
    datamodule = VessapGraphDataModule(cfg)
    dataset_config = cfg["dataset"]
    dataset_infos = VessapDatasetInfos(datamodule, dataset_config)
    save_path = os.path.join(PROJECT_ROOT_DIR, 'data', 'vessel', 'converted_pt_files')
    train_dataset = VessapGraphDataset(dataset_name='sample', split='train', root=save_path)
    metric = NonMolecularSamplingMetrics(dataset_infos, test=True)
    compute_statistical_diff(dataset_infos, train_dataset, metric)
    # The comparison between test and val splits
    val_dataset = VessapGraphDataset(dataset_name='sample', split='val', root=save_path)
    compute_statistical_diff(dataset_infos, val_dataset, metric)
    # Finally, the comparison between train and val splits
    metric = NonMolecularSamplingMetrics(dataset_infos, test=False)
    compute_statistical_diff(dataset_infos, train_dataset, metric)


def compute_statistical_diff(dataset_infos, train_dataset, metric):
    from torch_geometric.utils import to_dense_adj
    generated_graph_info_list = []
    for idx in range(len(train_dataset)):
        pos = train_dataset[idx].pos
        nodes = torch.argmax(train_dataset[idx].x, dim=1)
        edges = to_dense_adj(edge_index=train_dataset[idx].edge_index,
                             edge_attr=torch.argmax(train_dataset[idx].edge_attr, dim=1)).squeeze(0)
        num_node_types = len(dataset_infos.atom_decoder)
        generated_graph_info_list.append((nodes, edges, pos, num_node_types))
    metric(generated_graph_info_list=generated_graph_info_list, name='sample_test', current_epoch=0, local_rank=0)


if __name__ == '__main__':
    compute_distr_diff_on_splits()
