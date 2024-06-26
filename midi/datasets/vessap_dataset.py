import math
import os
import pathlib
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric
from torch.utils.data import Dataset
from torch_geometric.transforms import RemoveDuplicatedEdges, BaseTransform
from torch_geometric.utils import k_hop_subgraph, contains_isolated_nodes, \
    contains_self_loops
from tqdm import tqdm

from environment_setup import PROJECT_ROOT_DIR
from midi.analysis.topological_analysis import SimplicialComplex
from midi.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from midi.datasets.dataset_utils import Statistics, save_pickle, load_pickle, plot_list_as_hist, \
    plot_list_of_dict_as_hist, plot_counter_as_hist
from midi.utils import PlaceHolder


class VessapGraphDataset(Dataset):
    def __init__(self, dataset_name, split, root, transform=None, pre_transform=None, pre_filter=None, cfg=None):
        super(Dataset, self).__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        self.cfg = cfg
        base_load_dir = os.path.basename(root)
        self.preprocssed_graph_load_dir = os.path.join(PROJECT_ROOT_DIR, 'data', 'vessap', base_load_dir)
        self.raw_dir = os.path.join(PROJECT_ROOT_DIR, 'data', 'vessap', f'raw_{base_load_dir}')
        os.makedirs(self.raw_dir, exist_ok=True)
        # self.data, self.slices = torch.load(self.processed_paths[0])
        file_idx = {'train': 0, 'val': 1, 'test': 2}
        indices_file = self.raw_paths[file_idx[self.split]]
        if not os.path.exists(indices_file):
            self.download()
        self.indices = pd.read_csv(indices_file)["filenames"].tolist()
        # We also encode the number of "atom" types.
        # In our case, these are the node degree types
        # Load any data object
        sample_data = torch.load(os.path.join(self.preprocssed_graph_load_dir, self.indices[0]))
        self.num_degree_categories = sample_data.x.shape[1]
        # TODO: Fix this
        self.num_edge_categories = 4  # sample_data.edge_attr.shape[1]
        self.num_avg_diameter_types = sample_data.avg_diameter.size(1)
        # We can use custom names here if needed but degree in themselves are self-explanatory
        self.vessel_node_types = {x: x for x in range(self.num_degree_categories)}
        # Include the statistics for the dataset
        self.is_directed = 'directed' in base_load_dir
        stats_path = os.path.join(PROJECT_ROOT_DIR, 'data', 'vessap', f'statistics_{base_load_dir}', f"{split}.pkl")
        if not os.path.exists(stats_path):
            os.makedirs(os.path.dirname(stats_path), exist_ok=True)
            self.compute_statistics(stats_path)
        self.statistics = load_pickle(stats_path)
        self.cleanup_transform = RemoveDuplicatedEdges(key='edge_attr', reduce='min')

    @property
    def raw_file_names(self):
        return ['train_filename.csv', 'val_filename.csv', 'test_filename.csv']

    @property
    def raw_paths(self):
        return [os.path.join(self.raw_dir, f) for f in self.raw_file_names]

    @property
    def processed_file_names(self):
        return [self.split + '_filename.csv']

    def download(self):
        print("Creating fresh data split")
        all_graph_filenames = list(
            filter(lambda x: os.path.isfile(os.path.join(self.preprocssed_graph_load_dir, x)),
                   os.listdir(self.preprocssed_graph_load_dir)))
        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)
        num_graphs = len(all_graph_filenames)
        if num_graphs == 1:
            # This is debug mode
            print(f'Debug mode:- Dataset sizes: train 1, val 1, test 1')

            graph_data = torch.load(os.path.join(self.preprocssed_graph_load_dir, all_graph_filenames[0]))
            graph_filename = all_graph_filenames[0]
            if not hasattr(graph_data, 'y'):
                # Provide a dummy graph level label
                graph_data.y = torch.zeros((1, 0), dtype=torch.float)
                torch.save(graph_data, os.path.join(self.preprocssed_graph_load_dir, graph_filename))
            train_filenames, val_filenames, test_filenames = [graph_filename], [graph_filename], [graph_filename]
        else:
            train_filenames, val_filenames, test_filenames = [], [], []
            test_len = int(round(num_graphs * 0.1))
            train_len = int(round((num_graphs - test_len) * 0.9))
            val_len = num_graphs - train_len - test_len
            # For 100 -> 81 train, 9 val and 10 test
            indices = torch.randperm(num_graphs, generator=g_cpu)
            print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
            train_indices = indices[:train_len]
            val_indices = indices[train_len:train_len + val_len]
            test_indices = indices[train_len + val_len:]

            for i, graph_filename in enumerate(all_graph_filenames):
                if i in train_indices:
                    train_filenames.append(graph_filename)
                elif i in val_indices:
                    val_filenames.append(graph_filename)
                elif i in test_indices:
                    test_filenames.append(graph_filename)
                else:
                    raise ValueError(f'Index {i} not in any split')
                # We have decided the splits. Now we do the last bit of processing
                graph_data = torch.load(os.path.join(self.preprocssed_graph_load_dir, graph_filename))
                graph_data.y = torch.zeros((1, 0), dtype=torch.float)
                torch.save(graph_data, os.path.join(self.preprocssed_graph_load_dir, graph_filename))

        # Convert the list to a pandas DataFrame
        train_data = pd.DataFrame({"filenames": train_filenames})
        val_data = pd.DataFrame({"filenames": val_filenames})
        test_data = pd.DataFrame({"filenames": test_filenames})
        # Save the DataFrame to a CSV file
        train_data.to_csv(self.raw_paths[0], index=False)
        val_data.to_csv(self.raw_paths[1], index=False)
        test_data.to_csv(self.raw_paths[2], index=False)

    def __getitem__(self, item):
        file_name = self.indices[item]
        graph_data = torch.load(os.path.join(self.preprocssed_graph_load_dir, file_name))
        # TODO: Fix later
        graph_data.charges = graph_data.avg_diameter
        # graph_data = self.cleanup_transform(graph_data)
        if self.split == 'train':
            if self.transform is not None:
                graph_data = self.transform(graph_data)

        return graph_data

    def __len__(self):
        # return 1
        return len(self.indices)

    def compute_statistics(self, stats_path):
        # We already have indices.
        print(f"Computing statistics for {self.split=}")
        num_nodes = Counter()
        atom_types = torch.zeros(self.num_degree_categories)
        edge_types = torch.zeros(self.num_edge_categories)
        avg_diameter_types = torch.zeros(self.num_avg_diameter_types)
        # Not sure why the angles discretization is so large.
        all_edge_angles = np.zeros((self.num_degree_categories, 180 * 10 + 1))
        # Compute the bond lengths separately for each bond type
        all_edge_lengths = {idx: Counter() for idx in range(self.num_edge_categories)}
        # Let us also compute values for the betti numbers
        betti_val_dict = {idx: Counter() for idx in range(3)}  # betti 0, 1 and 2
        cc_cond_N, cycles_cond_N, global_feat, deg_cond_N = None, None, False, None
        for file_name in tqdm(self.indices):
            graph_data = torch.load(os.path.join(self.preprocssed_graph_load_dir, file_name))
            atom_types += graph_data.x.sum(dim=0).numpy()
            # count edge types
            edge_types += F.one_hot(graph_data.edge_attr, num_classes=self.num_edge_categories).sum(dim=0)
            # We count the different avg diameter types
            avg_diameter_types += graph_data.avg_diameter.sum(dim=0).numpy()
            # count the number of nodes
            N = graph_data.x.size(0)
            num_nodes[N] += 1
            self.update_edge_length_info(graph_data, all_edge_lengths)
            self.update_edge_angle_info(graph_data, all_edge_angles)
            self.update_betti_vals(graph_data, betti_val_dict)

        atom_types = atom_types / atom_types.sum()
        edge_types = edge_types / edge_types.sum()
        avg_diameter_types = avg_diameter_types / avg_diameter_types.sum()
        edge_lengths = self.normalize_edge_lengths(all_edge_lengths)
        edge_angles = self.normalize_edge_angles(all_edge_angles)
        betti_vals = self.normalize_betti_vals(betti_val_dict)
        # We have computed all the statistics now
        stats = Statistics(num_nodes=num_nodes, atom_types=atom_types, bond_types=edge_types,
                           bond_lengths=edge_lengths, bond_angles=edge_angles, betti_vals=betti_vals,
                           charge_types=avg_diameter_types, cc_cond_N=cc_cond_N, cycles_cond_N=cycles_cond_N,
                           deg_cond_N=deg_cond_N)
        print(stats)
        save_pickle(stats, stats_path)

    def update_edge_length_info(self, graph_data, all_edge_lengths):
        cdists = torch.cdist(graph_data.pos.unsqueeze(0), graph_data.pos.unsqueeze(0)).squeeze(0)
        edge_distances = cdists[graph_data.edge_index[0], graph_data.edge_index[1]]
        for edge_type in range(self.num_edge_categories):
            # bond_type_mask = torch.argmax(graph_data.edge_attr, dim=1) == edge_type
            bond_type_mask = graph_data.edge_attr == edge_type
            distances_to_consider = edge_distances[bond_type_mask]
            distances_to_consider = torch.round(distances_to_consider, decimals=2)
            for d in distances_to_consider:
                all_edge_lengths[edge_type][d.item()] += 1

    def normalize_edge_lengths(self, all_edge_lengths):
        for bond_type in range(self.num_edge_categories):
            s = sum(all_edge_lengths[bond_type].values())
            for d, count in all_edge_lengths[bond_type].items():
                all_edge_lengths[bond_type][d] = count / s
        return all_edge_lengths

    def update_edge_angle_info(self, graph_data, all_edge_angles):
        assert not torch.isnan(graph_data.pos).any()
        node_types = torch.argmax(graph_data.x, dim=1)
        for i in range(graph_data.x.size(0)):
            neighbors, _, _, _ = k_hop_subgraph(i, num_hops=1, edge_index=graph_data.edge_index,
                                                relabel_nodes=False, directed=self.is_directed,
                                                num_nodes=graph_data.x.size(0), flow='target_to_source')
            # All the degree one nodes are unfortunately skipped in this evaluation.
            # Hence, it is more about non-degree 1 nodes and their connectivity
            for j in neighbors:
                for k in neighbors:
                    if j == k or i == j or i == k:
                        continue
                    assert i != j and i != k and j != k, "i, j, k: {}, {}, {}".format(i, j, k)
                    a = graph_data.pos[j] - graph_data.pos[i]
                    b = graph_data.pos[k] - graph_data.pos[i]

                    # print(a, b, torch.norm(a) * torch.norm(b))
                    angle = torch.acos(torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-6))
                    angle = angle * 180 / math.pi

                    bin = int(torch.round(angle, decimals=1) * 10)

                    all_edge_angles[node_types[i].item(), bin] += 1

    def normalize_edge_angles(self, all_edge_angles):
        s = all_edge_angles.sum(axis=1, keepdims=True)
        s[s == 0] = 1
        all_bond_angles = all_edge_angles / s
        return all_bond_angles

    def update_betti_vals(self, graph_data, betti_val_dict):
        edges = graph_data.edge_index.T
        edge_list = edges.numpy().tolist()
        betti_val_info = SimplicialComplex(edge_list)
        for betti_number in [0, 1, 2]:
            # A counter is internally a dictionary
            val = betti_val_info.betti_number(betti_number)
            betti_val_dict[betti_number][val] += 1

    def normalize_betti_vals(self, betti_val_dict):
        """
        Structure is as follows
        {
        Betti 0:
            1 -> 10,
            2 -> 8
            3 -> 25
        Betti 1:
            1 -> 5
            2 -> 5
            ...
        }
        So, we are computing the probability distribution for this discrete random variable.
        """
        # Again, category 0 is the "pseudo connection"
        for betti_number in [0, 1, 2]:
            s = sum(betti_val_dict[betti_number].values())
            for component, count in betti_val_dict[betti_number].items():
                betti_val_dict[betti_number][component] = count / s
        return betti_val_dict


class VessapGraphDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs=200):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)

        train_dataset = VessapGraphDataset(dataset_name=self.cfg.dataset.name,
                                           split='train', root=cfg.dataset.datadir, cfg=cfg)
        val_dataset = VessapGraphDataset(dataset_name=self.cfg.dataset.name,
                                         split='val', root=cfg.dataset.datadir, cfg=cfg)
        test_dataset = VessapGraphDataset(dataset_name=self.cfg.dataset.name,
                                          split='test', root=cfg.dataset.datadir, cfg=cfg)
        self.statistics = {'train': train_dataset.statistics, 'val': val_dataset.statistics,
                           'test': test_dataset.statistics}

        super().__init__(cfg, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)
        self.node_encoder = self.train_dataset.vessel_node_types


class VessapDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        self.datamodule = datamodule
        self.statistics = datamodule.statistics
        self.name = 'vessel_graphs'
        super().complete_infos(datamodule.statistics, datamodule.node_encoder)

        print("Distribution of number of nodes", self.n_nodes)
        np.savetxt('n_counts.txt', self.n_nodes.numpy())
        print("Distribution of node types", self.atom_types)
        np.savetxt('atom_types.txt', self.atom_types.numpy())
        print("Distribution of edge types", self.edge_types)
        np.savetxt('edge_types.txt', self.edge_types.numpy())
        print("Distribution of charge types", self.charges_types)
        np.savetxt('charge_types.txt', self.charges_types.numpy())

        self.directed = cfg.dataset.get("is_directed", False)
        y_out = 0
        y_in = 1
        self.input_dims = PlaceHolder(X=self.num_atom_types, E=self.edge_types.size(0), y=y_in, pos=3,
                                      directed=self.directed, charges=self.charges_types.size(0))
        self.output_dims = PlaceHolder(X=self.num_atom_types, E=self.edge_types.size(0), y=y_out, pos=3,
                                       directed=self.directed, charges=self.charges_types.size(0))
        self.collapse_charges = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).int()
        print(f"VESSEL DATASET INFOS is directed: {self.directed}")

    def to_one_hot(self, X, charges, E, node_mask):
        E = F.one_hot(E, num_classes=self.edge_types.size(0)).float()
        placeholder = PlaceHolder(X=X, charges=charges, E=E, y=None, pos=None, directed=self.directed)
        pl = placeholder.mask(node_mask)
        return pl.X, pl.charges, pl.E


import hydra


@hydra.main(version_base='1.3', config_path='../../configs', config_name='config')
def check_deg_c_N(cfg):
    datamodule = VessapGraphDataModule(cfg)
    dataset_infos = VessapDatasetInfos(datamodule, cfg)
    node_dist_model = dataset_infos.nodes_dist
    deg_cond_N = dataset_infos.statistics['train'].deg_cond_N
    print(datamodule.train_dataset[0])


def utility_fn(graph_data):
    self_loop = contains_self_loops(graph_data.edge_index)
    list_of_tuple_edges = graph_data.edge_index.T.numpy().tolist()
    edge_dict = defaultdict(int)
    for edges in list_of_tuple_edges:
        edge_dict[tuple(edges)] += 1
    dup_edges = [(edge, count) for edge, count in edge_dict.items()
                 if count > 1]
    return self_loop, dup_edges


def compute_min_max_pos(split):
    save_path = os.path.join(PROJECT_ROOT_DIR, 'data', 'vessap', 'converted_laplacian_pt_files')
    dataset = VessapGraphDataset(dataset_name='sample', split=split, root=save_path)
    min_val, max_val = float("inf"), -float("inf")
    radius_info = []
    num_node = []
    isolated_node = 0
    degree = []
    for idx in tqdm(range(len(dataset))):
        graph_data = dataset[idx]
        self_loop, count_arr = utility_fn(graph_data=graph_data)
        if self_loop:
            print(f"{graph_data=}")
        if len(count_arr) > 0:
            print(f"{count_arr=}")
        if idx <= 5:
            print(graph_data.pos)
            print(f"{torch.mean(graph_data.pos, dim=0)=}")
        min_val = min(torch.min(graph_data.pos), min_val)
        max_val = max(torch.max(graph_data.pos), max_val)
        radius_info.extend(graph_data.edge_attr.cpu().numpy().tolist())
        num_node.append(graph_data.x.size(0))
        isolated_node += contains_isolated_nodes(graph_data.edge_index)
        degree.extend(torch_geometric.utils.degree(graph_data.edge_index[1]).numpy().tolist())
    print(f"{split=} - {min_val=} and {max_val=}")
    print(f"{Counter(radius_info)=}")
    print(f"{sorted(Counter(num_node))=}")
    print(f"{sorted(Counter(degree))=}")
    print(f"{isolated_node=}")
    return dataset


def plot_stats(dataset):
    stats = dataset.statistics
    # We can plot all the values for comparison
    plot_list_as_hist(data=stats.num_nodes, x_label='Num nodes')
    plot_list_as_hist(data=stats.bond_types, x_label='Num edges')
    plot_list_of_dict_as_hist(data_dict=stats.betti_vals)
    # plot_list_of_dict_as_hist(data_dict=stats.bond_angles)
    bond_lengths_dict_of_list = {key: list(c.values()) for key, c in stats.bond_lengths.items()}
    plot_list_of_dict_as_hist(data_dict=bond_lengths_dict_of_list)
    # Doing the same for the bond angles
    bond_angles_dict_of_list = {key: stats.bond_angles[key] for key in range(len(stats.bond_angles))}
    plot_list_of_dict_as_hist(data_dict=bond_angles_dict_of_list)


def plot_stat_comp():
    save_path = os.path.join(PROJECT_ROOT_DIR, 'data', 'vessap', 'converted_pt_files')
    val_dataset = VessapGraphDataset(dataset_name='sample', split='val', root=save_path)
    test_dataset = VessapGraphDataset(dataset_name='sample', split='test', root=save_path)
    # We can plot all the values for comparison
    plot_counter_as_hist(data_counter=val_dataset.statistics.num_nodes, x_label='Num nodes',
                         second_counter=test_dataset.statistics.num_nodes, name='num_nodes', y_label='frequency')
    plot_list_as_hist(data=val_dataset.statistics.atom_types, x_label='Atom types',
                      second_list=test_dataset.statistics.atom_types, name='node_types')
    plot_list_as_hist(data=val_dataset.statistics.bond_types, x_label='Edge type',
                      second_list=val_dataset.statistics.bond_types, name='edge_types')
    val_bond_lengths_dict_of_list = {key: list(c.values()) for key, c in val_dataset.statistics.bond_lengths.items()}
    test_bond_lengths_dict_of_list = {key: list(c.values()) for key, c in test_dataset.statistics.bond_lengths.items()}
    plot_list_of_dict_as_hist(data_dict=val_bond_lengths_dict_of_list, second_dict=test_bond_lengths_dict_of_list,
                              name='edge_lengths')
    # Doing the same for the bond angles
    val_bond_angles_dict_of_list = {key: val_dataset.statistics.bond_angles[key] for key in
                                    range(len(val_dataset.statistics.bond_angles))}
    test_bond_angles_dict_of_list = {key: test_dataset.statistics.bond_angles[key] for key in
                                     range(len(test_dataset.statistics.bond_angles))}
    plot_list_of_dict_as_hist(data_dict=val_bond_angles_dict_of_list, second_dict=test_bond_angles_dict_of_list,
                              name='edge_angles')


def get_edge_attr_counts():
    class dummy_obj(object):
        def __init__(self):
            super(dummy_obj, self).__init__()

    # Now we use the dummy object for random configuration
    cfg = dummy_obj()
    save_path = os.path.join(PROJECT_ROOT_DIR, 'data', 'vessap', 'converted_pt_files')
    dataset = VessapGraphDataset(dataset_name='sample', split='train', root=save_path, cfg=cfg)
    all_attr = []
    for idx in range(len(dataset)):
        edge_type = dataset[idx].edge_attr
        all_attr.append(edge_type)
    all_attr_tensor = torch.cat(all_attr)
    edge_type_sum = all_attr_tensor.sum(dim=0)
    print("Num Edge count distribution")
    print(edge_type_sum / edge_type_sum.sum())


if __name__ == '__main__':
    save_path = os.path.join(PROJECT_ROOT_DIR, 'data', 'vessap', 'converted_pt_files')
    dataset = VessapGraphDataset(dataset_name='sample', split='test', root=save_path)
    # plot_stats(dataset)
    # plot_stat_comp()
    check_deg_c_N()
    # compute_min_max_pos(split='train')
    # dataset = compute_min_max_pos(split='train')
    # from pyinstrument import Profiler
    #
    # with Profiler(interval=0.1) as profiler:
    #     save_path = os.path.join(PROJECT_ROOT_DIR, 'data', 'vessap', 'converted_pt_files')
    #     dataset = compute_min_max_pos(split='val')
    #     print(f"{len(dataset)=}")
    #     # dataset = VessapGraphDataset(dataset_name='sample', split='train', root=save_path)
    #     print(dataset[0])
    #     print(dataset[0].y)
    #     print(dataset[0].pos)
    #     print(torch.mean(dataset[0].pos, dim=0))
    #     exec_configs()
    # profiler.print()
