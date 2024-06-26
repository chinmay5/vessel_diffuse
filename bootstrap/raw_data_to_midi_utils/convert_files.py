import os
import sys
import warnings

import numpy as np
import pyvista
import torch
import torch.nn.functional as F
import torch_geometric.utils
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, contains_isolated_nodes, contains_self_loops
from torch_scatter import scatter_mean

warnings.filterwarnings("ignore")


def _save_vtp_file(nodes, edges, radius, filename, save_dir):
    mesh_edge = np.concatenate((np.int32(2 * np.ones((1, edges.shape[0]))), edges.T), 0)
    mesh = pyvista.UnstructuredGrid(mesh_edge.T, np.array([4] * len(radius)), nodes)
    mesh.cell_data['radius'] = radius
    mesh_structured = mesh.extract_surface()
    mesh_structured.save(os.path.join(save_dir, filename))


def read_vtp_file_as_pyg_data(filename, buffer, make_undirected, add_node_diameter, include_cont_radii):
    vtk_data = pyvista.read(filename)
    nodes = torch.tensor(np.float32(np.asarray(vtk_data.points)), dtype=torch.float)
    edges = torch.tensor(np.asarray(vtk_data.lines.reshape(-1, 3)), dtype=torch.int64)[:, 1:]
    # bugfix. Earlier I was loading the value as a long tensor and thus, it was already truncated.
    # Hence, the ceil function essentially did nothing to it.
    radius = torch.tensor(np.asarray(vtk_data.cell_data['radius']), dtype=torch.float)
    # Define the interval for discretization
    edge_attr_discrete = assign_edge_type(radius)
    edge_attr = edge_attr_discrete
    if include_cont_radii:
        edge_attr = torch.stack((edge_attr, radius), dim=1)
    # Now we will make the graph undirected
    edges = edges.T
    graph_data = Data(x=nodes, edge_index=edges, edge_attr=edge_attr)
    if contains_isolated_nodes(graph_data.edge_index, num_nodes=graph_data.size(0)):
        buffer.write(f"contains isolated nodes\n")
        # Finally, we remove the self loops, in case there are any.
    if contains_self_loops(graph_data.edge_index):
        buffer.write(f"contains self loops. Removing them\n")
        graph_data.edge_index, graph_data.edge_attr = remove_self_loops(graph_data.edge_index, graph_data.edge_attr)

    if make_undirected:
        # Making the graph undirected
        new_edges, new_edge_attr = torch_geometric.utils.to_undirected(edge_index=graph_data.edge_index,
                                                                       edge_attr=graph_data.edge_attr,
                                                                       num_nodes=graph_data.x.size(0))
    else:
        new_edges, new_edge_attr = graph_data.edge_index, graph_data.edge_attr
    # All the processing wrt edges is now done.
    # We would check if we have added continuous radii to them, and if so, save it separately.
    cont_edge_radii = None
    if include_cont_radii:
        new_edge_attr, cont_edge_radii = new_edge_attr[:, 0].to(torch.long), new_edge_attr[:, 1]
    updated_graph = Data(x=nodes, edge_index=new_edges, edge_attr=new_edge_attr, cont_radii=cont_edge_radii)
    degree = torch_geometric.utils.degree(index=updated_graph.edge_index[1],
                                          num_nodes=updated_graph.size(0), dtype=torch.long)
    if add_node_diameter:
        incoming_sum = scatter_mean(updated_graph.edge_attr, updated_graph.edge_index[0], dim=0)
        diameter = F.one_hot(incoming_sum, num_classes=10).to(torch.float)
        updated_graph.avg_diameter = diameter
    else:
        # We just add a dummy variable here for downstream task
        updated_graph.avg_diameter = torch.ones((updated_graph.x.size(0), 1), dtype=torch.float)
    buffer.write(f"{degree=}\n")
    if torch.any(new_edge_attr < 1):
        print(f"Something wrong with edge_attr for {filename}")
        sys.exit(0)
    return updated_graph, degree


def assign_edge_type(radius):
    """
    Assigns the edge type. This is for the vessap dataset.
    The Crown and TopCoW datasets already have this information available from the segmentation.
    :param radius:
    :return:
    """
    # Based on the vessap paper
    discrete_radius = radius.clone()
    discrete_radius[discrete_radius <= 1.5] = 1
    discrete_radius[(1.5 < discrete_radius) & (discrete_radius <= 4)] = 2
    discrete_radius[discrete_radius > 4] = 3
    return discrete_radius.to(torch.long)


def remove_self_loops_and_save_as_pt(filename, idx,
                                     buffer, save_path, make_undirected, add_node_diameter, include_cont_radii):
    buffer.write(f"Processing {filename=}\n")
    graph_data, degree = read_vtp_file_as_pyg_data(filename=filename,
                                                   buffer=buffer, make_undirected=make_undirected,
                                                   add_node_diameter=add_node_diameter,
                                                   include_cont_radii=include_cont_radii)
    num_nodes = graph_data.x.size(0)

    pos = graph_data.x[:, :3]
    pos = pos - torch.mean(pos, dim=0, keepdim=True)
    # Adding normalized node coordinates
    graph_data.pos = pos
    # Dummy node attributes
    graph_data.x = torch.ones(num_nodes, 1, dtype=torch.float)
    save_dir = os.path.join(save_path, "..", "vtk_sample_viz")
    filename = os.path.basename(filename)
    if idx <= 10:
        os.makedirs(save_dir, exist_ok=True)
        radius = graph_data.edge_attr.numpy()
        _save_vtp_file(graph_data.pos.numpy(), graph_data.edge_index.T.numpy(), radius,
                       f"{filename.replace('.vtp', '_checked.vtp')}", save_dir)
        print(f"edge index = {graph_data.edge_index.T} and degree = {degree}")
    filename_pt = filename.replace(".vtp", ".pt")
    torch.save(graph_data, os.path.join(save_path, filename_pt))
