import imageio
import os

import torch
import wandb
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def visualize_non_molecule(nodes, edges, pos, path, num_node_types):
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('equal', adjustable='datalim')
    ax.view_init(elev=90, azim=-90)
    # bg color is white
    ax.set_facecolor((1, 1, 1))
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax._axis3don = False

    ax.w_xaxis.line.set_color("white")

    # max_value = positions.abs().max().item()
    axis_lim = 0.7
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_zlim(-axis_lim, axis_lim)
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]

    # Normalize the positions for plotting
    max_x_dist = x.max() - x.min()
    max_y_dist = y.max() - y.min()
    max_z_dist = z.max() - z.min()
    max_dist = max(max_x_dist, max_y_dist, max_z_dist) / 1.8
    x_center = (x.min() + x.max()) / 2
    y_center = (y.min() + y.max()) / 2
    z_center = (z.min() + z.max()) / 2
    x = (x - x_center) / max_dist
    y = (y - y_center) / max_dist
    z = (z - z_center) / max_dist

    radii = 0.4
    areas = 300 * (radii ** 2)
    if isinstance(nodes, torch.Tensor):
        nodes = nodes.cpu().numpy().tolist()
    colormap = [f'C{a}' for a in range(num_node_types)]
    colors = [colormap[a] for a in nodes]
    for i in range(edges.shape[0]):
        for j in range(i + 1, edges.shape[1]):
            draw_edge = edges[i, j]
            if draw_edge > 0:
                ax.plot([x[i], x[j]],
                        [y[i], y[j]],
                        [z[i], z[j]],
                        linewidth=1, c='#000000', alpha=1)

    ax.scatter(x, y, z, s=areas, alpha=0.9, c=colors)
    plt.tight_layout()
    plt.savefig(path, format='png', pad_inches=0.0)
    plt.close()


def visualize(path: str, graph_list: list, num_graphs_to_visualize: int, log='graph'):
    if num_graphs_to_visualize == -1:
        num_graphs_to_visualize = len(graph_list)
    # define path to save figures
    if not os.path.exists(path):
        os.makedirs(path)
    all_file_paths = []
    # visualize the final molecules
    for i in range(num_graphs_to_visualize):
        file_path = os.path.join(path, 'graph_{}.png'.format(i))
        nodes, edges, pos, num_node_types = graph_list[i]
        if isinstance(pos, torch.Tensor):
            pos = pos.cpu().numpy()
        visualize_non_molecule(nodes=nodes, edges=edges, pos=pos, path=file_path, num_node_types=num_node_types)
        # im = plt.imread(file_path)
        all_file_paths.append(file_path)
        # if wandb.run and log is not None:
        #     wandb.log({log: [wandb.Image(im, caption=file_path)]})
    return all_file_paths


def visualize_chains(path, chain, num_nodes, num_node_types):
    pass
    pca = PCA(n_components=3)
    for i in range(chain.X.size(1)):  # Iterate over the molecules
        print(f'Visualizing chain {i}/{chain.X.size(1)}')
        result_path = os.path.join(path, f'chain_{i}')

        chain_nodes = chain.X[:, i][:, :num_nodes[i]].long()
        chain_edges = chain.E[:, i][:, :num_nodes[i], :][:, :, :num_nodes[i]].long()
        chain_positions = chain.pos[:, i, :][:, :num_nodes[i]]

        # Transform the positions using PCA to align best to the final molecule
        if chain_positions[-1].shape[0] > 2:
            pca.fit(chain_positions[-1])
        graphs = []
        for j in range(chain_nodes.shape[0]):
            pos = pca.transform(chain_positions[j]) if chain_positions[-1].shape[0] > 2 else chain_positions[j].numpy()
            graphs.append((chain_nodes[j], chain_edges[j], pos, num_node_types))
        print("Graph list generated.")

        for frame in range(len(graphs)):
            all_file_paths = visualize(result_path, graphs, num_graphs_to_visualize=-1)

    #     # Turn the frames into a gif
    #     imgs = [imageio.v3.imread(fn) for fn in all_file_paths]
    #     gif_path = os.path.join(os.path.dirname(path), f"{path.split('/')[-1]}_{i}.gif")
    #     print(f'Saving the gif at {gif_path}.')
    #     imgs.extend([imgs[-1]] * 10)
    #     imageio.mimsave(gif_path, imgs, subrectangles=True, duration=200)
    #
    #     if wandb.run:
    #         wandb.log({"chain": wandb.Video(gif_path, fps=5, format="gif")}, commit=True)
            # trainer.logger.experiment.log({'chain': [wandb.Video(gif_path, caption=gif_path, format="gif")]})
        # print("Non molecular Chain saved.")
