import os

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error
from timm import optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm

from environment_setup import PROJECT_ROOT_DIR
from midi.datasets.cow_dataset import CoWDataset
from midi.datasets.vessap_dataset import VessapGraphDataset


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, input_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = input_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_pos(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, pos, node_mask):
        model.eval()
        with torch.no_grad():
            x = torch.randn((pos.shape[0], pos.shape[1], pos.shape[2])).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0, total=self.noise_steps):
                t = (torch.ones(pos.shape[0]) * i).long().to(self.device)
                predicted_noise = model(x, t, node_mask)
                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (
                        x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
        model.train()
        return x


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, batch_first=True, num_heads=num_heads)
        self.ln = nn.LayerNorm([hidden_dim])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([hidden_dim]),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, key_padding_mask):
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln, key_padding_mask=key_padding_mask)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value


class Simple_FC(nn.Module):
    def __init__(self, hidden_dim, n_layers=2, eps=1e-5):
        super(Simple_FC, self).__init__()
        self.eps = eps
        self.hidden_dim = hidden_dim
        spatial_dim = 3
        self.mlp_in = nn.Linear(spatial_dim, hidden_dim)
        mlp_mid, sa, t_emb = [], [], []
        for _ in range(n_layers):
            mlp_mid.append(nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
            sa.append(SelfAttention(hidden_dim=hidden_dim, num_heads=4))
            t_emb.append(nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.mlp_out = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, spatial_dim))
        self.mlp_mid = nn.ModuleList(mlp_mid)
        self.sa = nn.ModuleList(sa)
        self.t_emb = nn.ModuleList(t_emb)

    def pos_encoding(self, t, proj_dims):
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, proj_dims, 2, device=t.device).float() / proj_dims)
        )
        pos_enc_a = torch.sin(t.repeat(1, proj_dims // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, proj_dims // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, pos, t, node_mask):
        """
        Conditioning made on the latens and their original position
        :param pos: 3d coordinate position
        :param t: timestep
        :param node_mask: mask applied on the node
        :return: noise/velocity prediction
        """
        t = self.pos_encoding(t.unsqueeze(1), self.hidden_dim)
        pos = self.mlp_in(pos)
        for sa, mlp, t_emb in zip(self.sa, self.mlp_mid, self.t_emb):
            pos = mlp(pos) + t_emb(t[:, None])
            pos = sa(pos, key_padding_mask=~node_mask)
        pos = self.mlp_out(pos)
        return pos


class dummy(object):
    def __init__(self):
        super(dummy, self).__init__()


def perform_coord_wise_kde(coords_synth, coords_real, epoch):
    colors = ['g', 'r']
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    labels = ['synth', 'real']
    for idx, coords in enumerate([coords_synth, coords_real]):
        x_coords, y_coords, z_coords = coords[:, 0], coords[:, 1], coords[:, 2]
        x_coord_kde = gaussian_kde(x_coords.tolist())
        y_coord_kde = gaussian_kde(y_coords.tolist())
        z_coord_kde = gaussian_kde(z_coords.tolist())
        pts = np.linspace(-1, 1, 2000)
        axes[0].plot(pts, x_coord_kde(pts), label=f"KDE_x_coords_{labels[idx]}", color=colors[idx], alpha=0.5 ** idx)
        axes[0].legend()
        axes[1].plot(pts, y_coord_kde(pts), label=f"KDE_y_coords_{labels[idx]}", color=colors[idx], alpha=0.5 ** idx)
        axes[1].legend()
        axes[2].plot(pts, z_coord_kde(pts), label=f"KDE_z_coords_{labels[idx]}", color=colors[idx], alpha=0.5 ** idx)
        axes[2].legend()
    plt.tight_layout()
    # Save the figure for usage
    os.makedirs('node_diffusion', exist_ok=True)
    plt.savefig(f'node_diffusion/line_graph_attr_{epoch}.png', dpi=80)
    plt.show()


def perform_icp(point_cloud1, point_cloud2):
    # Convert to Open3D point cloud format
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(point_cloud1)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(point_cloud2)

    # Perform ICP
    threshold = 0.02  # Distance threshold
    trans_init = np.eye(4)  # Initial transformation

    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd1, pcd2, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    # Apply transformation to align point clouds
    pcd2.transform(reg_p2p.transformation)

    # Compute the distance between aligned point clouds
    distances = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
    return np.mean(distances).item()


def perform_2d_mse(point_cloud1, point_cloud2):
    # Project onto x-y plane
    projection1 = point_cloud1[:, :2]
    projection2 = point_cloud2[:, :2]
    # Compute Mean Squared Error between projections
    return mean_squared_error(projection1, projection2)


def get_data(args):
    if args.dataset == 'vessap':
        save_path = os.path.join(PROJECT_ROOT_DIR, 'data', 'vessap', 'converted_pt_files')
        train_dataset = VessapGraphDataset(dataset_name='sample', split="train", root=save_path)
        val_dataset = VessapGraphDataset(dataset_name='sample', split="val", root=save_path)
    elif args.dataset == 'cow':
        # Doing this just for the compatibility
        cfg = dummy()
        cfg.dataset = {"is_multiclass": True,
                       "node_jitter": False}
        save_path = os.path.join(PROJECT_ROOT_DIR, 'data', 'cow_multi', 'pt_files')
        train_dataset = CoWDataset(dataset_name='sample', split='train', root=save_path, cfg=cfg)
        val_dataset = CoWDataset(dataset_name='sample', split="val", root=save_path, cfg=cfg)
        # Dataloaders using torch geometric
    else:
        raise AttributeError(f'Invalid dataset name: {args.dataset}')
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=4, shuffle=False)
    return train_loader, val_loader


def pairwise_distance(input_tensor):
    # Squared L2 distance between each pair of vectors
    # input_tensor.shape = (B, N, 3)
    squared_distance = torch.sum((input_tensor.unsqueeze(2) - input_tensor.unsqueeze(1)) ** 2, dim=-1)
    return squared_distance


def distance_kl_loss(pos, x_t, predicted_noise, t, alpha_hat, node_mask):
    alpha_t_bar = alpha_hat[t][:, None, None]
    x_0_hat = (1 / torch.sqrt(alpha_t_bar)) * (x_t - predicted_noise * torch.sqrt(1 - alpha_t_bar))
    dist_gt = pairwise_distance(pos) * node_mask.unsqueeze(-1) * node_mask.unsqueeze(1)
    dist_pred = pairwise_distance(x_0_hat) * node_mask.unsqueeze(-1) * node_mask.unsqueeze(1)
    # Ignore the zero values
    dist_gt_dist, dist_pred_dist = torch.flatten(dist_gt[dist_gt > 0]), torch.flatten(dist_pred[dist_pred > 0])
    return (F.mse_loss(torch.mean(dist_gt_dist), torch.mean(dist_pred_dist), reduction="none") +
            F.mse_loss(torch.std(dist_gt_dist), torch.std(dist_pred_dist), reduction="none"))


def train(args):
    device = args.device
    train_dataloader, val_loader = get_data(args)
    diffusion = Diffusion(input_size=args.image_size, device=device, noise_steps=args.noise_steps)
    model = Simple_FC(n_layers=args.n_layers, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(train_dataloader)

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        epoch_loss = ctr = 0
        print(f"Starting epoch {epoch}:")
        pbar = tqdm(train_dataloader)
        for i, data in enumerate(pbar):
            data = data.to(device)
            pos, node_mask = to_dense_batch(x=data.pos, batch=data.batch)
            t = diffusion.sample_timesteps(pos.shape[0]).to(device)
            x_t, noise = diffusion.noise_pos(pos, t)
            predicted_noise = model(x_t, t, node_mask)
            mse_loss = mse(noise * node_mask.unsqueeze(-1), predicted_noise * node_mask.unsqueeze(-1))
            loss = mse_loss  # + 0.1 * kl_loss  # 0.0001
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", mse_loss.item(), global_step=epoch * l + i)
            # logger.add_scalar("kl_loss", kl_loss.item(), global_step=epoch * l + i)
            epoch_loss += loss.item()
            ctr += 1
        if epoch % 100 == 99:
            with torch.no_grad():
                sampled_positions, actual_positions = [], []
                model.eval()
                vl = val_ctr = 0
                for i, data in enumerate(val_loader):
                    data = data.to(device)
                    pos, node_mask = to_dense_batch(x=data.pos, batch=data.batch)
                    t = diffusion.sample_timesteps(pos.shape[0]).to(device)
                    x_t, noise = diffusion.noise_pos(pos, t)
                    predicted_noise = model(x_t, t, node_mask)
                    val_loss = mse(noise * node_mask.unsqueeze(-1), predicted_noise * node_mask.unsqueeze(-1))
                    vl += val_loss.item()
                    val_ctr += 1
                    # We sample a few examples to check the KDE plots.
                    sampled_pos = diffusion.sample(model, pos, node_mask)
                    sampled_pos = sampled_pos[node_mask]
                    sampled_positions.append(sampled_pos.view(-1, 3))
                    true_pos = pos[node_mask]
                    actual_positions.append(true_pos.view(-1, 3))

                point_cloud_pred = torch.cat(sampled_positions)
                point_cloud_gt = torch.cat(actual_positions)
                perform_coord_wise_kde(point_cloud_pred, point_cloud_gt, epoch=epoch)
                # We can perform some other tests as well.
                proj_loss = perform_2d_mse(point_cloud_pred.cpu().numpy(),
                                           point_cloud_gt.cpu().numpy())
                icp_loss = perform_icp(point_cloud_pred.cpu().numpy(),
                                       point_cloud_gt.cpu().numpy())
                print(f"ICP loss {icp_loss:.4f} and 2D mse loss {proj_loss:.4f}")
                if icp_loss < best_val_loss:
                    torch.save(model.state_dict(), os.path.join(PROJECT_ROOT_DIR, 'midi', 'models',
                                                                f'checks_{args.dataset}_best_coord_diff_model.pth'))
                    print(f"New best val model saved at {epoch=}")
                    best_val_loss = icp_loss
            # Putting model back to training
            model.train()
        total_loss = epoch_loss / ctr
        print(f"epoch {epoch} finished. Loss = {total_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(PROJECT_ROOT_DIR, 'midi', 'models',
                                                    f'checks_{args.dataset}_coord_diff_model.pth'))


def test(args):
    device = args.device
    _, val_loader = get_data(args)
    diffusion = Diffusion(input_size=args.image_size, device=device, noise_steps=args.noise_steps)
    model = Simple_FC(n_layers=args.n_layers, hidden_dim=args.hidden_dim)
    model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT_DIR, 'midi', 'models',
                                                  f'checks_{args.dataset}_best_coord_diff_model.pth')))
    model.to(device)
    sampled_positions, actual_positions = [], []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            data = data.to(device)
            pos, node_mask = to_dense_batch(x=data.pos, batch=data.batch)
            sampled_pos = diffusion.sample(model, pos, node_mask)
            sampled_pos = sampled_pos[node_mask]
            sampled_positions.append(sampled_pos.view(-1, 3))
            true_pos = pos[node_mask]
            actual_positions.append(true_pos.view(-1, 3))
        perform_coord_wise_kde(torch.cat(sampled_positions), torch.cat(actual_positions), epoch='best_val')


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 5000
    args.batch_size = 64
    args.hidden_dim = 256
    args.image_size = 3
    args.noise_steps = 1000
    args.device = "cuda"
    args.dataset = 'cow'
    args.lr = 3 * 1e-4
    args.weight_decay = 1e-4  # 0 for vessap
    args.n_layers = 4
    train(args)


if __name__ == '__main__':
    launch()
