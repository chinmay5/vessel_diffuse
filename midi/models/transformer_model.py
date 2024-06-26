import math

import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

import midi.utils as utils
from midi.diffusion import diffusion_utils
from midi.models.layers import Xtoy, Etoy, masked_softmax, EtoX, SE3Norm


def pairwise_angles(coords):
    """
    Compute pairwise spherical angles (azimuthal and polar angles) between each pair of 3D coordinates.

    Args:
    - coords (torch.Tensor): Tensor of shape (B, N, 3) containing batched 3D coordinates

    Returns:
    - azimuthal_angles (torch.Tensor): Pairwise azimuthal angles tensor of shape (B, N, N)
    - polar_angles (torch.Tensor): Pairwise polar angles tensor of shape (B, N, N)
    """
    B, N, _ = coords.shape
    x, y, z = coords[:, :, 0], coords[:, :, 1], coords[:, :, 2]

    # Compute azimuthal angles
    x_diff = x[:, :, None] - x[:, None, :]
    y_diff = y[:, :, None] - y[:, None, :]
    azimuthal_angles = torch.atan2(y_diff, x_diff)
    azimuthal_angles = (azimuthal_angles + 2 * torch.pi) % (2 * torch.pi)  # Adjust range to [0, 2*pi]

    # Compute polar angles
    r_xy = torch.sqrt(x_diff ** 2 + y_diff ** 2)

    # Handle the case where all coordinates are zero
    polar_angles = torch.where(r_xy == 0, torch.zeros_like(r_xy), torch.atan2(z[:, :, None] - z[:, None, :], r_xy))

    return azimuthal_angles, polar_angles


def compute_pair_coordinates(points):
    batch_size, num_points, _ = points.shape

    # Reshape points tensor to enable broadcasting
    reshaped_points = points.unsqueeze(2)  # Shape: batch_size x N x 1 x 3

    # Repeat points tensor along new axis to create pairs
    repeated_points = reshaped_points.repeat(1, 1, num_points, 1)  # Shape: batch_size x N x N x 3

    # Transpose repeated_points to align pairs
    transposed_points = repeated_points.transpose(1, 2)  # Shape: batch_size x N x N x 3

    # Concatenate the two tensors along the last dimension
    pair_coordinates = torch.cat((repeated_points, transposed_points), dim=3)  # Shape: batch_size x N x N x 6

    return pair_coordinates


class NodeEdgeBlock(nn.Module):
    """ Self attention layer that also updates the representations on the edges. """

    def __init__(self, dx, de, dy, n_head, last_layer=False):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        self.in_E = Linear(de, de)

        # FiLM X to E
        # self.x_e_add = Linear(dx, de)
        self.x_e_mul1 = Linear(dx, de)
        self.x_e_mul2 = Linear(dx, de)

        # Distance encoding
        self.lin_dist1 = Linear(2, de)
        self.lin_dist_coords = Linear(6, de)

        self.lin_norm_pos1 = Linear(dx, de)
        self.lin_norm_pos2 = Linear(dx, de)

        # Angle encoding
        self.ang_info = Linear(2, de)

        self.dist_add_e = Linear(de, de)
        self.dist_mul_e = Linear(de, de)

        # Attention

        self.k = Linear(dx, dx)
        self.q = Linear(dx, dx)
        self.v = Linear(dx, dx)
        self.a = Linear(dx, n_head, bias=False)
        self.out = Linear(dx * n_head, dx)

        # Incorporate e to x
        # self.e_att_add = Linear(de, n_head)
        self.e_att_mul = Linear(de, n_head)

        self.pos_att_mul = Linear(de, n_head)

        self.e_x_mul = EtoX(de, dx)

        self.pos_x_mul = EtoX(de, dx)

        # FiLM y to E
        self.y_e_mul = Linear(dy, de)  # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, de)

        self.pre_softmax = Linear(de, dx)  # Unused, but needed to load old checkpoints

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        self.last_layer = last_layer
        if not last_layer:
            self.y_y = Linear(dy, dy)
            self.x_y = Xtoy(dx, dy)
            self.e_y = Etoy(de, dy)
            self.dist_y = Etoy(de, dy)

        # Process_pos
        self.e_pos1 = Linear(de, de, bias=False)
        self.e_pos2 = Linear(de, 1, bias=False)  # For EGNN v3: map to pi, pj

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(de, de)
        if not last_layer:
            self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, pos, node_mask):
        """ :param X: bs, n, d        node features
            :param E: bs, n, n, d     edge features
            :param y: bs, dz           global features
            :param pos: bs, n, 3
            :param node_mask: bs, n
            :return: newX, newE, new_y with the same shape. """
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1

        # 0. Create a distance matrix that can be used later
        pos = pos * x_mask
        # norm_pos = torch.norm(pos, dim=-1, keepdim=True)  # bs, n, 1
        # normalized_pos = pos / (norm_pos + 1e-7)  # bs, n, 3
        pair_coords = compute_pair_coordinates(points=pos)  # bs, n, n, 6
        dist1 = F.relu(self.lin_dist_coords(pair_coords)) * e_mask1 * e_mask2

        # 1. Process E
        # Remember that Y is the edge features and not the "labels".
        # Often times the symbol Y is used for the latter. Not here though.
        Y = self.in_E(E)

        # 1.1 Incorporate x
        x_e_mul1 = self.x_e_mul1(X) * x_mask
        x_e_mul2 = self.x_e_mul2(X) * x_mask
        Y = Y * x_e_mul1.unsqueeze(1) * x_e_mul2.unsqueeze(2) * e_mask1 * e_mask2

        # 1.2. Incorporate distances
        dist_add = self.dist_add_e(dist1)
        dist_mul = self.dist_mul_e(dist1)
        Y = (Y + dist_add + Y * dist_mul) * e_mask1 * e_mask2  # bs, n, n, dx

        # 1.3 Incorporate y to E
        y_e_add = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        y_e_mul = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        E = (Y + y_e_add + Y * y_e_mul) * e_mask1 * e_mask2

        # Output E
        Eout = self.e_out(E) * e_mask1 * e_mask2  # bs, n, n, de

        diffusion_utils.assert_correctly_masked(Eout, e_mask1 * e_mask2)

        Q = (self.q(X) * x_mask).unsqueeze(2)  # bs, n, 1, dx
        K = (self.k(X) * x_mask).unsqueeze(1)  # bs, 1, n, dx
        prod = Q * K / math.sqrt(Y.size(-1))  # bs, n, n, dx
        a = self.a(prod) * e_mask1 * e_mask2  # bs, n, n, n_head

        # 2.1 Incorporate edge features
        e_x_mul = self.e_att_mul(E)
        a = a + e_x_mul * a

        # 2.2 Incorporate position features
        pos_x_mul = self.pos_att_mul(dist1)
        a = a + pos_x_mul * a
        a = a * e_mask1 * e_mask2

        # 2.3 Self-attention
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)
        alpha = masked_softmax(a, softmax_mask, dim=2).unsqueeze(-1)  # bs, n, n, n_head, 1
        V = (self.v(X) * x_mask).unsqueeze(1).unsqueeze(3)  # bs, 1, n, 1, dx
        weighted_V = alpha * V  # bs, n, n, n_heads, dx
        weighted_V = weighted_V.sum(dim=2)  # bs, n, n_head, dx
        weighted_V = weighted_V.flatten(start_dim=2)  # bs, n, n_head x dx
        weighted_V = self.out(weighted_V) * x_mask  # bs, n, dx

        # Incorporate E to X
        e_x_mul = self.e_x_mul(E, e_mask2)
        weighted_V = weighted_V + e_x_mul * weighted_V

        pos_x_mul = self.pos_x_mul(dist1, e_mask2)
        weighted_V = weighted_V + pos_x_mul * weighted_V

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)  # bs, 1, dx
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = weighted_V * (yx2 + 1) + yx1

        # Output X
        Xout = self.x_out(newX) * x_mask
        diffusion_utils.assert_correctly_masked(Xout, x_mask)

        # Process y based on X and E
        if self.last_layer:
            y_out = None
        else:
            y = self.y_y(y)
            e_y = self.e_y(Y, e_mask1, e_mask2)
            x_y = self.x_y(newX, x_mask)
            dist_y = self.dist_y(dist1, e_mask1, e_mask2)
            new_y = y + x_y + e_y + dist_y
            y_out = self.y_out(new_y)  # bs, dy

        # Update the positions
        pos1 = pos.unsqueeze(1).expand(-1, n, -1, -1)  # bs, 1, n, 3
        pos2 = pos.unsqueeze(2).expand(-1, -1, n, -1)  # bs, n, 1, 3
        delta_pos = pos2 - pos1  # bs, n, n, 3

        messages = self.e_pos2(F.relu(self.e_pos1(Y)))  # bs, n, n, 1, 2
        vel = (messages * delta_pos).sum(dim=2) * x_mask
        vel = utils.remove_mean_with_mask(vel, node_mask)

        return Xout, Eout, y_out, vel


class XEyTransformerLayer(nn.Module):
    """ Transformer that updates node, edge and global features
        d_x: node features
        d_e: edge features
        dz : global features
        n_head: the number of heads in the multi_head_attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probablility. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """

    def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffX: int = 2048,
                 dim_ffE: int = 128, dim_ffy: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None, last_layer=False,
                 is_directed=False, update_node_loc=True) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()
        print(f"{update_node_loc=}")

        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, last_layer=last_layer)
        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        # self.normX1 = SetNorm(feature_dim=dx, eps=layer_norm_eps, **kw)
        # self.normX2 = SetNorm(feature_dim=dx, eps=layer_norm_eps, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        # self.normE1 = GraphNorm(feature_dim=de, eps=layer_norm_eps, **kw)
        # self.normE2 = GraphNorm(feature_dim=de, eps=layer_norm_eps, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.norm_pos1 = SE3Norm(eps=layer_norm_eps, **kw)
        # Let us project the coordinates to a high dim space.
        # Maybe that is also useful
        self.pos_proj = nn.Sequential(Linear(3, 3), nn.ReLU(), Linear(3, 3))

        self.last_layer = last_layer
        if not last_layer:
            self.lin_y1 = Linear(dy, dim_ffy, **kw)
            self.lin_y2 = Linear(dim_ffy, dy, **kw)
            self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
            self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
            self.dropout_y1 = Dropout(dropout)
            self.dropout_y2 = Dropout(dropout)
            self.dropout_y3 = Dropout(dropout)

        self.activation = F.relu
        self.is_directed = is_directed
        self.update_node_loc = update_node_loc

    def forward(self, features: utils.PlaceHolder):
        """ Pass the input through the encoder layer.
            X: (bs, n, d)
            E: (bs, n, n, d)
            y: (bs, dy)
            node_mask: (bs, n) Mask for the src keys per batch (optional)
            Output: newX, newE, new_y with the same shape.
        """
        X = features.X
        E = features.E
        y = features.y
        pos = features.pos
        node_mask = features.node_mask
        # x_mask = node_mask.unsqueeze(-1)  # bs, n, 1

        # pos should never change
        newX, newE, new_y, _ = self.self_attn(X, E, y, pos, node_mask=node_mask)

        new_pos = pos
        newX_d = self.dropoutX1(newX)
        # X = self.normX1(X + newX_d, x_mask)
        X = self.normX1(X + newX_d)

        # new_pos = self.norm_pos1(vel, x_mask) + pos
        # new_pos = self.pos_proj(new_pos)
        #
        # if torch.isnan(new_pos).any():
        #     raise ValueError("NaN in new_pos")

        newE_d = self.dropoutE1(newE)
        # E = self.normE1(E + newE_d, e_mask1, e_mask2)
        E = self.normE1(E + newE_d)

        if not self.last_layer:
            new_y_d = self.dropout_y1(new_y)
            y = self.norm_y1(y + new_y_d)

        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        # X = self.normX2(X + ff_outputX, x_mask)
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        # E = self.normE2(E + ff_outputE, e_mask1, e_mask2)
        E = self.normE2(E + ff_outputE)
        if not self.is_directed:
            E = 0.5 * (E + torch.transpose(E, 1, 2))

        if not self.last_layer:
            ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
            ff_output_y = self.dropout_y3(ff_output_y)
            y = self.norm_y2(y + ff_output_y)

        out = utils.PlaceHolder(X=X, E=E, y=y, pos=new_pos, charges=None, node_mask=node_mask,
                                directed=self.is_directed).mask()

        return out


class FFEncoder(torch.nn.Module):
    """Encodes positions with random fourier features."""
    pos_bases: torch.Tensor

    def __init__(self, latent_dim: int, spatial_dim: int, pos_enc_sigma: float):
        super().__init__()
        if pos_enc_sigma <= 0:
            raise ValueError(f'Stdev for pos. enc. must be positive, got {pos_enc_sigma}.')
        if latent_dim % 2 != 0:
            raise ValueError(f'Latent dimension must be divisible by 2, instead got {latent_dim}.')
        # Register a buffer -- these vectors are pushed to GPU and stored with the model, (but not
        # optimized).
        self.register_buffer('pos_bases',
                             torch.normal(0.0, pos_enc_sigma, [spatial_dim, int(latent_dim / 2)]))

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        angles = 2 * np.pi * torch.matmul(positions, self.pos_bases)
        return torch.cat([torch.cos(angles), torch.sin(angles)], -1)


class GraphTransformer(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """

    def __init__(self, input_dims: utils.PlaceHolder, n_layers: int, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: utils.PlaceHolder, is_directed: bool = False, cfg=None):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims.X
        self.out_dim_E = output_dims.E
        self.out_dim_y = output_dims.y
        self.out_dim_charges = output_dims.charges

        act_fn_in = nn.ReLU()
        act_fn_out = nn.ReLU()
        # self.ff_encoder = FFEncoder(latent_dim=hidden_dims['dx'], spatial_dim=3, pos_enc_sigma=1)
        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims.X + input_dims.charges, hidden_mlp_dims['X']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in)
        self.mlp_in_E = nn.Sequential(nn.Linear(input_dims.E, hidden_mlp_dims['E']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in)

        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims.y, hidden_mlp_dims['y']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), act_fn_in)
        self.is_directed = is_directed

        self.tf_layers = nn.ModuleList([XEyTransformerLayer(dx=hidden_dims['dx'],
                                                            de=hidden_dims['de'],
                                                            dy=hidden_dims['dy'],
                                                            n_head=hidden_dims['n_head'],
                                                            dim_ffX=hidden_dims['dim_ffX'],
                                                            dim_ffE=hidden_dims['dim_ffE'],
                                                            is_directed=self.is_directed,
                                                            update_node_loc=False,
                                                            last_layer=False,
                                                            )
                                        # needed to load old checkpoints
                                        # last_layer=(i == n_layers - 1))
                                        for i in range(n_layers)])

        self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['X'], output_dims.X + output_dims.charges))
        self.mlp_out_E = nn.Sequential(nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['E'], output_dims.E))
        self.mlp_in_pos_high_dim = nn.Sequential(nn.Linear(3, hidden_mlp_dims['X']), act_fn_in,
                                                 nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in)
        # We do not train for y. The variable is used to provide global feature such as the time
        # and few other graph level feature terms. However, we will not update it and compute CE for it.

    def forward(self, data: utils.PlaceHolder):
        bs, n = data.X.shape[0], data.X.shape[1]
        node_mask = data.node_mask

        diag_mask = ~torch.eye(n, device=data.X.device, dtype=torch.bool)
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)
        X = torch.cat((data.X, data.charges), dim=-1)

        X_to_out = X[..., :self.out_dim_X + self.out_dim_charges]
        E_to_out = data.E[..., :self.out_dim_E]
        # We are copying over the values of y.
        # Hence, the global condition is not changed.
        # We do not want to update it.
        y_to_out = data.y[..., :self.out_dim_y]
        new_E = self.mlp_in_E(data.E)
        if not self.is_directed:
            new_E = (new_E + new_E.transpose(1, 2)) / 2
        # if no_node_denoising:
        # We are only denoising the edge connectivity
        pos = data.pos.detach()
        # We look for high dimensional projections of the node positions
        # We also get the FF features so that we can give the positional sense to the nodes
        features = utils.PlaceHolder(X=self.mlp_in_pos_high_dim(pos), E=new_E, y=self.mlp_in_y(data.y), charges=None,
                                     # Passing pos derived features separately
                                     pos=pos, node_mask=node_mask,
                                     directed=self.is_directed).mask()

        for layer in self.tf_layers:
            features = layer(features)
            # Trying skip connection
            # features = layer(features) + features
        # We make the X predictions very simple.
        # We took the input features, and just passed it through a few MLP blocks
        X = self.mlp_out_X(self.mlp_in_X(X))
        E = self.mlp_out_E(features.E)
        # This line was commented in base version
        # y = self.mlp_out_y(features.y)
        # pos = self.mlp_out_pos(features.pos, node_mask)
        # pos = features.pos
        # skip connections
        X = (X + X_to_out)
        E = (E + E_to_out) * diag_mask

        # Copy over the values of y since we do not want to update it.
        # It is a global condition that should not change.
        y = y_to_out
        if not self.is_directed:
            E = 1 / 2 * (E + torch.transpose(E, 1, 2))

        final_X = X[..., :self.out_dim_X]
        charges = X[..., self.out_dim_X:]
        # We do not touch the node coordinates at all
        out = utils.PlaceHolder(pos=data.pos, X=final_X, charges=charges, E=E, y=y, node_mask=node_mask,
                                directed=self.is_directed, skip_mean=True).mask()
        return out
