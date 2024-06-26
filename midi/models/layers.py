import torch
import torch.nn as nn
from torch.nn import init

from midi import utils


class PositionsMLP(nn.Module):
    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mlp = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, pos, node_mask):
        norm = torch.norm(pos, dim=-1, keepdim=True)  # bs, n, 1
        new_norm = self.mlp(norm)  # bs, n, 1
        new_pos = pos * new_norm / (norm + self.eps)
        new_pos = new_pos * node_mask.unsqueeze(-1)
        new_pos = new_pos - torch.mean(new_pos, dim=1, keepdim=True)
        return new_pos


class PositionsMLP2(nn.Module):
    """
    The MLP variation which is not rotation equivariant.
    Hence, it will be useful in cases wherein isotropic nature is not needed.
    """

    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mlp = nn.Sequential(nn.Linear(3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))

    def forward(self, pos, node_mask):
        new_pos = self.mlp(pos) * node_mask.unsqueeze(-1)
        # new_pos = new_pos * node_mask.unsqueeze(-1)
        # NOTE: Is zero centring needed now?
        # new_pos = new_pos - torch.mean(new_pos, dim=1, keepdim=True)
        return new_pos


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

    def forward(self, x):
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value


class SimpleFC(nn.Module):
    def __init__(self, hidden_dim=2, eps=1e-5):
        super(SimpleFC, self).__init__()
        self.eps = eps
        self.mlp_in = nn.Sequential(nn.Linear(3, hidden_dim))
        self.sa1 = SelfAttention(hidden_dim=hidden_dim, num_heads=4)
        self.mlp_mid = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.sa2 = SelfAttention(hidden_dim=hidden_dim, num_heads=4)
        self.mlp_out = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, 3))
        self.hidden_dim = hidden_dim
        self.emb_layer1 = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.emb_layer2 = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    def pos_encoding(self, t, time_dim):
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, time_dim, 2, device=t.device).float() / time_dim)
        )
        pos_enc_a = torch.sin(t.repeat(1, time_dim // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, time_dim // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, pos, t_int, node_mask):
        t_info = self.pos_encoding(t_int, self.hidden_dim)
        new_pos = self.mlp_in(pos) + self.emb_layer1(t_info.unsqueeze(1))  # add node-dimension to time
        new_pos = self.sa1(new_pos)
        new_pos = self.mlp_mid(new_pos) + self.emb_layer2(t_info.unsqueeze(1))
        new_pos = self.sa2(new_pos)
        new_pos = self.mlp_out(new_pos)
        # Finally, checking the node mask term
        new_pos = new_pos * node_mask.unsqueeze(-1)
        return new_pos


class SE3Norm(nn.Module):
    def __init__(self, eps: float = 1e-5, device=None, dtype=None) -> None:
        """ Note: There is a relatively similar layer implemented by NVIDIA:
            https://catalog.ngc.nvidia.com/orgs/nvidia/resources/se3transformer_for_pytorch.
            It computes a ReLU on a mean-zero normalized norm, which I find surprising.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.normalized_shape = (1,)  # type: ignore[arg-type]
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.weight)

    def forward(self, pos, node_mask):
        norm = torch.norm(pos, dim=-1, keepdim=True)  # bs, n, 1
        mean_norm = torch.sum(norm, dim=1, keepdim=True) / torch.sum(node_mask, dim=1, keepdim=True)  # bs, 1, 1
        new_pos = self.weight * pos / (mean_norm + self.eps)
        return new_pos

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}'.format(**self.__dict__)


class CustomBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None,
                 dtype=None):
        super(CustomBatchNorm1d, self).__init__(num_features=num_features, eps=eps, momentum=momentum,
                                                affine=affine, track_running_stats=track_running_stats,
                                                device=device, dtype=dtype)

    def forward(self, pos, node_mask):
        new_pos = torch.zeros_like(pos)
        dummy_vals = pos * ~node_mask
        valid_pos = pos * node_mask
        b, n, d = valid_pos.shape
        valid_pos = valid_pos.view(b * n, valid_pos.shape[-1])
        valid_pos = super().forward(valid_pos)
        valid_pos = valid_pos.view(b, n, d)
        # Now we get back the original values
        new_pos[node_mask] = valid_pos
        new_pos[~node_mask] = dummy_vals
        return pos


class Xtoy(nn.Module):
    def __init__(self, dx, dy):
        """ Map node features to global features """
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X, x_mask):
        """ X: bs, n, dx. """
        x_mask = x_mask.expand(-1, -1, X.shape[-1])
        float_imask = 1 - x_mask.float()
        m = X.sum(dim=1) / torch.sum(x_mask, dim=1)
        mi = (X + 1e5 * float_imask).min(dim=1)[0]
        ma = (X - 1e5 * float_imask).max(dim=1)[0]
        std = torch.sum(((X - m[:, None, :]) ** 2) * x_mask, dim=1) / torch.sum(x_mask, dim=1)
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


class Etoy(nn.Module):
    def __init__(self, d, dy):
        """ Map edge features to global features. """
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E, e_mask1, e_mask2):
        """ E: bs, n, n, de
            Features relative to the diagonal of E could potentially be added.
        """
        mask = (e_mask1 * e_mask2).expand(-1, -1, -1, E.shape[-1])
        float_imask = 1 - mask.float()
        divide = torch.sum(mask, dim=(1, 2))
        m = E.sum(dim=(1, 2)) / divide
        mi = (E + 1e5 * float_imask).min(dim=2)[0].min(dim=1)[0]
        ma = (E - 1e5 * float_imask).max(dim=2)[0].max(dim=1)[0]
        std = torch.sum(((E - m[:, None, None, :]) ** 2) * mask, dim=(1, 2)) / divide
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


class EtoX(nn.Module):
    def __init__(self, de, dx):
        super().__init__()
        self.lin = nn.Linear(4 * de, dx)

    def forward(self, E, e_mask2):
        """ E: bs, n, n, de"""
        bs, n, _, de = E.shape
        e_mask2 = e_mask2.expand(-1, n, -1, de)
        float_imask = 1 - e_mask2.float()
        m = E.sum(dim=2) / torch.sum(e_mask2, dim=2)
        mi = (E + 1e5 * float_imask).min(dim=2)[0]
        ma = (E - 1e5 * float_imask).max(dim=2)[0]
        std = torch.sum(((E - m[:, :, None, :]) ** 2) * e_mask2, dim=2) / torch.sum(e_mask2, dim=2)
        z = torch.cat((m, mi, ma, std), dim=2)
        out = self.lin(z)
        return out


def masked_softmax(x, mask, **kwargs):
    if torch.sum(mask) == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)


class SetNorm(nn.LayerNorm):
    def __init__(self, feature_dim=None, **kwargs):
        super().__init__(normalized_shape=feature_dim, **kwargs)
        self.weights = nn.Parameter(torch.empty(1, 1, feature_dim))
        self.biases = nn.Parameter(torch.empty(1, 1, feature_dim))
        torch.nn.init.constant_(self.weights, 1.)
        torch.nn.init.constant_(self.biases, 0.)

    def forward(self, x, x_mask):
        bs, n, d = x.shape
        divide = torch.sum(x_mask, dim=1, keepdim=True) * d  # bs
        means = torch.sum(x * x_mask, dim=[1, 2], keepdim=True) / divide
        var = torch.sum((x - means) ** 2 * x_mask, dim=[1, 2], keepdim=True) / (divide + self.eps)
        out = (x - means) / (torch.sqrt(var) + self.eps)
        out = out * self.weights + self.biases
        out = out * x_mask
        return out


class GraphNorm(nn.LayerNorm):
    def __init__(self, feature_dim=None, **kwargs):
        super().__init__(normalized_shape=feature_dim, **kwargs)
        self.weights = nn.Parameter(torch.empty(1, 1, 1, feature_dim))
        self.biases = nn.Parameter(torch.empty(1, 1, 1, feature_dim))
        torch.nn.init.constant_(self.weights, 1.)
        torch.nn.init.constant_(self.biases, 0.)

    def forward(self, E, emask1, emask2):
        bs, n, _, d = E.shape
        divide = torch.sum(emask1 * emask2, dim=[1, 2], keepdim=True) * d  # bs
        means = torch.sum(E * emask1 * emask2, dim=[1, 2], keepdim=True) / divide
        var = torch.sum((E - means) ** 2 * emask1 * emask2, dim=[1, 2], keepdim=True) / (divide + self.eps)
        out = (E - means) / (torch.sqrt(var) + self.eps)
        out = out * self.weights + self.biases
        out = out * emask1 * emask2
        return out
