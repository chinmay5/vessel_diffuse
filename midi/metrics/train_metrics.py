import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torchmetrics import MeanSquaredError, MeanMetric

from midi.metrics.abstract_metrics import CrossEntropyMetric


def differentiable_histogram(x, bins=255, min=0.0, max=1.0):
    if len(x.shape) == 4:
        n_samples, n_chns, _, _ = x.shape
    elif len(x.shape) == 2:
        n_samples, n_chns = 1, 1
    else:
        raise AssertionError('The dimension of input tensor should be 2 or 4.')

    hist_torch = torch.zeros(n_samples, n_chns, bins).to(x.device)
    delta = (max - min) / bins

    BIN_Table = torch.range(start=0, end=bins, step=1) * delta

    for dim in range(1, bins - 1, 1):
        h_r = BIN_Table[dim].item()  # h_r
        h_r_sub_1 = BIN_Table[dim - 1].item()  # h_(r-1)
        h_r_plus_1 = BIN_Table[dim + 1].item()  # h_(r+1)

        mask_sub = ((h_r > x) & (x >= h_r_sub_1)).float()
        mask_plus = ((h_r_plus_1 > x) & (x >= h_r)).float()

        hist_torch[:, :, dim] += torch.sum(((x - h_r_sub_1) * mask_sub).view(n_samples, n_chns, -1), dim=-1)
        hist_torch[:, :, dim] += torch.sum(((h_r_plus_1 - x) * mask_plus).view(n_samples, n_chns, -1), dim=-1)

    return hist_torch / delta


def compute_histogram(values):
    hist = differentiable_histogram(torch.tensor(values), bins=10, min=0, max=181)
    return hist / hist.sum()


class TrainLoss(nn.Module):
    """ Train with Cross entropy"""

    def __init__(self, lambda_train, cfg=None, dataset_infos=None):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.charges_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.degree_loss = MeanMetric()
        self.y_loss = CrossEntropyMetric()
        self.use_deg_loss = cfg.model.get("deg_loss", False)
        self.train_pos_mse = MeanSquaredError(sync_on_compute=False, dist_sync_on_step=False)
        print(f"Using node degree loss: {self.use_deg_loss}")
        self.lambda_train = lambda_train

    def forward(self, masked_pred, masked_true, log: bool, epoch=-1):
        """ Compute train metrics. Warning: the predictions and the true values are masked, but the relevant entriese
            need to be computed before calculating the loss

            masked_pred, masked_true: placeholders
            log : boolean. """

        node_mask = masked_true.node_mask
        bs, n = node_mask.shape

        true_pos = masked_true.pos[node_mask]  # q x 3
        masked_pred_pos = masked_pred.pos[node_mask]  # q x 3

        true_X = masked_true.X[node_mask]  # q x 4
        masked_pred_X = masked_pred.X[node_mask]  # q x 4

        true_charges = masked_true.charges[node_mask]  # q x 3
        masked_pred_charges = masked_pred.charges[node_mask]

        diag_mask = ~torch.eye(n, device=node_mask.device, dtype=torch.bool).unsqueeze(0).repeat(bs, 1, 1)
        edge_mask = diag_mask & node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
        masked_pred_E = masked_pred.E[edge_mask]  # r x num_categ
        true_E = masked_true.E[edge_mask]  # r x num_categ

        # Check that the masking is correct
        assert (true_X != 0.).any(dim=-1).all()
        assert (true_E != 0.).any(dim=-1).all()
        # Calculating pos_mse for the MiDi case.
        # For other cases, the loss is zero.
        loss_pos = self.train_pos_mse(masked_pred_pos, true_pos) if true_pos.numel() > 0 else 0.0
        loss_X = self.node_loss(masked_pred_X, true_X) if true_X.numel() > 0 else 0.0
        loss_charges = self.charges_loss(masked_pred_charges, true_charges) if true_charges.numel() > 0 else 0.0
        loss_E = self.edge_loss(masked_pred_E, true_E) if true_E.numel() > 0 else 0.0
        loss_deg = 0
        # Let us also compute the node degree loss
        if self.use_deg_loss:
            loss_deg = self.node_degree_loss(pred=masked_pred, true=masked_true, node_mask=node_mask, epoch=epoch,
                                             edge_mask=edge_mask)
        self.degree_loss.update(loss_deg)
        loss_y = self.y_loss(masked_pred.y, masked_true.y) if masked_true.y.numel() > 0 else -1
        batch_loss = (self.lambda_train[0] * loss_pos +
                      self.lambda_train[1] * loss_X +
                      self.lambda_train[2] * loss_charges +
                      self.lambda_train[5] * loss_deg +
                      self.lambda_train[3] * loss_E +
                      self.lambda_train[4] * loss_y)

        to_log = {
            "train_loss/pos_mse": self.lambda_train[0] * self.train_pos_mse.compute() if true_X.numel() > 0 else -1,
            "train_loss/X_CE": self.lambda_train[1] * self.node_loss.compute() if true_X.numel() > 0 else -1,
            "train_loss/charges_CE": self.lambda_train[
                                         2] * self.charges_loss.compute() if true_charges.numel() > 0 else -1,
            "train_loss/deg_kl": self.lambda_train[5] * loss_deg if true_X.numel() > 0 else -1,
            "train_loss/E_CE": self.lambda_train[3] * self.edge_loss.compute() if true_E.numel() > 0 else -1.0,
            "train_loss/y_CE": self.lambda_train[4] * self.y_loss.compute() if masked_true.y.numel() > 0 else -1.0,
            "train_loss/batch_loss": batch_loss.item()} if log else None

        if log and wandb.run:
            wandb.log(to_log, commit=True)
        return batch_loss, to_log

    def reset(self):
        for metric in [self.node_loss, self.charges_loss, self.edge_loss, self.y_loss,
                       self.train_pos_mse, self.degree_loss]:
            metric.reset()

    def node_degree_loss(self, pred, true, node_mask, epoch, edge_mask, max_node_deg=15):
        """
        Computes the difference in the distribution of the node degrees between the gt
        and the predicted nodes.
        :param pred: Placeholder. Has Adj of shape B, N, N, C
        :param true: Placeholder. Has Adj of shape B, N, N, C
        :param node_mask: B, N
        :param edge_mask: B, N, N
        :param max_node_deg: Maximum number of nodes in the dataset.
        :return: MSE loss between the mean and std dev of the two lists
        """
        # if epoch <= 10:
        #     return torch.tensor([0.]).to(pred.E.device)
        gt_adj = true.E
        pred_adj = self.get_adjacency(placeholder=pred)  # B, N, N, C
        # Collapsing the dimension and ignoring the background class
        gt_adj = gt_adj[..., 1:].sum(dim=-1)  # BxNxN
        pred_adj = pred_adj[..., 1:].sum(dim=-1)  # BxNxN
        # We ignore the padded edges
        gt_adj = gt_adj * edge_mask
        pred_adj = pred_adj * edge_mask
        # Compute the differentiable node degrees
        gt_degrees = gt_adj.sum(dim=2)  # BxN
        pred_degrees = pred_adj.sum(dim=2)
        gt_degrees = gt_degrees[node_mask]  # BxN
        pred_degrees = pred_degrees[node_mask]  # BxN
        gt_degree_prob = self.convert_list_to_hist(gt_degrees, max_node_deg)
        pred_degrees = self.convert_list_to_hist(pred_degrees, max_node_deg)
        return F.kl_div(torch.log(pred_degrees), gt_degree_prob, reduction='batchmean')

    def convert_list_to_hist(self, degrees, max_node_deg):
        # Flatten the differentiable node degrees
        hist = torch.zeros(max_node_deg + 1).to(degrees.device)
        # We make one last change of clubbing degree greater than max into max_degree
        # degrees[degrees > max_node_deg] = max_node_deg
        # Compute the histogram of the differentiable node degrees
        # degrees [0, max_node_deg -1]
        degrees = degrees + 1  # [1, max_node_deg]
        for i in range(max_node_deg + 1):
            if i in degrees:
                # hist[i] = ((degrees * (degrees == i + 1)) / (i + 1)).sum()
                hist[i] = (degrees == i + 1).sum()
        # handle numerical instability
        hist[hist == 0] = 1e-6
        # normalize the histogram to make it valid likelihood to be used in loss function
        norm_hist = hist / hist.sum()
        return norm_hist

    def get_adjacency(self, placeholder):
        b, n, _, c = placeholder.E.shape
        edges = F.gumbel_softmax(placeholder.E.view(-1, c), hard=True, dim=-1, tau=0.5)  # , tau=0.1 tau=0.01)
        edges = edges.view(b, n, n, c)
        return edges

    def log_epoch_metrics(self):
        epoch_pos_loss = self.train_pos_mse.compute().item() if self.train_pos_mse.total > 0 else -1.0
        epoch_node_loss = self.node_loss.compute().item() if self.node_loss.total_samples > 0 else -1.0
        epoch_charges_loss = self.charges_loss.compute().item() if self.charges_loss > 0 else -1.0
        epoch_edge_loss = self.edge_loss.compute().item() if self.edge_loss.total_samples > 0 else -1.0
        epoch_y_loss = self.y_loss.compute().item() if self.y_loss.total_samples > 0 else -1.0
        # Since we are feeding in dummy value of 0 for both these metrics, no need to check for num_samples
        epoch_deg_loss = self.degree_loss.compute().item()

        to_log = {
            "train_epoch/pos_mse": epoch_pos_loss,
            "train_epoch/degree_kl": epoch_deg_loss,
            "train_epoch/x_CE": epoch_node_loss,
            "train_epoch/charges_CE": epoch_charges_loss,
            "train_epoch/E_CE": epoch_edge_loss,
            "train_epoch/y_CE": epoch_y_loss}
        if wandb.run:
            wandb.log(to_log, commit=False)
        return to_log


class ValLoss(nn.Module):
    def __init__(self, lambda_train, cfg=None, dataset_infos=None):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.charges_loss = CrossEntropyMetric()
        self.val_y_loss = CrossEntropyMetric()
        self.lambda_val = lambda_train

    def forward(self, masked_pred, masked_true, log: bool):
        """ Compute val metrics. Warning: the predictions and the true values are masked, but the relevant entriese
            need to be computed before calculating the loss

            masked_pred, masked_true: placeholders
            log : boolean. """

        node_mask = masked_true.node_mask
        bs, n = node_mask.shape

        true_X = masked_true.X[node_mask]  # q x 4
        masked_pred_X = masked_pred.X[node_mask]  # q x 4

        true_charges = masked_true.charges[node_mask]  # q x 3
        masked_pred_charges = masked_pred.charges[node_mask]

        diag_mask = ~torch.eye(n, device=node_mask.device, dtype=torch.bool).unsqueeze(0).repeat(bs, 1, 1)
        edge_mask = diag_mask & node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
        masked_pred_E = masked_pred.E[edge_mask]  # r x num_categ
        true_E = masked_true.E[edge_mask]  # r x num_categ

        # Check that the masking is correct
        assert (true_X != 0.).any(dim=-1).all()
        assert (true_E != 0.).any(dim=-1).all()

        loss_X = self.node_loss(masked_pred_X, true_X) if true_X.numel() > 0 else 0.0
        loss_charges = self.charges_loss(masked_pred_charges, true_charges) if true_charges.numel() > 0 else 0.0
        loss_E = self.edge_loss(masked_pred_E, true_E) if true_E.numel() > 0 else 0.0
        loss_y = self.val_y_loss(masked_pred.y, masked_true.y) if masked_true.y.numel() > 0 else 0.0

        batch_loss = (self.lambda_val[1] * loss_X +
                      self.lambda_val[2] * loss_charges +
                      self.lambda_val[3] * loss_E +
                      self.lambda_val[4] * loss_y)

        to_log = {
            "val_loss/X_CE": self.lambda_val[1] * self.node_loss.compute() if true_X.numel() > 0 else -1,
            "val_loss/charges_CE": self.lambda_val[
                                       2] * self.charges_loss.compute() if true_charges.numel() > 0 else -1,
            "val_loss/E_CE": self.lambda_val[3] * self.edge_loss.compute() if true_E.numel() > 0 else -1.0,
            "val_loss/y_CE": self.lambda_val[4] * self.val_y_loss.compute() if masked_true.y.numel() > 0 else -1.0,
            "val_loss/batch_loss": batch_loss.item()} if log else None

        if log and wandb.run:
            wandb.log(to_log, commit=True)
        return batch_loss, to_log

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.val_y_loss, self.charges_loss]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_loss = self.node_loss.compute().item() if self.node_loss.total_samples > 0 else -1.0
        epoch_edge_loss = self.edge_loss.compute().item() if self.edge_loss.total_samples > 0 else -1.0
        epoch_charges_loss = self.charges_loss.compute().item() if self.charges_loss > 0 else -1.0
        epoch_y_loss = self.val_y_loss.compute().item() if self.val_y_loss.total_samples > 0 else -1.0

        to_log = {
            "val_epoch/x_CE": epoch_node_loss,
            "val_epoch/E_CE": epoch_edge_loss,
            "val_epoch/charges_CE": epoch_charges_loss,
            "val_epoch/y_CE": epoch_y_loss}
        if wandb.run:
            wandb.log(to_log, commit=False)
        return to_log


class TrainMolecularMetrics(nn.Module):
    def __init__(self, dataset_infos):
        super().__init__()
        # self.train_atom_metrics = AtomMetricsCE(dataset_infos=dataset_infos)
        # self.train_bond_metrics = BondMetricsCE()

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