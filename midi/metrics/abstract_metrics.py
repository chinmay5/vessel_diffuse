import torch
import torchmetrics
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import Metric, MeanSquaredError

from midi.diffusion.extra_features import ExtraFeatures
from midi.utils import PlaceHolder


# TODO: these full check updates are probably not necessary

class SumExceptBatchMetric(Metric):
    full_state_update = True

    def __init__(self):
        super().__init__(sync_on_compute=False, dist_sync_on_step=False)
        self.add_state('total_value', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, values) -> None:
        self.total_value += torch.sum(values)
        self.total_samples += values.shape[0]

    def compute(self):
        return self.total_value / max(self.total_samples, 1)


class SumExceptBatchMSE(MeanSquaredError):
    full_state_update = True

    def __init__(self):
        super().__init__(sync_on_compute=False, dist_sync_on_step=False)

    def update(self, preds: Tensor, target: Tensor) -> None:
        assert preds.shape == target.shape, f"Preds has shape {preds.shape} while target has shape {target.shape}"
        sum_squared_error, n_obs = self._mean_squared_error_update(preds, target)

        self.sum_squared_error += sum_squared_error
        self.total += n_obs

    def _mean_squared_error_update(self, preds: Tensor, target: Tensor):
        """ Updates and returns variables required to compute Mean Squared Error. Checks for same shape of input
            tensors.
                preds: Predicted tensor
                target: Ground truth tensor
            """
        diff = preds - target
        sum_squared_error = torch.sum(diff * diff)
        # n_obs = preds.shape[0]
        n_obs = target.numel()
        return sum_squared_error, n_obs


class SumExceptBatchKL(torchmetrics.KLDivergence):
    def __init__(self):
        super().__init__(log_prob=False, reduction='sum', sync_on_compute=False, dist_sync_on_step=False)

    def update(self, p, q) -> None:
        p = p.flatten(start_dim=1)
        q = q.flatten(start_dim=1)
        super().update(p, q)


class CrossEntropyMetric(Metric):
    full_state_update = True

    def __init__(self):
        super().__init__(sync_on_compute=False, dist_sync_on_step=False)
        self.add_state('total_ce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """ Update state with predictions and targets.
            preds: Predictions from model   (bs * n, d) or (bs * n * n, d)
            target: Ground truth values     (bs * n, d) or (bs * n * n, d). """
        target = torch.argmax(target, dim=-1)
        output = F.cross_entropy(preds, target, reduction='sum')
        self.total_ce += output
        self.total_samples += preds.size(0)

    def compute(self):
        return self.total_ce / self.total_samples


class DummyMetric(Metric):
    def __init__(self):
        self.total_samples = -1
        super().__init__()

    def update(self, *args, **kwargs):
        pass  # Do nothing

    def compute(self):
        return torch.tensor(-1.0)


class GraphFeatureMetric(Metric):
    full_state_update = True

    def __init__(self, cfg, dataset_infos):
        super().__init__(sync_on_compute=False, dist_sync_on_step=False)
        self.add_state('total_ce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.extra_features = ExtraFeatures(extra_features_type='loss', dataset_info=dataset_infos)

    def update(self, pred, target):
        """
        Pass the generated graph vs the ground truth graph
        :param pred:
        :param target:
        :return:
        """
        # Convert predictions into one hot
        clone_X, clone_E = pred.X, pred.E
        pred.X = F.one_hot(torch.argmax(pred.X, dim=-1), num_classes=target.X.shape[-1]).to(torch.float)
        # One more line of unnecessary tuning just to completely match gt.
        # Node of deg 0 does not exist
        empty_indices = torch.argmax(target.X, dim=-1) == 0
        pred.X[empty_indices] = target.X[empty_indices]
        # Now we shall take a look at the edges
        # Edges of type 0 exist hence, we need not do extra processing.
        pred.E = F.one_hot(torch.argmax(pred.E, dim=-1), num_classes=10)
        gt_conn_comp, gt_y_cycles = self.extra_features(target)
        pred_conn_comp, pred_y_cycles = self.extra_features(pred)
        # CE loss for the connected components
        pred_conn_comp, gt_conn_comp = pred_conn_comp.squeeze(1), gt_conn_comp.squeeze(1)
        # print(f"{pred_conn_comp.max()=} and {gt_conn_comp.max()=}; {gt_conn_comp.min()=}")
        conn_loss = F.mse_loss(pred_conn_comp.float(), gt_conn_comp.float(), reduction="sum")
        # conn_loss = F.cross_entropy(F.one_hot(pred_conn_comp, num_classes=105).to(torch.float),
        #                            gt_conn_comp, reduction='sum')
        cycle_loss = F.binary_cross_entropy_with_logits(pred_y_cycles, gt_y_cycles, reduction='sum')
        self.total_ce += conn_loss + cycle_loss
        self.total_samples += pred.X.size(0)  # batch size
        # Return the actual values
        pred.X, pred.E = clone_X, clone_E

    def compute(self):
        return self.total_ce / self.total_samples


class ProbabilityMetric(Metric):
    def __init__(self):
        """ This metric is used to track the marginal predicted probability of a class during training. """
        super().__init__(sync_on_compute=False, dist_sync_on_step=False)
        self.add_state('prob', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: Tensor) -> None:
        self.prob += preds.sum()
        self.total += preds.numel()

    def compute(self):
        return self.prob / self.total


class NLL(Metric):
    full_state_update = True

    def __init__(self):
        super().__init__(sync_on_compute=False, dist_sync_on_step=False)
        self.add_state('total_nll', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, batch_nll) -> None:
        self.total_nll += torch.sum(batch_nll)
        self.total_samples += batch_nll.numel()

    def compute(self):
        return self.total_nll / max(self.total_samples, 1)


class PosMSE(SumExceptBatchMSE):
    def update(self, preds: PlaceHolder, target: PlaceHolder):
        super().update(preds.pos, target.pos)


class XKl(SumExceptBatchKL):
    def update(self, preds: PlaceHolder, target: PlaceHolder):
        super().update(preds.X, target.X)


class ChargesKl(SumExceptBatchKL):
    def update(self, preds: PlaceHolder, target: PlaceHolder):
        super().update(preds.charges, target.charges)


class EKl(SumExceptBatchKL):
    def update(self, preds: PlaceHolder, target: PlaceHolder):
        super().update(preds.E, target.E)


class YKl(SumExceptBatchKL):
    def update(self, preds: PlaceHolder, target: PlaceHolder):
        super().update(preds.y, target.y)


class PosLogp(SumExceptBatchMetric):
    def update(self, preds, target):
        # TODO
        return -1


class XLogP(SumExceptBatchMetric):
    def update(self, preds, target):
        super().update(target.X * preds.X.log())


class ELogP(SumExceptBatchMetric):
    def update(self, preds, target):
        super().update(target.E * preds.E.log())


class YLogP(SumExceptBatchMetric):
    def update(self, preds, target):
        super().update(target.y * preds.y.log())


class NodeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class EdgeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)
