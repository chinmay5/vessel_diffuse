import os

import torch
import torch.nn.functional as F

from bootstrap.diff_nodes import Diffusion, Simple_FC
from environment_setup import PROJECT_ROOT_DIR
from midi import utils
from midi.diffusion import diffusion_utils
from midi.diffusion.noise_model import NoiseModel


class MarginalTwoStageNoiseModel(NoiseModel):
    def __init__(self, cfg, x_marginals, e_marginals, charges_marginals, y_classes, dataset_infos):
        super().__init__(cfg=cfg)
        self.X_classes = len(x_marginals)
        self.E_classes = len(e_marginals)
        self.charges_classes = len(charges_marginals)
        self.y_classes = y_classes
        self.X_marginals = x_marginals
        self.charges_marginals = charges_marginals

        self.Px = x_marginals.unsqueeze(0).expand(self.X_classes, -1).unsqueeze(0)

        print(f"The marginal distribution was {e_marginals}")
        if cfg.dataset.name == "vessel":
            e_marginals_new = torch.tensor([0.1, 0.3, 0.3, 0.3])
            self.E_marginals = e_marginals_new
        else:
            print("Using noise model for crwon dataset")
            if cfg.dataset.get("is_multiclass", False):
                # e_marginals_new = e_marginals
                e_marginals_new = torch.zeros_like(e_marginals)
                e_marginals_new[0] = 1 / 66
                e_marginals_new[1:] = 5 / 40  # equally for the rest of the samples
                e_marginals_new = e_marginals_new / torch.sum(e_marginals_new)
                assert e_marginals_new.sum() == 1, "The marginal distribution does not sum to 1"
            else:
                e_marginals_new = torch.tensor([0.1, 0.9])
            self.E_marginals = e_marginals_new
        self.Pe = e_marginals_new.unsqueeze(0).expand(self.E_classes, -1).unsqueeze(0).clone()
        self.get_long_term_probability()
        print(f"The new marginal distribution is {self.E_marginals}")
        self.Pcharges = charges_marginals.unsqueeze(0).expand(self.charges_classes, -1).unsqueeze(0)
        self.y_out = dataset_infos.output_dims.y

        # We will sample the positions based on the pretrained diffusion model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        diffusion_pos = Diffusion(input_size=3, device=device)
        diff_pos_model = Simple_FC(hidden_dim=cfg.model.diff_1_config['hidden_dim'],
                                   n_layers=cfg.model.diff_1_config['n_layers']).to(device)
        if cfg.dataset.name == "vessel":
            print("Vessap coordinate sampling")
            diff_pos_model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT_DIR, 'midi', 'models',
                                                                   'checks_vessap_best_coord_diff_model.pth')))
        elif cfg.dataset.name == "cow":
            print("CoW coordinate sampling")
            diff_pos_model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT_DIR, 'midi', 'models',
                                                                   'checks_cow_best_coord_diff_model.pth')))
        else:
            raise AttributeError(f"Invalid dataset choice: {cfg.dataset.name}")
        self.pos_denoiser = diff_pos_model
        self.diffusion_pos = diffusion_pos

    def get_long_term_probability(self):
        # The stationary state is defined based on the left eigenvectors and eigenvalues
        eigenvalues, eigenvectors = torch.linalg.eig(self.Pe.squeeze(0).T)

        # Find the index of the eigenvalue closest to 1
        eigenvalue_index = torch.argmin(torch.abs(eigenvalues - 1))

        # Normalize the eigenvector
        stationary_distribution = eigenvectors[:, eigenvalue_index] / eigenvectors[:, eigenvalue_index].sum()

        def is_real(complex_number):
            return torch.isclose(complex_number.imag, torch.zeros_like(complex_number.imag))

        assert torch.all(is_real(stationary_distribution)), "Stationary distribution has imaginary part."
        stationary_distribution = stationary_distribution.squeeze(0).to(torch.float)
        print("Stationary distribution:", stationary_distribution)
        return stationary_distribution

    def move_P_device(self, tensor):
        """ Move the transition matrices to the device specified by tensor."""
        # Let us also move some important variables to cuda
        self._alphas = self._alphas.to(tensor.device)
        self._alphas_bar = self._alphas_bar.to(tensor.device)
        self._betas = self._betas.to(tensor.device)
        return diffusion_utils.PlaceHolder(X=self.Px.float().to(tensor.device),
                                           charges=self.Pcharges.float().to(tensor.device),
                                           E=self.Pe.float().to(tensor.device).float(),
                                           y=None, pos=None,
                                           directed=self.is_directed)

    def get_Qt(self, t_int):
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        P = self.move_P_device(t_int)
        kwargs = {'device': t_int.device, 'dtype': torch.float32}

        bx = self.get_beta(t_int=t_int, key='x').unsqueeze(1)
        q_x = bx * P.X + (1 - bx) * torch.eye(self.X_classes, **kwargs).unsqueeze(0)

        bc = self.get_beta(t_int=t_int, key='c').unsqueeze(1)
        q_c = bc * P.charges + (1 - bc) * torch.eye(self.charges_classes, **kwargs).unsqueeze(0)

        be = self.get_beta(t_int=t_int, key='e').unsqueeze(1)
        q_e = be * P.E + (1 - be) * torch.eye(self.E_classes, **kwargs).unsqueeze(0)

        return utils.PlaceHolder(X=q_x, charges=q_c, E=q_e, y=None, pos=None, directed=self.is_directed)

    def get_Qt_bar(self, t_int):
        """ Returns t-step transition matrices for X and E, from step 0 to step t.
            Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) / K

            alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
            returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        a_x = self.get_alpha_bar(t_int=t_int, key='x').unsqueeze(1)
        a_c = self.get_alpha_bar(t_int=t_int, key='c').unsqueeze(1)
        a_e = self.get_alpha_bar(t_int=t_int, key='e').unsqueeze(1)
        # a_y = self.get_alpha_bar(t_int=t_int, key='y').unsqueeze(1)

        P = self.move_P_device(t_int)
        # [X, charges, E, y, pos]
        dev = t_int.device
        q_x = a_x * torch.eye(self.X_classes, device=dev).unsqueeze(0) + (1 - a_x) * P.X
        q_c = a_c * torch.eye(self.charges_classes, device=dev).unsqueeze(0) + (1 - a_c) * P.charges
        q_e = a_e * torch.eye(self.E_classes, device=dev).unsqueeze(0) + (1 - a_e) * P.E

        assert ((q_x.sum(dim=2) - 1.).abs() < 1e-4).all(), q_x.sum(dim=2) - 1
        assert ((q_e.sum(dim=2) - 1.).abs() < 1e-4).all()

        return utils.PlaceHolder(X=q_x, charges=q_c, E=q_e, y=None, pos=None)

    def sample_limit_dist(self, node_mask, is_directed=False):
        """ Sample from the limit distribution of the diffusion process"""
        bs, n_max = node_mask.shape
        x_limit = self.X_marginals.expand(bs, n_max, -1)
        e_limit = self.E_marginals[None, None, None, :].expand(bs, n_max, n_max, -1)
        charges_limit = self.charges_marginals.expand(bs, n_max, -1)
        if 'x' in self.skip_noise_list:
            print("Using lookup based node degree")
            raise AttributeError("No point in skipping node degree denoising. It is dummy.")
            # U_X = deg.to(node_mask.device)
        else:
            print("Sampling random node degree")
            U_X = x_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max).to(node_mask.device)
            U_X = F.one_hot(U_X, num_classes=x_limit.shape[-1]).float()
        U_c = charges_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max).to(node_mask.device)
        U_E = e_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max, n_max).to(node_mask.device)

        y = torch.zeros((bs, 0), device=node_mask.device)
        U_E = F.one_hot(U_E, num_classes=e_limit.shape[-1]).float()
        U_c = F.one_hot(U_c, num_classes=charges_limit.shape[-1]).float()

        if not is_directed:
            # Here the Upper triangular part of the adjacency matrix is connected to its transpose.
            # This way, the graph is assured to be undirected.
            # Get upper triangular part of edge noise, without main diagonal
            upper_triangular_mask = torch.zeros_like(U_E)
            indices = torch.triu_indices(row=U_E.size(1), col=U_E.size(2), offset=1)
            upper_triangular_mask[:, indices[0], indices[1], :] = 1

            U_E = U_E * upper_triangular_mask
            U_E = (U_E + torch.transpose(U_E, 1, 2))
            assert (U_E == torch.transpose(U_E, 1, 2)).all()

        required_pos = torch.randn(node_mask.shape[0], node_mask.shape[1], 3)
        print("Sampling coordinate locations")
        pos = self.diffusion_pos.sample(model=self.pos_denoiser, pos=required_pos, node_mask=node_mask)
        pos = pos * node_mask.unsqueeze(-1)

        t_array = pos.new_ones((pos.shape[0], 1))
        t_int_array = self.T * t_array.long()
        return utils.PlaceHolder(X=U_X, E=U_E, y=y, pos=pos, t_int=t_int_array, t=t_array, charges=U_c,
                                 node_mask=node_mask, directed=is_directed).mask(node_mask)

    def sample_zs_from_zt_and_pred(self, z_t, pred, s_int, is_directed=False):
        """Samples from zs ~ p(zs | zt). Only used during sampling. """
        bs, n, dxs = z_t.X.shape
        node_mask = z_t.node_mask
        t_int = z_t.t_int

        # Retrieve transitions matrix
        Qtb = self.get_Qt_bar(t_int=t_int)
        Qsb = self.get_Qt_bar(t_int=s_int)
        Qt = self.get_Qt(t_int)

        assert torch.all(t_int == t_int[0]), "All denoising steps should be identical for the batch"
        # node mask has already been removed
        pos = z_t.pos

        # denoise connectivity
        # Normalize predictions for the categorical features
        pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0
        pred_charges = F.softmax(pred.charges, dim=-1)

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=z_t.X,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=z_t.E,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)

        p_s_and_t_given_0_c = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=z_t.charges,
                                                                                           Qt=Qt.charges,
                                                                                           Qsb=Qsb.charges,
                                                                                           Qtb=Qtb.charges)

        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X  # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)  # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        weighted_c = pred_charges.unsqueeze(-1) * p_s_and_t_given_0_c  # bs, n, d0, d_t-1
        unnormalized_prob_c = weighted_c.sum(dim=2)  # bs, n, d_t-1
        unnormalized_prob_c[torch.sum(unnormalized_prob_c, dim=-1) == 0] = 1e-5
        prob_c = unnormalized_prob_c / torch.sum(unnormalized_prob_c, dim=-1, keepdim=True)  # bs, n, d_t-1

        # Checking value for the edges
        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E  # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_c.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, prob_c, node_mask=z_t.node_mask,
                                                             is_directed=is_directed)

        X_s = F.one_hot(sampled_s.X, num_classes=self.X_classes).float()
        charges_s = F.one_hot(sampled_s.charges, num_classes=self.charges_classes).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.E_classes).float()

        if not is_directed:
            assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (z_t.X.shape == X_s.shape) and (z_t.E.shape == E_s.shape)

        if 'x' in self.skip_noise_list:
            raise AttributeError("No point in skipping node degree denoising. It is dummy.")
        z_s = utils.PlaceHolder(X=X_s, charges=charges_s,
                                E=E_s, y=z_t.y, pos=pos,
                                t_int=s_int, t=s_int / self.T, node_mask=node_mask,
                                directed=is_directed).mask(node_mask)
        return z_s


class UniformTwoStageNoiseModel(MarginalTwoStageNoiseModel):
    def __init__(self, cfg, output_dims, x_marginals, e_marginals, charges_marginals, y_classes, dataset_infos):
        super().__init__(cfg=cfg, x_marginals=x_marginals, e_marginals=e_marginals, charges_marginals=charges_marginals,
                         dataset_infos=dataset_infos, y_classes=y_classes)
        self.X_classes = output_dims.X
        self.charges_classes = output_dims.charges
        self.E_classes = output_dims.E
        self.y_classes = output_dims.y
        self.X_marginals = torch.ones(self.X_classes) / self.X_classes
        self.charges_marginals = torch.ones(self.charges_classes) / self.charges_classes

        self.y_marginals = torch.ones(self.y_classes) / self.y_classes
        self.Px = torch.ones(1, self.X_classes, self.X_classes) / self.X_classes
        self.Pcharges = torch.ones(1, self.charges_classes, self.charges_classes) / self.charges_classes
        self.Pe = torch.ones(1, self.E_classes, self.E_classes) / self.E_classes
        self.E_marginals = torch.ones(self.E_classes) / self.E_classes
        self.Py = torch.ones(1, self.y_classes, self.y_classes) / self.y_classes
        self.y_out = dataset_infos.output_dims.y
        print(f"{self.Pe}")
