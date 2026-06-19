"""
EPPINN model — Evidential Physics-Informed Neural Network for CT Perfusion.

Networks (SIREN-based, used in the published ISLES experiments):
  - NN_aif   : C_a(t)         (1D arterial input function)
  - NN_tissue: C(t, x, y, z)  (4D tissue concentration)
  - NN_ode   : (CBV, MTT, delay, alpha, beta, nu)(x, y, z)
                (parameter + NIG evidential head)

Physics residual (box-residue convolution model):
  r(t,x) = dC/dt - CBF * ( C_a(t-delay) - C_a(t-delay-MTT) )

Evidential NIG marginal NLL is used as the residual loss; aleatoric/epistemic
uncertainty is the closed-form decomposition of (alpha, beta, nu).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

import tinycudann as tcnn


# ============================================================
# SIREN
# ============================================================

class _Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)


class _Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0=1.0, c=6.0, is_first=False, activation=None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        w_std = (1.0 / dim_in) if is_first else (math.sqrt(c / dim_in) / w0)
        self.weight = nn.Parameter(torch.empty(dim_out, dim_in).uniform_(-w_std, w_std))
        self.bias = nn.Parameter(torch.empty(dim_out).uniform_(-w_std, w_std))
        self.activation = _Sine(w0) if activation is None else activation

    def forward(self, x):
        return self.activation(F.linear(x, self.weight, self.bias))


def _siren_stack(dim_in, dim_hidden, dim_out, num_layers, w0=1.0, w0_initial=15.0):
    layers = []
    for i in range(num_layers):
        layers.append(_Siren(
            dim_in=dim_in if i == 0 else dim_hidden,
            dim_out=dim_hidden,
            w0=w0_initial if i == 0 else w0,
            is_first=(i == 0),
        ))
    layers.append(_Siren(dim_in=dim_hidden, dim_out=dim_out, w0=w0, activation=nn.Identity()))
    return nn.Sequential(*layers)


# ============================================================
# Hash encoding (tiny-cuda-nn HashGrid)
# ============================================================

class MultiResHashGrid(nn.Module):
    def __init__(self, dim=3, n_levels=16, n_features_per_level=2,
                 log2_hashmap_size=15, base_resolution=16, finest_resolution=4096):
        super().__init__()
        per_level_scale = math.exp(
            (math.log(finest_resolution) - math.log(base_resolution)) / (n_levels - 1)
        )
        self.encoding = tcnn.Encoding(int(dim), {
            "otype": "HashGrid",
            "n_levels": int(n_levels),
            "n_features_per_level": int(n_features_per_level),
            "log2_hashmap_size": int(log2_hashmap_size),
            "base_resolution": int(base_resolution),
            "per_level_scale": per_level_scale,
        })
        self.output_dim = int(self.encoding.n_output_dims)

    def forward(self, x):
        return self.encoding(x.float().contiguous()).float()


# ============================================================
# SIREN MLPs
# ============================================================

class _MLP_Tissue(nn.Module):
    """C_tissue(t, encoded_xyz) -> scalar."""
    def __init__(self, hash_encoding_dim, dim_hidden, num_layers, w0=15.0):
        super().__init__()
        self.net = _siren_stack(1 + hash_encoding_dim, dim_hidden, 1, num_layers, w0=w0, w0_initial=w0)

    def forward(self, t, xyz_encoded):
        return self.net(torch.cat([t, xyz_encoded], dim=-1))


class _MLP_ODE(nn.Module):
    """Parameter + evidential head: encoded_xyz -> (CBV, MTT, delay, alpha, beta, nu)."""
    def __init__(self, hash_encoding_dim, dim_hidden, num_layers, dim_out, w0=15.0):
        super().__init__()
        self.net = _siren_stack(hash_encoding_dim, dim_hidden, dim_out, num_layers, w0=w0, w0_initial=w0)

    def forward(self, xyz_encoded):
        return self.net(xyz_encoded)


class _MLP_AIF(nn.Module):
    """C_a(t) -> scalar.  w0=1 for smooth global AIF."""
    def __init__(self, dim_hidden, num_layers):
        super().__init__()
        self.net = _siren_stack(1, dim_hidden, 1, num_layers, w0=1.0, w0_initial=1.0)

    def forward(self, t):
        out = self.net(t)
        return out[..., 0] if out.ndim > 1 else out


# ============================================================
# Parameter mapping helpers
# ============================================================

def _map_param(raw, scale, eps=0.0):
    """Map raw activation to non-negative physical units via softplus."""
    return scale * F.softplus(raw) + eps


def _get_time_scaling(config):
    """Protocol-adaptive scales for MTT (s), delay (s), CBV (frac)."""
    mtt = getattr(config, "mtt_scale_s", None)
    delay = getattr(config, "delay_scale_s", None)
    cbv = getattr(config, "cbv_scale_s", None)
    dt = getattr(config, "time_step_s", None)
    if mtt is None:
        mtt = max(4.0, min(10.0, 5.0 * dt)) * dt if dt and dt > 0 else 24.0
    if delay is None:
        delay = max(1.0, min(4.0, 2.0 * dt)) * dt if dt and dt > 0 else 3.0
    if cbv is None:
        cbv = 0.1
    return float(mtt), float(delay), float(cbv)


def _inv_softplus(y):
    y = max(float(y), 1e-8)
    return math.log(math.expm1(y))


def _initialize_pinn(model, config):
    """Initialize NN_ode last layer bias toward physiological priors."""
    mtt_scale_s, delay_scale_s, cbv_scale_s = _get_time_scaling(config)
    cbf0_1ps = float(getattr(config, "init_cbf0_1ps", 0.015))
    mtt0_s = float(getattr(config, "init_mtt0_s", 6.0))
    delay0_s = float(getattr(config, "init_delay0_s", 2.0))
    cbv0 = cbf0_1ps * mtt0_s

    def find_last_siren(seq):
        for layer in reversed(list(seq)):
            if isinstance(layer, _Siren):
                return layer
        return None

    last_w_std = 1e-3
    ode_last = find_last_siren(model.NN_ode.net)
    if ode_last is not None and ode_last.bias.numel() >= 3:
        with torch.no_grad():
            ode_last.weight.normal_(0.0, last_w_std)
            ode_last.bias.zero_()
            ode_last.bias[0] = _inv_softplus(cbv0 / max(cbv_scale_s, 1e-8))
            ode_last.bias[1] = _inv_softplus(mtt0_s / max(mtt_scale_s, 1e-8))
            ode_last.bias[2] = _inv_softplus(delay0_s / max(delay_scale_s, 1e-8))

    for name in ("NN_aif", "NN_tissue"):
        net = getattr(model, name, None)
        if net is None:
            continue
        last = find_last_siren(net.net)
        if last is not None:
            with torch.no_grad():
                last.weight.normal_(0.0, last_w_std)
                last.bias.zero_()


# ============================================================
# EPPINN model
# ============================================================

class EPPINN(nn.Module):
    """
    Evidential Physics-Informed Neural Network for CT Perfusion.

    Single-pass uncertainty quantification via Normal-Inverse-Gamma evidential
    prior on the physics residual.
    """

    def __init__(self, config, data_dict):
        super().__init__()
        self.config = config
        self.std_t = float(data_dict['std_t'])
        self.aif_scale = float(data_dict.get('aif_scale', 1.0))
        self.device = torch.device(
            f"cuda:{config.gpu_device}" if torch.cuda.is_available() and config.cuda else "cpu"
        )
        self.original_data_shape = None
        self.original_data_indices = None
        self.current_iteration = 0

        # Hash encoder for (x,y,z) -> features
        self.xyz_encoder = MultiResHashGrid(
            dim=3,
            n_levels=config.hash_n_levels,
            n_features_per_level=config.hash_n_features,
            log2_hashmap_size=config.hash_log2_size,
            base_resolution=config.hash_base_res,
            finest_resolution=config.hash_finest_res,
        )

        hash_dim = self.xyz_encoder.output_dim
        siren_w0 = config.siren_w0
        ode_out = 6  # (CBV, MTT, delay, alpha, beta, nu)

        self.NN_tissue = _MLP_Tissue(hash_dim, config.hidden_tissue, config.n_layers, w0=siren_w0)
        self.NN_ode = _MLP_ODE(hash_dim, config.hidden_ode, config.n_layers, ode_out, w0=siren_w0)
        self.NN_aif = _MLP_AIF(config.hidden_aif, config.n_layers)

        _initialize_pinn(self, config)

        self.optimizer = None
        self.scheduler = None
        self.to(self.device)
        self.float()

    # ----- forward primitives -----

    def _forward_NNs(self, aif_time, txyz):
        t = txyz[..., :1]
        xyz_encoded = self.xyz_encoder(txyz[..., 1:])
        c_tissue = self.NN_tissue(t, xyz_encoded)
        c_aif = self.NN_aif(aif_time)
        return c_aif, c_tissue

    def _forward_complete(self, aif_time, txyz):
        """Forward pass that returns AIF, tissue, residual, evidential logits."""
        t = txyz[..., :1]
        xyz_encoded = self.xyz_encoder(txyz[..., 1:])
        c_tissue = self.NN_tissue(t, xyz_encoded)
        c_aif = self.NN_aif(aif_time)

        v = torch.ones_like(c_tissue, requires_grad=True)
        g = torch.autograd.grad([c_tissue], [t], [v], create_graph=True)[0]
        w = torch.ones_like(g, requires_grad=True)
        c_tissue_dt = (1.0 / self.std_t) * torch.autograd.grad([g], [v], [w], create_graph=True)[0]

        t = t.detach()
        out = self.NN_ode(xyz_encoded)
        ode_params = out[..., :3]
        evi_logits = out[..., 3:]

        mtt_scale_s, delay_scale_s, cbv_scale_s = _get_time_scaling(self.config)
        cbv = _map_param(ode_params[..., :1], cbv_scale_s)
        mtt = _map_param(ode_params[..., 1:2], mtt_scale_s)
        delay = _map_param(ode_params[..., 2:], delay_scale_s)
        cbf = cbv / (mtt + 1e-6)

        t_a = t - delay / self.std_t
        t_b = t - delay / self.std_t - mtt / self.std_t
        c_aif_a = self.NN_aif(t_a)
        c_aif_b = self.NN_aif(t_b)
        residual = c_tissue_dt - cbf * (c_aif_a - c_aif_b).unsqueeze(-1)
        return c_aif, c_tissue, residual, evi_logits

    # ----- losses -----

    def _loss_data(self, aif, curves, c_aif, c_tissue):
        return F.l1_loss(aif.expand_as(c_aif), c_aif) + F.l1_loss(curves, c_tissue)

    def _nig_nll(self, y, alpha, beta, nu):
        """Student-t marginal NLL of the NIG prior, gamma fixed at 0."""
        eps = 1e-6
        alpha = alpha.clamp_min(1.0 + eps)
        beta = beta.clamp_min(eps)
        nu = nu.clamp_min(eps)
        two_beta = 2.0 * beta
        sq = y ** 2
        return (
            0.5 * torch.log(torch.tensor(math.pi, device=y.device) / nu)
            - alpha * torch.log(two_beta)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
            + (alpha + 0.5) * torch.log(two_beta + nu * sq + eps)
        ).mean()

    def _loss_residual_evidential(self, residual, evi_logits):
        if residual.ndim == 3:
            residual = residual.squeeze(-1)
        if residual.ndim == 1:
            residual = residual.unsqueeze(-1)

        eps = 1e-3
        alpha = 1.0 + F.softplus(evi_logits[..., 0:1]) + eps
        beta = F.softplus(evi_logits[..., 1:2]) + eps
        nu = F.softplus(evi_logits[..., 2:3]) + eps

        l1 = torch.mean(torch.abs(residual))
        nll = self._nig_nll(residual, alpha, beta, nu)

        # Linear annealing of the evidential terms over training
        total = max(int(self.config.iterations), 1)
        progress = min(1.0, self.current_iteration / total)
        edl_factor = progress  # warmup_ratio=0, anneal_ratio=1

        evidence = 2.0 * nu + alpha
        reg = torch.mean(residual.abs().detach() * evidence)
        return l1 + edl_factor * self.config.evi_lambda * (nll + self.config.evi_reg_weight * reg)

    # ----- optimization -----

    def _init_optimizer(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer = torch.optim.AdamW(
            params, lr=self.config.lr, weight_decay=self.config.weight_decay
        )

    def _init_scheduler(self, total_iters):
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.config.lr, total_steps=total_iters,
            pct_start=0.0, final_div_factor=5, anneal_strategy='cos',
        )

    def _step(self, b_aif_time, b_coords, batch_aif, batch_curves, batch_collo):
        b_coords.requires_grad = True
        batch_collo.requires_grad = True
        self.train()
        self.optimizer.zero_grad()

        c_aif, c_tissue = self._forward_NNs(b_aif_time, b_coords)
        loss_data = self._loss_data(batch_aif, batch_curves, c_aif, c_tissue)

        _, _, residual, evi_logits = self._forward_complete(b_aif_time, batch_collo)
        loss_res = self._loss_residual_evidential(residual, evi_logits)

        loss = self.config.lw_data * loss_data + self.config.lw_res * loss_res
        if torch.isnan(loss):
            raise ValueError("Loss is NaN")
        loss.backward()
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.grad_clip)
        self.optimizer.step()
        return loss.item()

    def _pretrain_aif(self, aif_time, aif, steps=10000, lr=5e-3):
        opt = torch.optim.Adam(self.NN_aif.parameters(), lr=lr)
        self.NN_aif.train()
        best, patience, counter = float('inf'), 100, 0
        pbar = tqdm(range(int(steps)), desc="AIF pretrain", leave=False)
        for _ in pbar:
            opt.zero_grad()
            pred = self.NN_aif(aif_time).reshape(-1, 1)
            loss = F.l1_loss(pred, aif)
            loss.backward()
            opt.step()
            if loss.item() < best - 1e-5:
                best, counter = loss.item(), 0
            else:
                counter += 1
            if counter >= patience:
                break

    # ----- training loop -----

    def fit(self, data_dict):
        curves = data_dict['curves']  # (z, h, w, t)
        aif = data_dict['aif'].to(self.device)
        if 'time_step_s' in data_dict:
            self.config.time_step_s = float(data_dict['time_step_s'])

        # AIF pretrain
        aif_time_src = data_dict.get('aif_time', data_dict['time'][0])
        if isinstance(aif_time_src, torch.Tensor) and aif_time_src.ndim >= 3:
            aif_time = aif_time_src.to(self.device)[:, 0, 0, :].reshape(-1, 1)
        else:
            aif_time = torch.as_tensor(aif_time_src).to(self.device).reshape(-1, 1)
        aif_target = aif.reshape(-1, 1)
        self._pretrain_aif(aif_time, aif_target)

        # Mask -> valid voxels
        mask_3d = data_dict['mask']
        if isinstance(mask_3d, torch.Tensor):
            mask_3d = mask_3d.cpu().numpy()
        valid = np.where(mask_3d == 1)
        self.original_data_indices = valid
        self.original_data_shape = mask_3d.shape

        if len(valid[0]) * curves.shape[-1] < self.config.batch_size:
            d, h, w = mask_3d.shape
            zeros = torch.zeros((d, h, w), dtype=torch.float32)
            return {k: zeros for k in ('cbf', 'cbv', 'mtt', 'delay', 'tmax')}

        self.data_coordinates_xyz = data_dict['coordinates_xyz_only'][valid].to(self.device)

        # Flatten (voxel, time) pairs
        data_curves_nt = curves.cpu().numpy()[valid]
        data_coords_ntv = data_dict['coordinates_xyz'].cpu().numpy()[valid]
        collo_ntv = np.zeros_like(data_coords_ntv)
        collo_ntv[..., 1:] = data_coords_ntv[..., 1:]

        data_curves_gpu = torch.from_numpy(data_curves_nt.reshape(-1, 1)).float().to(self.device)
        data_coords_gpu = torch.from_numpy(rearrange(data_coords_ntv, 'n t v -> (n t) v')).float().to(self.device)
        collo_coords_gpu = torch.from_numpy(rearrange(collo_ntv, 'n t v -> (n t) v')).float().to(self.device)

        data_time = data_dict['time'].to(self.device)
        if data_time.ndim == 5:
            c_min = float(torch.min(data_time[:, :, 0, 0, :]).cpu())
            c_max = float(torch.max(data_time[:, :, 0, 0, :]).cpu())
        else:
            c_min, c_max = float(torch.min(data_time)), float(torch.max(data_time))

        # Optimizer + scheduler
        self._init_optimizer()
        self._init_scheduler(self.config.iterations)

        pbar = tqdm(total=self.config.iterations, desc="Fitting EPPINN")
        for _ in range(self.config.iterations):
            # Random time collocation
            n = collo_coords_gpu.shape[0]
            collo_coords_gpu[:, 0:1] = torch.rand(n, 1, device=self.device) * (c_max - c_min) + c_min
            idx = torch.randint(0, len(data_curves_gpu), (self.config.batch_size,), device=self.device)

            b_curves = data_curves_gpu[idx]
            b_coords = data_coords_gpu[idx]
            b_collo = collo_coords_gpu[idx]

            self._step(aif_time, b_coords, aif, b_curves, b_collo)
            self.current_iteration += 1
            if self.scheduler is not None:
                self.scheduler.step()
            pbar.update(1)
        pbar.close()

        return self._get_results(data_dict)

    # ----- post-processing -----

    def _get_ode_params_full(self):
        xyz_encoded = self.xyz_encoder(self.data_coordinates_xyz)
        out = self.NN_ode(xyz_encoded)
        ode_params = out[..., :3]
        evi_logits = out[..., 3:6]

        mtt_scale_s, delay_scale_s, cbv_scale_s = _get_time_scaling(self.config)
        cbv = _map_param(ode_params[..., 0], cbv_scale_s)
        mtt = _map_param(ode_params[..., 1], mtt_scale_s)
        delay = _map_param(ode_params[..., 2], delay_scale_s)
        cbf = cbv / (mtt + 1e-6)

        scale = self.aif_scale  # auc_scale ≈ 1 after normalize_amplitude
        cbv = cbv * scale
        cbf = cbv / (mtt + 1e-6)
        tmax = delay + 0.5 * mtt

        d, h, w = self.original_data_shape
        dd, rr, cc = self.original_data_indices

        def grid(t):
            g = torch.zeros((d, h, w), dtype=t.dtype, device='cpu')
            g[dd, rr, cc] = t.detach().cpu()
            return g

        return {
            'cbf': grid(cbf), 'cbv': grid(cbv), 'mtt': grid(mtt),
            'delay': grid(delay), 'tmax': grid(tmax),
            'alpha': grid(evi_logits[..., 0]),
            'beta': grid(evi_logits[..., 1]),
            'nu': grid(evi_logits[..., 2]),
        }

    def _get_results(self, data_dict):
        maps = self._get_ode_params_full()
        eps = 1e-3
        alpha = 1.0 + F.softplus(maps['alpha']) + eps
        beta = F.softplus(maps['beta']) + eps
        nu = F.softplus(maps['nu']) + eps
        aleatoric = beta / (alpha - 1 + eps)
        epistemic = beta / (nu * (alpha - 1) + eps)

        rho = float(self.config.rho)
        hcf = float(self.config.hcf)
        unit = (100.0 / rho) * hcf
        cbf_clin = maps['cbf'] * unit * 60.0     # ml/100g/min
        cbv_clin = maps['cbv'] * unit            # ml/100g

        results = {
            'cbf': cbf_clin, 'cbv': cbv_clin,
            'mtt': maps['mtt'], 'delay': maps['delay'], 'tmax': maps['tmax'],
            'aleatoric': aleatoric, 'epistemic': epistemic,
            'total': aleatoric + epistemic,
        }
        return results
