"""
EPPINN training / inference entry-point.

    python train.py --case_dir /path/to/isles/case_001 --output_dir results
"""
import argparse
import os

import numpy as np
import torch

from data import load_ctp_data, save_results
from model import EPPINN
from utils import drop_unphysical, set_seed


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected")


def _build_parser():
    p = argparse.ArgumentParser(description="EPPINN — Evidential PINN for CT perfusion")
    p.add_argument('--case_dir', type=str, required=True,
                   help='Path to a single case directory containing NIfTI volumes')
    p.add_argument('--output_dir', type=str, default='results',
                   help='Directory to write CBF/CBV/MTT/Delay/Tmax and uncertainty maps')

    # Hardware / runtime
    p.add_argument('--cuda', type=_str2bool, default=True)
    p.add_argument('--gpu_device', type=int, default=0)
    p.add_argument('--seed', type=int, default=1)

    # Training schedule
    p.add_argument('--iterations', type=int, default=10000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch_size', type=int, default=25000)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--grad_clip', type=float, default=1.0)

    # Loss weights
    p.add_argument('--lw_data', type=float, default=1.0)
    p.add_argument('--lw_res', type=float, default=1.0)

    # Network sizes
    p.add_argument('--n_layers', type=int, default=3)
    p.add_argument('--hidden_tissue', type=int, default=128)
    p.add_argument('--hidden_ode', type=int, default=64)
    p.add_argument('--hidden_aif', type=int, default=16)
    p.add_argument('--siren_w0', type=int, default=15)

    # Hash encoding
    p.add_argument('--hash_n_levels', type=int, default=16)
    p.add_argument('--hash_n_features', type=int, default=2)
    p.add_argument('--hash_log2_size', type=int, default=15)
    p.add_argument('--hash_base_res', type=int, default=16)
    p.add_argument('--hash_finest_res', type=int, default=4096)
    p.add_argument('--adaptive_hash_scale', type=_str2bool, default=True)

    # Physical scaling
    p.add_argument('--rho', type=float, default=1.05)
    p.add_argument('--hcf', type=float, default=0.73)
    p.add_argument('--mtt_scale_s', type=float, default=None)
    p.add_argument('--delay_scale_s', type=float, default=None)
    p.add_argument('--cbv_scale_s', type=float, default=None)

    # Evidential head
    p.add_argument('--evi_lambda', type=float, default=0.01)
    p.add_argument('--evi_reg_weight', type=float, default=1e-3)
    return p


def main():
    config = _build_parser().parse_args()
    set_seed(config.seed)

    case_dir = config.case_dir
    case_id = os.path.basename(os.path.normpath(case_dir))

    data_dict = load_ctp_data(
        case_path=case_dir,
        temporal_smoothing=False,
        baseline_zero=True,
        normalize_amplitude=False,
        normalize_time=False,
        use_adaptive_scale=config.adaptive_hash_scale,
    )

    model = EPPINN(config, data_dict)
    results = model.fit(data_dict)

    # Convert torch -> numpy, drop unphysical ranges, apply brain mask
    results = {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in results.items()}
    results = drop_unphysical(results)

    mask = data_dict['mask']
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    mask = mask.astype(np.float32)
    for k in results:
        results[k] = results[k] * mask

    out_dir = os.path.join(config.output_dir, case_id, 'EPPINN')
    save_results(results, data_dict['template'], out_dir)
    print(f"Saved to {out_dir}")


if __name__ == '__main__':
    main()
