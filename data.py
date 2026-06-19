"""
Data loading and result saving for EPPINN on ISLES2018 CT-perfusion cases.

Expected per-case directory layout:
    <case_dir>/
        CTP_4D.nii.gz      # 4D CTP volume (t, z, y, x) with 4D spacing for dt
        aif.npy            # arterial input function curve (length t)
        vof.npy            # (optional) venous output function for PVE correction
        time.npy           # (optional) explicit time points, else derived from spacing
        brainmask.nii.gz   # 3D brain mask (z, y, x)
        CBF.nii.gz         # (optional) template for output geometry
"""
import os

import numpy as np
import SimpleITK as sitk
import torch
from einops import rearrange, repeat
from scipy.ndimage import convolve


def _correct_aif_pve(aif, vof):
    """Partial-volume correction of AIF using VOF area ratio."""
    aif_bl = np.mean(aif[:4])
    vof_bl = np.mean(vof[:4])
    aif0 = aif - aif_bl
    vof0 = vof - vof_bl
    ratio = np.cumsum(vof0)[-1] / np.cumsum(aif0)[-1]
    return aif0 * ratio + aif_bl


def _smooth_aif(aif):
    return convolve(aif, np.array([0.25, 0.5, 0.25]), mode='nearest')


def _subtract_baseline(aif, curves, n=4):
    aif_bl = np.mean(aif[:n])
    curves_bl = np.mean(curves[:n], axis=0, keepdims=True)
    return aif - aif_bl, curves - curves_bl


def _build_coords(data_dict, use_adaptive_scale=False):
    """Build (t, x_01, y_01, z_01) coordinates for hash encoding."""
    time = data_dict['time']
    z_dim, t_dim, h, w, _ = time.shape

    if use_adaptive_scale:
        sx, sy, sz = data_dict.get('spacing', (1.0, 1.0, 1.0))[:3]
        norm = max(w * sx, h * sy, z_dim * sz, 256.0, 1e-6)
        ys = np.linspace(0, h - 1, h, dtype=np.float32)
        xs = np.linspace(0, w - 1, w, dtype=np.float32)
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        x01 = (xx * sx / norm).astype(np.float32)
        y01 = (yy * sy / norm).astype(np.float32)
        z01 = (np.arange(z_dim, dtype=np.float32) * sz / norm)
    else:
        ys = np.linspace(0, h - 1, h, dtype=np.float32)
        xs = np.linspace(0, w - 1, w, dtype=np.float32)
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        x01 = (xx / max(1.0, float(w - 1))).astype(np.float32)
        y01 = (yy / max(1.0, float(h - 1))).astype(np.float32)
        z01 = np.linspace(0, 1, z_dim, dtype=np.float32)

    z01 = z01[:, np.newaxis, np.newaxis].repeat(h, 1).repeat(w, 2)
    xyz_only = np.stack(
        [x01[np.newaxis].repeat(z_dim, 0), y01[np.newaxis].repeat(z_dim, 0), z01],
        axis=-1
    ).astype(np.float32)  # (z, h, w, 3)
    xyz_zt = np.tile(xyz_only[:, np.newaxis], (1, t_dim, 1, 1, 1))
    coords_xyz_zt = np.concatenate([time, xyz_zt], axis=-1)
    coords_xyz = rearrange(coords_xyz_zt, 'z t h w v -> z h w t v').astype(np.float32)

    data_dict['coordinates_xyz_only'] = xyz_only
    data_dict['coordinates_xyz'] = coords_xyz
    return data_dict


def load_ctp_data(case_path, temporal_smoothing=False, baseline_zero=True,
                  normalize_amplitude=False, normalize_time=False,
                  use_adaptive_scale=True):
    """Load one ISLES2018 case and prepare tensors for EPPINN training."""
    # 1. CTP 4D
    ctp = sitk.ReadImage(os.path.join(case_path, 'CTP_4D.nii.gz'))
    arr = sitk.GetArrayFromImage(ctp).astype(np.float32)  # (t, z, h, w) or (t, h, w)
    if arr.ndim == 3:
        arr = arr[:, np.newaxis]
    spacing = ctp.GetSpacing()
    time_step_s = float(spacing[-1]) if len(spacing) >= 4 else 1.0
    spatial_spacing = tuple(spacing[:3])

    # 2. AIF / VOF
    aif = np.load(os.path.join(case_path, 'aif.npy'))
    vof_path = os.path.join(case_path, 'vof.npy')
    vof = np.load(vof_path) if os.path.exists(vof_path) else None

    # 3. Brain mask
    mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(case_path, 'brainmask.nii.gz')))
    if mask.ndim == 2:
        mask = mask[np.newaxis]
    arr = arr * np.expand_dims(mask, 0)

    # 4. Time vector
    time_path = os.path.join(case_path, 'time.npy')
    if os.path.exists(time_path):
        time = np.load(time_path).astype(np.float32)
    else:
        time = np.arange(arr.shape[0], dtype=np.float32) * np.float32(time_step_s)

    # 5. AIF processing
    if vof is not None:
        aif = _correct_aif_pve(aif, vof)
    if baseline_zero:
        aif, arr = _subtract_baseline(aif, arr)
    if temporal_smoothing:
        aif = _smooth_aif(aif)

    curves = rearrange(arr, 't d h w -> d h w t')
    data_dict = {
        'aif': aif, 'time': time, 'mask': mask, 'curves': curves,
        'time_step_s': time_step_s, 'spacing': spatial_spacing,
    }

    # Template for output writing (prefer CBF if present)
    template_path = os.path.join(case_path, 'CBF.nii.gz')
    if not os.path.exists(template_path):
        template_path = os.path.join(case_path, 'brainmask.nii.gz')
    data_dict['template'] = sitk.ReadImage(template_path)

    # 6. Normalization
    if normalize_amplitude:
        max_val = max(float(np.max(data_dict['aif'])), 1e-12)
        data_dict['aif'] = data_dict['aif'] / max_val
        data_dict['curves'] = data_dict['curves'] / max_val
        data_dict['aif_scale'] = np.float32(max_val)
    else:
        data_dict['aif_scale'] = np.float32(1.0)

    if normalize_time:
        data_dict['std_t'] = np.float32(60.0)
        data_dict['time'] = data_dict['time'] / 60.0
    else:
        data_dict['std_t'] = np.float32(1.0)
        data_dict['time'] = np.array(data_dict['time'], dtype=np.float32)

    # Tile time to (z, t, h, w, 1)
    x_dim = data_dict['curves'].shape[1]
    data_dict['time'] = repeat(data_dict['time'], 't -> d t', d=mask.shape[0])
    data_dict['time'] = np.tile(
        data_dict['time'][..., np.newaxis, np.newaxis, np.newaxis],
        (1, x_dim, x_dim, 1)
    ).astype(np.float32)

    data_dict = _build_coords(data_dict, use_adaptive_scale=use_adaptive_scale)
    data_dict['aif_time'] = data_dict['time'][0]

    # Convert numpy arrays to torch tensors
    for k in list(data_dict.keys()):
        if isinstance(data_dict[k], np.ndarray):
            data_dict[k] = torch.as_tensor(data_dict[k], dtype=torch.float32)
    return data_dict


def _np2itk(arr, template):
    img = sitk.GetImageFromArray(arr, False)
    img.CopyInformation(template)
    return img


def _align_to_template(vol, template):
    dim = template.GetDimension()
    if dim == 2 and vol.ndim == 3 and vol.shape[0] == 1:
        return vol[0]
    if dim == 3 and vol.ndim == 2:
        return vol[None]
    return vol


def save_results(results, template, out_dir):
    """Save EPPINN outputs (perfusion maps + uncertainty) as NIfTI volumes."""
    os.makedirs(out_dir, exist_ok=True)
    name = {
        'cbf': 'CBF', 'cbv': 'CBV', 'mtt': 'MTT', 'delay': 'Delay', 'tmax': 'Tmax',
        'aleatoric': 'AleatoricUncertainty',
        'epistemic': 'EpistemicUncertainty',
        'total': 'TotalUncertainty',
    }
    for k, v in results.items():
        if k not in name:
            continue
        arr = v.numpy() if hasattr(v, 'numpy') else np.asarray(v)
        arr = _align_to_template(arr, template)
        sitk.WriteImage(_np2itk(arr, template), os.path.join(out_dir, f"{name[k]}.nii.gz"))
