"""Minimal helpers for EPPINN training and post-processing."""
import os
import random

import numpy as np
import torch


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def drop_unphysical(results):
    """Clip perfusion maps to physiologically plausible ranges (ISLES)."""
    results = dict(results)
    results['cbf'] = np.clip(results['cbf'], 0, 1000)
    results['cbv'] = np.clip(results['cbv'], 0, 100)
    results['mtt'] = np.clip(results['mtt'], 0, 100)
    results['delay'] = np.clip(results['delay'], 0, 100)
    results['tmax'] = np.clip(results['tmax'], 0, 100)
    return results
