import random
import numpy as np
import torch
import yaml


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_gaussian_3d(patch_size, sigma_scale=0.125):
    sigma = patch_size * sigma_scale
    ax = np.arange(patch_size, dtype=np.float32) - (patch_size - 1) / 2.0
    gauss_1d = np.exp(-0.5 * (ax / sigma) ** 2)
    gauss_3d = gauss_1d[:, None, None] * gauss_1d[None, :, None] * gauss_1d[None, None, :]
    gauss_3d = gauss_3d / gauss_3d.max()
    gauss_3d = np.clip(gauss_3d, a_min=1e-4, a_max=None)
    return torch.from_numpy(gauss_3d).float()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
