"""Reproducibility helpers copied from mypt.code_utils.pytorch_utils."""

import os
import random
import warnings

import numpy as np
import torch


def get_default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int = 69) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed=seed)
    # certain layers have no deterministic implementation
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

    env_var = os.getenv("CUBLAS_WORKSPACE_CONFIG")
    if env_var is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    elif env_var not in [":16:8"]:
        warnings.warn(
            f"the env variable 'CUBLAS_WORKSPACE_CONFIG' is set to the value {env_var}. setting it to: ':16:8' "
        )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    if "cuda" in get_default_device():
        torch.set_float32_matmul_precision("medium")


def set_worker_seed(_, seed: int = 69) -> None:
    np.random.seed(seed=seed)
    random.seed(seed)
