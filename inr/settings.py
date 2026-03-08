import torch
from functools import cache


def _warmup_cuda_context() -> None:
    """Initialize CUDA context before first use to avoid cuBLAS warning during backward."""
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.init()
    except AttributeError:
        # Older PyTorch: trigger context with a dummy CUDA op
        torch.ones(1, device="cuda").cpu()


@cache
def get_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        _warmup_cuda_context()
    return device
