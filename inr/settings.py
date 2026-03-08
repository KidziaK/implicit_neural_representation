import torch
from functools import cache


@cache
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
