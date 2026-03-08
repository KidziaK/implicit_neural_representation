import torch
from torch import Tensor


def ncadr_gaussian_curvature_loss(x: Tensor, grad_y: Tensor, eps: float = 1e-12) -> Tensor:
    H_cols = []
    for i in range(3):
        (grad_y_i,) = torch.autograd.grad(
            grad_y[..., i].sum(),
            x,
            retain_graph=True,
            create_graph=False,
        )
        H_cols.append(grad_y_i)

    H = torch.stack(H_cols, dim=-1)

    g = grad_y
    g_norm_sq = (g * g).sum(dim=-1, keepdim=True) + eps

    H_g = torch.cat([H, g.unsqueeze(-1)], dim=-1)
    bottom = torch.cat([g.unsqueeze(-2), torch.zeros_like(g[..., :1].unsqueeze(-2))], dim=-1)
    ext = torch.cat([H_g, bottom], dim=-2)

    det_ext = torch.linalg.det(ext)
    K = -det_ext / g_norm_sq.squeeze(-1)
    return K.abs().mean()
