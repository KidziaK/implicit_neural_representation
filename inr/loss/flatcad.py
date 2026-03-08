import torch
from torch import Tensor
import torch.nn.functional as F


def flatcad_loss(x: Tensor, grad_y: Tensor, num_samples: int = 2, eps: float = 1e-9) -> Tensor:
    n = F.normalize(grad_y, dim=-1, eps=eps)
    n = n.detach()

    duv_list = []
    for _ in range(num_samples):
        rand = torch.randn_like(n)
        v = F.normalize(rand - (rand * n).sum(dim=-1, keepdim=True) * n, dim=-1)
        u = F.normalize(torch.linalg.cross(n, v, dim=-1), dim=-1)

        with torch.enable_grad():
            gv = (grad_y * v).sum(dim=-1)
            Hv = torch.autograd.grad(
                outputs=gv,
                inputs=x,
                grad_outputs=torch.ones_like(gv),
                retain_graph=True,
            )[0]
        D_uv = (Hv * u).sum(dim=-1)
        duv_list.append(D_uv.abs())

    return torch.stack(duv_list).mean(dim=0).mean()
