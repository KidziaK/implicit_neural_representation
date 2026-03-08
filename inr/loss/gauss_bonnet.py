import torch
from torch import Tensor


def gauss_bonnet_loss(x: Tensor, grad_y: Tensor, num_samples: int = 2) -> Tensor:
    laplacian_estimates = []
    trace_h2_estimates = []

    for _ in range(num_samples):
        v = torch.randint_like(x, low=0, high=2) * 2.0 - 1.0

        with torch.enable_grad():
            grad_v_prod = (grad_y * v).sum(dim=-1, keepdim=True)
            Hv = torch.autograd.grad(
                outputs=grad_v_prod,
                inputs=x,
                grad_outputs=torch.ones_like(grad_v_prod),
                retain_graph=True,
            )[0]

        laplacian_estimates.append((v * Hv).sum(dim=-1))
        trace_h2_estimates.append((Hv * Hv).sum(dim=-1))

    mean_laplacian = torch.stack(laplacian_estimates).mean(dim=0)
    mean_trace_h2 = torch.stack(trace_h2_estimates).mean(dim=0)

    curvature = 0.5 * (mean_laplacian**2 - mean_trace_h2)

    return curvature.abs().mean()
