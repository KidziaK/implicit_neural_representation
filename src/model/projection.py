import torch
import torch.nn as nn

def project_last_layer_to_zero_on_surface(
    model: nn.Module,
    surface_pts: torch.Tensor,
    eps: float = 1e-6,
    cap_n: int = 64,
) -> dict:
    """One projection step on a capped subset of surface_pts.

    We solve:  J δp ≈ -r   with δp = [δw, δb], where
      r = f(surface_pts)  (N x 1),
      J = [H, 1]          (N x (D+1)),
      H = h(surface_pts)  (N x D).

    Using a Tikhonov-regularized minimum-norm update:
      δp = - Jᵀ (J Jᵀ + eps I)^{-1} r

    Returns metrics for QC and debugging.
    """
    device = next(model.parameters()).device
    x = surface_pts.to(device)

    # Cap the projection system size to keep it cheap and stable.
    n = min(cap_n, x.shape[0])
    x = x[:n]

    # Compute features and current residual.
    with torch.no_grad():
        H = model.backbone(x)            # (N, D)
        lin = model.readout
        r = lin(H)                                     # (N, 1)

    pre_max = float(r.abs().max().item())
    pre_l2 = float(torch.linalg.vector_norm(r).item())

    # Build Jacobian J = [H, 1].
    ones = torch.ones((n, 1), device=device, dtype=H.dtype)
    J = torch.cat([H, ones], dim=1)                # (N, D+1)

    # Solve (J Jᵀ + eps I) y = r, then δp = -Jᵀ y.
    JJt = J @ J.T                                  # (N, N)
    A = JJt + eps * torch.eye(n, device=device, dtype=JJt.dtype)

    # y has shape (N, 1)
    y = torch.linalg.solve(A, r)
    delta_p = -(J.T @ y).squeeze(1)                # (D+1,)

    # Apply update to last layer parameters.
    D = H.shape[1]
    delta_w = delta_p[:D].view_as(lin.weight)
    delta_b = delta_p[D:].view_as(lin.bias)

    with torch.no_grad():
        lin.weight.add_(delta_w)
        lin.bias.add_(delta_b)

    # Residual after update (same subset).
    with torch.no_grad():
        r_post = lin(H)
    post_max = float(r_post.abs().max().item())
    post_l2 = float(torch.linalg.vector_norm(r_post).item())

    return {
        "n_proj": int(n),
        "pre_max": pre_max,
        "post_max": post_max,
        "pre_l2": pre_l2,
        "post_l2": post_l2,
        "eps": float(eps),
        "cap_n": int(cap_n),
    }
