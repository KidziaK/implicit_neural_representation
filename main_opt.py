import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import trimesh
import mcubes
import open3d as o3d


# ---------------------------------------------------------
# 0. Reproducibility
# ---------------------------------------------------------
def set_seed(seed=42):
    """Locks the RNG for consistent initialization across runs."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------
# 1. Model Definition (3D)
# ---------------------------------------------------------
class SDFNetSmooth3D(nn.Module):
    """
    Implicit Neural Representation using Softplus.
    (Required because curvature needs non-zero 2nd spatial derivatives).
    """

    def __init__(self, hidden_dim=128):
        super().__init__()
        # Using Softplus(beta=20) for a relatively tight approximation of ReLU
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Softplus(beta=20),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(beta=20),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(beta=20),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------
# 2. 3D Data Loading and Sampling
# ---------------------------------------------------------
def load_and_normalize_mesh(obj_path, n_samples=20000):
    """Loads mesh, normalizes it, and samples surface points."""
    print(f"Loading mesh from: {obj_path}")
    mesh = trimesh.load(obj_path, force='mesh')

    # Normalization: Center at origin
    center = mesh.vertices.mean(axis=0)
    mesh.vertices -= center

    # Normalization: Scale to fit roughly within a radius of 0.9
    max_dist = np.max(np.linalg.norm(mesh.vertices, axis=1))
    scale_factor = 0.9 / max_dist
    mesh.vertices *= scale_factor
    print(f"Mesh normalized. Scale factor applied: {scale_factor:.4f}")

    # Sample boundary points
    surface_points, _ = trimesh.sample.sample_surface(mesh, n_samples)
    return torch.tensor(surface_points, dtype=torch.float32)


def sample_interior_points_uniform_3d(n, bounds=1.0, device="cpu"):
    """Sample uniform spatial points across the 3D volume."""
    return (torch.rand(n, 3, device=device) * 2 - 1.0) * bounds


def build_eik_linearization_points(Xb, Xi, n_eik):
    """Combine surface and interior points to enforce constraints."""
    X_all = torch.cat([Xb, Xi], dim=0)
    if n_eik < X_all.shape[0]:
        idx = torch.randperm(X_all.shape[0], device=X_all.device)[:n_eik]
        return X_all[idx]
    return X_all


# ---------------------------------------------------------
# 3. Guided Projection / Coupled Gauss-Newton Core
# ---------------------------------------------------------
def flatten_params(model):
    """Extracts and flattens the model's parameters."""
    tensors = []
    shapes = []
    for p in model.parameters():
        shapes.append(p.shape)
        tensors.append(p.data.reshape(-1))
    return torch.cat(tensors), shapes


def unflatten_to_model(model, theta_flat, shapes):
    """Applies the updated flattened parameters back to the model."""
    idx = 0
    for p, shape in zip(model.parameters(), shapes):
        numel = torch.prod(torch.tensor(shape)).item()
        p.data = theta_flat[idx:idx + numel].reshape(shape)
        idx += numel


def build_joint_residual_and_jacobian_dense(model, Xb, Xe, Xc, mu_eik, mu_curv):
    """
    Computes Dirichlet, Eikonal, and Curvature residuals & Jacobians.
    """
    params = list(model.parameters())

    # -- 1. Dirichlet Block (gb) & Jacobian (Jb) [Zero level-set] --
    gb = model(Xb).squeeze(-1)
    Jb_rows = []
    for i in range(gb.shape[0]):
        grads_i = torch.autograd.grad(gb[i], params, retain_graph=True,
                                      create_graph=False)
        Jb_rows.append(torch.cat([gj.reshape(-1) for gj in grads_i]))
    Jb = torch.stack(Jb_rows, dim=0) if len(Jb_rows) > 0 else torch.empty(0, sum(
        p.numel() for p in params), device=Xb.device)

    # -- 2. Eikonal Block (r_e) & Jacobian (Je) [Gradient norm = 1] --
    Xew = Xe.detach().clone().requires_grad_(True)
    pred_e = model(Xew).squeeze(-1)

    grad_xe = torch.autograd.grad(
        outputs=pred_e, inputs=Xew, grad_outputs=torch.ones_like(pred_e),
        create_graph=True, retain_graph=True
    )[0]
    r_e = grad_xe.norm(dim=-1) - 1.0

    Je_rows = []
    for j in range(r_e.shape[0]):
        grads_j = torch.autograd.grad(r_e[j], params, retain_graph=True,
                                      create_graph=False, allow_unused=True)
        row_j = []
        for p, gj in zip(params, grads_j):
            row_j.append((torch.zeros_like(p) if gj is None else gj).reshape(-1))
        Je_rows.append(torch.cat(row_j, dim=0))
    Je = torch.stack(Je_rows, dim=0) if len(Je_rows) > 0 else torch.empty(0, sum(
        p.numel() for p in params), device=Xe.device)

    # -- 3. Mean Curvature Block (r_c) & Jacobian (Jc) [Laplacian = 0] --
    # For an SDF where |grad| = 1, mean curvature simplifies exactly to the Laplacian
    Xcw = Xc.detach().clone().requires_grad_(True)
    pred_c = model(Xcw).squeeze(-1)

    # First spatial derivative
    grad_xc = torch.autograd.grad(
        outputs=pred_c, inputs=Xcw, grad_outputs=torch.ones_like(pred_c),
        create_graph=True, retain_graph=True
    )[0]

    # Second spatial derivative (Laplacian = trace of Hessian)
    laplacian = torch.zeros(Xcw.shape[0], device=Xcw.device)
    for i in range(3):
        grad2_xc_i = torch.autograd.grad(
            outputs=grad_xc[:, i], inputs=Xcw,
            grad_outputs=torch.ones_like(grad_xc[:, i]),
            create_graph=True, retain_graph=True
        )[0]
        laplacian += grad2_xc_i[:, i]

    r_c = laplacian  # Target mean curvature is 0

    Jc_rows = []
    for j in range(r_c.shape[0]):
        grads_j = torch.autograd.grad(r_c[j], params, retain_graph=True,
                                      create_graph=False, allow_unused=True)
        row_j = []
        for p, gj in zip(params, grads_j):
            row_j.append((torch.zeros_like(p) if gj is None else gj).reshape(-1))
        Jc_rows.append(torch.cat(row_j, dim=0))
    Jc = torch.stack(Jc_rows, dim=0) if len(Jc_rows) > 0 else torch.empty(0, sum(
        p.numel() for p in params), device=Xc.device)

    # Assemble dense dual system
    sqrt_mu_eik = math.sqrt(mu_eik)
    sqrt_mu_curv = math.sqrt(mu_curv)

    A = torch.cat([Jb, sqrt_mu_eik * Je, sqrt_mu_curv * Jc], dim=0)
    r = torch.cat(
        [gb.detach(), sqrt_mu_eik * r_e.detach(), sqrt_mu_curv * r_c.detach()], dim=0)

    aux = {
        "mean_abs_boundary_before": gb.abs().mean().item() if gb.numel() > 0 else 0.0,
        "mean_abs_eik_linear_before": r_e.abs().mean().item() if r_e.numel() > 0 else 0.0,
        "mean_abs_curv_before": r_c.abs().mean().item() if r_c.numel() > 0 else 0.0,
    }
    return A.detach(), r.detach(), aux


def guided_step_coupled_gn_dense_dual(model, Xb, Xe, Xc, mu_eik, mu_curv, eps,
                                      alpha_joint):
    A, r, aux = build_joint_residual_and_jacobian_dense(model, Xb, Xe, Xc, mu_eik,
                                                        mu_curv)
    m = A.shape[0]

    # Upcast to float64 for stable solving
    A_f64 = A.to(torch.float64)
    r_f64 = r.to(torch.float64)

    Gram = A_f64 @ A_f64.T
    Gram = 0.5 * (Gram + Gram.T)
    Gram = Gram + eps * torch.eye(m, device=A.device, dtype=torch.float64)

    try:
        chol = torch.linalg.cholesky(Gram)
        lam = torch.cholesky_solve(r_f64.unsqueeze(-1), chol).squeeze(-1)
    except RuntimeError:
        lam = torch.linalg.lstsq(Gram, r_f64.unsqueeze(-1)).solution.squeeze(-1)

    delta_vec = -(A_f64.T @ lam).to(torch.float32)
    delta_step = alpha_joint * delta_vec

    # Update parameters directly
    theta_before, shapes = flatten_params(model)
    theta_after = theta_before + delta_step
    unflatten_to_model(model, theta_after, shapes)

    with torch.no_grad():
        gb_after = model(Xb).squeeze(-1)

        Xew = Xe.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            pred_e = model(Xew).squeeze(-1)
            grad_xe = torch.autograd.grad(pred_e, Xew, torch.ones_like(pred_e))[0]
        r_e_after = grad_xe.norm(dim=-1) - 1.0

        Xcw = Xc.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            pred_c = model(Xcw).squeeze(-1)
            grad_xc = torch.autograd.grad(pred_c, Xcw, torch.ones_like(pred_c),
                                          create_graph=True)[0]
            laplacian_after = torch.zeros(Xcw.shape[0], device=Xcw.device)
            for i in range(3):
                grad2_xc_i = \
                torch.autograd.grad(grad_xc[:, i], Xcw, torch.ones_like(grad_xc[:, i]),
                                    create_graph=True)[0]
                laplacian_after += grad2_xc_i[:, i]
        r_c_after = laplacian_after

    aux["mean_abs_boundary_after"] = gb_after.abs().mean().item()
    aux["mean_abs_eik_linear_after"] = r_e_after.abs().mean().item()
    aux["mean_abs_curv_after"] = r_c_after.abs().mean().item()

    return aux


# ---------------------------------------------------------
# 4. 3D Mesh Extraction and Visualization
# ---------------------------------------------------------
def extract_and_visualize_mesh(model, out_path="learned_sdf_mesh.obj", resolution=64,
                               bounds=1.0, device="cpu"):
    """Evaluates the volume and extracts mesh via object-oriented Marching Cubes."""
    print(f"\nEvaluating network over {resolution}^3 grid...")
    model.eval()

    lin = torch.linspace(-bounds, bounds, resolution)
    X, Y, Z = torch.meshgrid(lin, lin, lin, indexing="ij")
    grid_pts = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3).to(device)

    batch_size = 32768
    sdf_vals = []
    with torch.no_grad():
        for i in range(0, grid_pts.shape[0], batch_size):
            batch = grid_pts[i:i + batch_size]
            sdf_vals.append(model(batch).squeeze(-1))

    sdf_vol = torch.cat(sdf_vals, dim=0).reshape(resolution, resolution,
                                                 resolution).cpu().numpy()

    print("Running Marching Cubes via Object API...")
    dims = (resolution, resolution, resolution)
    size = (2.0 * bounds, 2.0 * bounds, 2.0 * bounds)
    sampling_interval = (
        size[0] / (resolution - 1),
        size[1] / (resolution - 1),
        size[2] / (resolution - 1)
    )
    offset = (-bounds, -bounds, -bounds)

    mc_engine = mcubes.MarchingCubes(dims, size, sampling_interval, offset,
                                     sdf_vol.ravel().tolist(), 0.0)
    mesh_out = mc_engine.generate(mcubes.MeshSide.OutsideOnly)

    verts = np.array([v.posit for v in mesh_out.vertices], dtype=np.float64)
    faces = np.array(mesh_out.indices, dtype=np.int32).reshape(-1, 3)

    print(f"Extracted {len(verts)} vertices and {len(faces)} faces.")

    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(verts)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
    mesh_o3d.compute_vertex_normals()

    o3d.io.write_triangle_mesh(out_path, mesh_o3d)
    print(f"Mesh successfully saved to: {os.path.abspath(out_path)}")

    print("Opening Open3D Viewer...")
    o3d.visualization.draw_geometries([mesh_o3d],
                                      window_name="Softplus Stage 1 Guided Projection")


# ---------------------------------------------------------
# 5. Training Runner
# ---------------------------------------------------------
def run_3d_pipeline():
    # -----------------------------------------------------
    # --- Configuration ---
    # -----------------------------------------------------
    seed_val = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"

    part = "00800136"

    # Data params
    obj_path = "/Users/mikolajkida/Downloads/objabc_0080_obj_v00/00800004/00800004_34b3515cba9e63dad1d598dd_trimesh_000.obj"
    out_path = "reconstructed_trimesh_smooth.obj"
    n_samples_boundary = 20000

    # Network params
    hidden_dim = 128

    # Training timeline
    total_iters = 1000

    # Solver batches & weights
    batch_b = 128  # Random mesh surface points per iter
    batch_i = 512  # Random uniform spatial points per iter
    batch_eik = 128  # Total points evaluated for Eikonal per iter

    # Curvature batch is intentionally kept extremely small.
    # Extracting the Jacobian of a Hessian involves deep nested autograd graphs.
    batch_curv = 32

    mu_eik = 1.0  # Weight for the Eikonal block
    mu_curv = 0.5  # Weight for the Zero Mean Curvature block
    eps = 1e-3  # Tikhonov damping factor
    alpha_joint = 0.5  # Step size scaling

    # Extraction params
    resolution = 150
    bounds = 1.0
    # -----------------------------------------------------

    set_seed(seed_val)
    print(f"Using device: {device}")

    # Load data
    full_boundary_points = load_and_normalize_mesh(obj_path,
                                                   n_samples=n_samples_boundary)

    # Initialize the smooth network
    model = SDFNetSmooth3D(hidden_dim=hidden_dim).to(device)

    print(f"\n--- Starting Smooth Architecture Optimization ---")
    model.train()

    for it in range(total_iters):
        # Sample mini-batches
        idx_b = torch.randint(0, full_boundary_points.shape[0], (batch_b,))
        Xb = full_boundary_points[idx_b].to(device)

        Xi = sample_interior_points_uniform_3d(batch_i, bounds=bounds, device=device)
        Xe = build_eik_linearization_points(Xb, Xi, batch_eik)

        # We sample a few random interior points strictly for the curvature penalty
        Xc = sample_interior_points_uniform_3d(batch_curv, bounds=bounds, device=device)

        info = guided_step_coupled_gn_dense_dual(
            model, Xb, Xe, Xc, mu_eik, mu_curv, eps, alpha_joint
        )

        if (it + 1) % 10 == 0:
            print(f"Iter {it + 1:03d}/{total_iters} | "
                  f"|Bound|: {info['mean_abs_boundary_before']:.4f}->{info['mean_abs_boundary_after']:.4f} | "
                  f"|Eik|: {info['mean_abs_eik_linear_before']:.4f}->{info['mean_abs_eik_linear_after']:.4f} | "
                  f"|Curv|: {info['mean_abs_curv_before']:.4f}->{info['mean_abs_curv_after']:.4f}")

    extract_and_visualize_mesh(
        model,
        out_path=out_path,
        resolution=resolution,
        bounds=bounds,
        device=device
    )


if __name__ == "__main__":
    run_3d_pipeline()