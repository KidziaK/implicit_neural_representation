import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from skimage import measure
import open3d as o3d
import os
from enum import Enum


class ActivationType(Enum):
    RELU = 1
    SOFTPLUS = 2
    SIREN = 3


class SineLayer(nn.Module):
    """
    A specialized linear layer with Sine activation required for SIREN.
    It includes the specific weight initialization scheme to keep activations stable.
    """

    def __init__(self, in_features, out_features, bias=True, is_first=False,
                 omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class DevelopableSDF(nn.Module):
    """Neural SDF optimized for developable CAD surfaces with selectable activations."""

    def __init__(self, activation_type=ActivationType.SIREN):
        super().__init__()
        self.activation_type = activation_type

        if activation_type == ActivationType.SIREN:
            # SIREN architecture: 4 hidden layers, 256 units each
            self.net = nn.Sequential(
                SineLayer(3, 256, is_first=True),
                SineLayer(256, 256, is_first=False),
                SineLayer(256, 256, is_first=False),
                SineLayer(256, 256, is_first=False),
                nn.Linear(256, 1)
            )
            # Special initialization for the final linear layer
            with torch.no_grad():
                self.net[-1].weight.uniform_(-np.sqrt(6 / 256) / 30,
                                             np.sqrt(6 / 256) / 30)

        else:
            # Standard MLP architecture
            layers = []
            dims = [3, 256, 256, 256, 256]
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if activation_type == ActivationType.SOFTPLUS:
                    layers.append(nn.Softplus(beta=100))
                elif activation_type == ActivationType.RELU:
                    layers.append(nn.ReLU())

            layers.append(nn.Linear(dims[-1], 1))
            self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def fast_double_trace_surrogate(sdf_func, x, num_samples=2):
    """
    Computes S(x) = (Delta phi)^2 - Tr(H^2) using Hutchinson.
    Returns S_surrogate, the gradient vectors, and the raw SDF values.
    """
    laplacian_estimates = []
    trace_h2_estimates = []

    for _ in range(num_samples):
        v = torch.randint_like(x, low=0, high=2) * 2.0 - 1.0

        with torch.enable_grad():
            y = sdf_func(x)

            # Gradient
            grad_phi = torch.autograd.grad(
                outputs=y, inputs=x,
                grad_outputs=torch.ones_like(y),
                create_graph=True
            )[0]

            # HVP
            grad_v_prod = (grad_phi * v).sum(dim=-1, keepdim=True)
            Hv = torch.autograd.grad(
                outputs=grad_v_prod, inputs=x,
                grad_outputs=torch.ones_like(grad_v_prod),
                retain_graph=True
            )[0]

        laplacian_estimates.append((v * Hv).sum(dim=-1))
        trace_h2_estimates.append((Hv * Hv).sum(dim=-1))

    mean_laplacian = torch.stack(laplacian_estimates).mean(dim=0)
    mean_trace_h2 = torch.stack(trace_h2_estimates).mean(dim=0)

    S_surrogate = (mean_laplacian ** 2 - mean_trace_h2)
    return S_surrogate, grad_phi, y


def generalized_charbonnier_loss(S, alpha=0.1, c=2.0):
    """Tolerates sharp corners where S(x) natively spikes to infinity."""
    return torch.mean((S ** 2 + c ** 2) ** alpha - (c ** 2) ** alpha)


def non_manifold_penalty(sdf_values, alpha_decay=50.0):
    """
    Dirichlet Non-Manifold penalty.
    Heavily penalizes points in the volume if their SDF value gets too close to 0.
    """
    return torch.mean(torch.exp(-alpha_decay * torch.abs(sdf_values)))


def load_and_sample_mesh(file_path, num_points=10000):
    """
    Loads an .obj file, samples its surface, and normalizes the points.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find the file at {file_path}")

    print(f"Loading mesh from {file_path}...")
    mesh = o3d.io.read_triangle_mesh(file_path)

    if mesh.is_empty():
        raise ValueError(f"Open3D failed to load the mesh. Check the file formatting.")

    print(f"Sampling {num_points} points uniformly from the surface...")
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    points = np.asarray(pcd.points)

    # Normalization: Center the points and scale them to fit within [-0.8, 0.8]
    center = np.mean(points, axis=0)
    points -= center
    max_dist = np.max(np.linalg.norm(points, axis=1))
    points = (points / max_dist) * 0.8

    return torch.tensor(points, dtype=torch.float32), pcd


def extract_and_visualize_mesh(model, device, original_pcd, output_path, resolution=128,
                               bounds=1.0):
    """
    Extracts the zero-level set using skimage's Marching Cubes, saves it, and visualizes it.
    """
    print("\nExtracting mesh with scikit-image Marching Cubes...")
    model.eval()

    x_lin = np.linspace(-bounds, bounds, resolution)
    y_lin = np.linspace(-bounds, bounds, resolution)
    z_lin = np.linspace(-bounds, bounds, resolution)
    X, Y, Z = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')

    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device)

    sdf_values = []
    chunk_size = 100000
    with torch.no_grad():
        for i in range(0, grid_points_tensor.shape[0], chunk_size):
            chunk = grid_points_tensor[i:i + chunk_size]
            sdf_chunk = model(chunk)
            sdf_values.append(sdf_chunk.cpu().numpy())

    sdf_volume = np.concatenate(sdf_values).reshape(
        (resolution, resolution, resolution))

    try:
        vertices, triangles, normals, values = measure.marching_cubes(sdf_volume,
                                                                      level=0.0)
    except ValueError as e:
        print(f"Marching cubes failed (network may not have converged): {e}")
        return

    vertices = (vertices / (resolution - 1)) * (2 * bounds) - bounds

    reconstructed_mesh = o3d.geometry.TriangleMesh()
    reconstructed_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    reconstructed_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    reconstructed_mesh.compute_vertex_normals()

    print(f"Saving reconstructed mesh to: {output_path}")
    o3d.io.write_triangle_mesh(output_path, reconstructed_mesh)

    print("Opening Open3D Viewer...")
    reconstructed_mesh.paint_uniform_color([0.4, 0.7, 0.9])
    original_pcd.paint_uniform_color([1.0, 0.0, 0.0])

    o3d.visualization.draw_geometries([reconstructed_mesh, original_pcd],
                                      window_name="Neural CAD Reconstruction")


def train_pipeline(file_path, act_type=ActivationType.SIREN):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing network with {act_type.name} activation...")
    model = DevelopableSDF(activation_type=act_type).to(device)

    # SIREN generally requires a slightly lower learning rate to stay stable
    lr = 5e-5 if act_type == ActivationType.SIREN else 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)

    surface_points_tensor, original_pcd = load_and_sample_mesh(file_path,
                                                               num_points=10000)
    surface_points = surface_points_tensor.to(device)

    original_pcd.points = o3d.utility.Vector3dVector(surface_points_tensor.numpy())

    epochs = 10000
    batch_size_vol = 5000

    print("Starting optimization on loaded shape...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 1. Manifold Phase
        p_surface = surface_points.clone().requires_grad_(True)
        sdf_surface = model(p_surface)
        loss_manifold = torch.mean(torch.abs(sdf_surface))

        # 2. Global Volumetric Phase
        p_volume = (torch.rand(batch_size_vol, 3, device=device) * 2.0 - 1.0)
        p_volume.requires_grad_(True)

        S_volume, grad_volume, sdf_volume = fast_double_trace_surrogate(model, p_volume)

        # Eikonal condition
        loss_eikonal = torch.mean((torch.norm(grad_volume, dim=-1) - 1.0) ** 2)

        # Volumetric Developability Loss (Curvature)
        # We skip this entirely if ReLU is used, as the second derivative is identically 0
        if act_type == ActivationType.RELU:
            loss_developable = torch.tensor(0.0, device=device)
        else:
            loss_developable = generalized_charbonnier_loss(S_volume)

        # Dirichlet Non-Manifold penalty
        loss_dnm = non_manifold_penalty(sdf_volume, alpha_decay=50.0)

        # Combine losses
        total_loss = loss_manifold + (0.1 * loss_eikonal) + (
                    0.01 * loss_developable) + (0.05 * loss_dnm)

        total_loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:<4} | Total: {total_loss.item():.4f} | Eikonal: {loss_eikonal.item():.4f} | Dev: {loss_developable.item():.4f} | DNM: {loss_dnm.item():.4f}")

    base_name, ext = os.path.splitext(file_path)
    output_mesh_path = f"{base_name}_reconstructed_{act_type.name.lower()}.obj"

    extract_and_visualize_mesh(model, device, original_pcd,
                               output_path=output_mesh_path, resolution=170)


if __name__ == "__main__":
    obj_path = "/Users/mikolajkida/Documents/github/implicit_neural_representation/data/00808652_a5311ecc077bfdd2cc3d9aa7_trimesh_000.obj"

    # You can now easily swap out the activation here!
    train_pipeline(obj_path, act_type=ActivationType.SIREN)