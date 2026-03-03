import torch

from typing import Any

import matplotlib.pyplot as plt
from torch import Tensor




import os

def extract_mesh(obj: Any) -> Any:
    """Extracts open3d TriangleMesh from 3D SDF Experiment."""
    import skimage.measure
    import open3d as o3d
    import numpy as np
    
    res = obj.visualization_config.resolution
    a = 1.
    interval = torch.linspace(-a, a, res)
    grid_3d = torch.stack(torch.meshgrid(interval, interval, interval, indexing='ij'), dim=-1)
    
    with torch.no_grad():
        sdf_vol = obj.evaluate(grid_3d.view(-1, 3)).view(res, res, res)

    volume_np = sdf_vol.detach().cpu().numpy()
    
    # Isolate the zero level set.
    verts, faces, normals, values = skimage.measure.marching_cubes(volume_np, level=0.0)
    
    # Normalize back explicitly to the [-1.0, 1.0] world grid.
    verts = (verts / (res - 1)) * 2.0 - 1.0

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    
    return mesh


def show(obj: Any, epoch: int = 0) -> None:
    from src.experiment import Experiment

    os.makedirs("visualization", exist_ok=True)

    match obj:
        case Tensor():
            plot_2d_sdf(obj, epoch)
        case Experiment():
            res = obj.visualization_config.resolution
            a = 1.
            interval = torch.linspace(-a, a, res)

            in_dims = obj.model.in_dims
            if in_dims == 2:
                grid = torch.stack(torch.meshgrid(interval, interval, indexing='ij'), dim=-1)
                sdf = obj.evaluate(grid)
                show(sdf, epoch)
            elif in_dims == 3:
                import open3d as o3d
                import numpy as np

                try:
                    mesh = extract_mesh(obj)
                except ValueError as e:
                    print(f"Marching cubes cannot extract surface yet (no level=0.0 found). Error: {e}")
                    return
                
                # Headless/off-screen rendering trick for Open3D
                vis = o3d.visualization.Visualizer()
                vis.create_window(visible=False, window_name="SDF Surface Reconstruction")
                vis.add_geometry(mesh)
                
                # Update viewing angles
                vis.poll_events()
                vis.update_renderer()
                
                # Use Open3D's native screen capture directly to file
                vis.capture_screen_image(f"visualization/mesh_epoch_{epoch}.png", do_render=True)
                
                vis.destroy_window()
                print(f"Saved 3D visualization: visualization/mesh_epoch_{epoch}.png")
            else:
                raise NotImplementedError(f"Visualization for in_dims={in_dims} not implemented.")
        case _:
            raise NotImplementedError

def plot_2d_sdf(sdf: Tensor, epoch: int = 0) -> None:
    grid_np = sdf.detach().cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_np, origin='lower', extent=[-1, 1, -1, 1], cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(label='SDF Value')
    plt.contour(grid_np, levels=[0], colors='black', origin='lower', extent=[-1, 1, -1, 1], linewidths=2)
    plt.title(f'2D SDF Contour - Epoch {epoch}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.show()
