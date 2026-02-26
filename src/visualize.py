import torch

from typing import Any

import matplotlib.pyplot as plt
from torch import Tensor




def show(obj: Any) -> None:
    from src.experiment import Experiment

    match obj:
        case Tensor():
            plot_2d_sdf(obj)
        case Experiment():
            res = obj.visualization_config.resolution
            a = 1.
            interval = torch.linspace(-a, a, res)

            in_dims = obj.model.in_dims
            if in_dims == 2:
                grid = torch.stack(torch.meshgrid(interval, interval, indexing='ij'), dim=-1)
                sdf = obj.evaluate(grid)
                show(sdf)
            elif in_dims == 3:
                # Create a 3D grid [res, res, res, 3]
                grid_3d = torch.stack(torch.meshgrid(interval, interval, interval, indexing='ij'), dim=-1)
                
                # Evaluate the model on the entire 3-dimensional volume
                with torch.no_grad():
                    sdf_vol = obj.evaluate(grid_3d.view(-1, 3)).view(res, res, res)

                # Convert to numpy for skimage
                volume_np = sdf_vol.detach().cpu().numpy()

                import skimage.measure
                import open3d as o3d
                import numpy as np
                
                try:
                    # Isolate the zero level set. If the network hasn't learned the shape perfectly yet,
                    # level=0 might fail, so we catch the ValueError and print a message instead of crashing.
                    verts, faces, normals, values = skimage.measure.marching_cubes(volume_np, level=0.0)
                    
                    # Verts initially come back in raw voxel array indices [0, res-1]. 
                    # We normalize them back explicitly to the [-1.0, 1.0] world grid.
                    verts = (verts / (res - 1)) * 2.0 - 1.0

                    mesh = o3d.geometry.TriangleMesh()
                    mesh.vertices = o3d.utility.Vector3dVector(verts)
                    mesh.triangles = o3d.utility.Vector3iVector(faces)
                    
                    # Compute normals natively via Open3D to ensure the renderer shades it properly
                    mesh.compute_triangle_normals()
                    mesh.compute_vertex_normals()

                    print("Visualizing 3D Reconstructed Mesh (Close the window to resume training)...")
                    o3d.visualization.draw_geometries([mesh], window_name="SDF Surface Reconstruction")
                except ValueError as e:
                    print(f"Marching cubes cannot extract surface yet (no level=0.0 found). Error: {e}")
            else:
                raise NotImplementedError(f"Visualization for in_dims={in_dims} not implemented.")
        case _:
            raise NotImplementedError

def plot_2d_sdf(sdf: Tensor) -> None:
    grid_np = sdf.detach().cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_np, origin='lower', extent=[-1, 1, -1, 1], cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(label='SDF Value')
    plt.contour(grid_np, levels=[0], colors='black', origin='lower', extent=[-1, 1, -1, 1], linewidths=2)
    plt.title('2D SDF Contour')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.show()
