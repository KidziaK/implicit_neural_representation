import matplotlib.pyplot as plt
from torch import Tensor

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