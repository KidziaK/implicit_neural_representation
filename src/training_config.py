from dataclasses import dataclass

@dataclass
class VisualizationConfig:
    every: int = 100
    resolution: int = 64
    visualize: bool = True

@dataclass
class TrainingConfig:
    epochs: int
    use_projection: bool = False
    proj_every: int = 1
    proj_eps: float = 1e-6
    proj_cap_n: int = 128