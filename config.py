from dataclasses import dataclass
from typing import Literal


@dataclass
class TrainConfig:
    """Training configuration for all experiments."""
    model_type: Literal["logreg", "mlp"] = "logreg"
    batch_size: int = 128
    lr: float = 1e-3
    num_epochs: int = 5
    device: str = "cuda"
    dp: bool = True
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.0  
    seed: int = 0
    save_model: bool = True          
    use_pretrained: bool = True


@dataclass
class ExperimentConfig:
    """Experiment setup."""
    task: Literal["fairness", "membership"] = "fairness"
    dataset: Literal["adult", "compas", "mnist"] = "compas"
    protected_attr: Literal["gender", "race"] = "gender"
    out_dir: str = "results"
    model_dir: str = "models"
