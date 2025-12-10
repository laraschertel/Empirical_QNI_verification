from pathlib import Path
import numpy as np
import torch
import json

from torch.utils.data import DataLoader, Subset

from config import TrainConfig, ExperimentConfig
from datasets import get_mnist_dataloaders
from models import MNISTMLP
from training import train_standard, train_dp
from stats import softmax, ks_test, welch_ttest, total_variation_distance, empirical_epsilon


def run_membership_experiment(train_cfg, exp_cfg):
    """Run membership experiment on MNIST (with vs without one sample)."""
    train_loader_full, test_loader = get_mnist_dataloaders(batch_size=train_cfg.batch_size)
    train_dataset = train_loader_full.dataset

    indices = list(range(len(train_dataset)))
    indices_with = indices
    indices_without = indices[1:]  # remove member_idx=0

    ds_with = Subset(train_dataset, indices_with)
    ds_without = Subset(train_dataset, indices_without)

    loader_with = DataLoader(ds_with, batch_size=train_cfg.batch_size, shuffle=True)
    loader_without = DataLoader(ds_without, batch_size=train_cfg.batch_size, shuffle=True)

    model_with = MNISTMLP()
    model_without = MNISTMLP()

    if train_cfg.dp:
        model_with = train_dp(model_with, _wrap_mnist_loader(loader_with), train_cfg)
        model_without = train_dp(model_without, _wrap_mnist_loader(loader_without), train_cfg)
    else:
        model_with = train_standard(model_with, _wrap_mnist_loader(loader_with), train_cfg)
        model_without = train_standard(model_without, _wrap_mnist_loader(loader_without), train_cfg)

    device = train_cfg.device
    model_with.eval().to(device)
    model_without.eval().to(device)

    probs_with = []
    probs_without = []

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            logits_with = model_with(x).cpu().numpy()
            logits_without = model_without(x).cpu().numpy()

            p_with = softmax(logits_with)
            p_without = softmax(logits_without)

            probs_with.append(p_with)
            probs_without.append(p_without)

    probs_with = np.vstack(probs_with)
    probs_without = np.vstack(probs_without)

    conf_with = probs_with.max(axis=1)
    conf_without = probs_without.max(axis=1)

    ks_stat, ks_p = ks_test(conf_with, conf_without)
    t_stat, t_p = welch_ttest(conf_with, conf_without)
    tv = total_variation_distance(probs_with, probs_without)
    eps_hat = empirical_epsilon(probs_with, probs_without)

    Path(exp_cfg.out_dir).mkdir(parents=True, exist_ok=True)
    base_name = f"membership_mnist_dp{train_cfg.dp}"
    out_txt = Path(exp_cfg.out_dir) / f"{base_name}.txt"
    out_json = Path(exp_cfg.out_dir) / f"{base_name}.json"

    with out_txt.open("w") as f:
        f.write(f"KS stat: {ks_stat:.4f}, p-value: {ks_p:.4e}\n")
        f.write(f"Welch t stat: {t_stat:.4f}, p-value: {t_p:.4e}\n")
        f.write(f"Total variation distance: {tv:.4f}\n")
        f.write(f"Empirical epsilon: {eps_hat:.4f}\n")

    results = {
        "ks_stat": float(ks_stat),
        "ks_p_value": float(ks_p),
        "welch_t_stat": float(t_stat),
        "welch_t_p_value": float(t_p),
        "total_variation_distance": float(tv),
        "empirical_epsilon": float(eps_hat),
    }

    with out_json.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Results written to {out_txt} and {out_json}")


def _wrap_mnist_loader(loader):
    """Wrap MNIST (x, y) batches into dicts {x, y} for reuse of training code."""
    from torch.utils.data import DataLoader

    class WrapperDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            x, y = self.dataset[idx]
            return {"x": x.view(-1), "y": y}

    wrapped_ds = WrapperDataset(loader.dataset)
    return DataLoader(wrapped_ds, batch_size=loader.batch_size, shuffle=True)
