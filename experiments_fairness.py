from pathlib import Path
from dataclasses import asdict
import json

import numpy as np
import torch

from config import TrainConfig, ExperimentConfig
from datasets import load_adult, load_compas, make_dataloader
from models import build_model
from training import train_standard, train_dp
from adjacency import make_fairness_adjacent_batch
from stats import (
    softmax,
    ks_test,
    welch_ttest,
    total_variation_distance,
    empirical_epsilon,
    accuracy_from_probs,
)


def _get_model_path(train_cfg, exp_cfg):
    """Canonical path for saving/loading fairness models."""
    model_dir = Path(exp_cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{exp_cfg.dataset}_{exp_cfg.protected_attr}_{train_cfg.model_type}_dp{int(train_cfg.dp)}.pt"
    return model_dir / fname


def _save_model_checkpoint(model, meta, train_cfg, exp_cfg):
    """Save model state and configs."""
    ckpt = {
        "state_dict": model.state_dict(),
        "train_cfg": asdict(train_cfg),
        "exp_cfg": asdict(exp_cfg),
        "meta": meta,
    }
    path = _get_model_path(train_cfg, exp_cfg)
    torch.save(ckpt, path)
    print(f"Saved model checkpoint to {path}")
    return path


def _load_model_checkpoint(train_cfg, exp_cfg, meta_from_loader):
    """Load model checkpoint (uses meta from checkpoint if present)."""
    path = _get_model_path(train_cfg, exp_cfg)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found at {path}")

    ckpt = torch.load(path, map_location=train_cfg.device, weights_only=False)

    meta_ckpt = ckpt.get("meta", meta_from_loader)
    input_dim = meta_ckpt["input_dim"]
    num_classes = meta_ckpt["num_classes"]

    model = build_model(train_cfg.model_type, input_dim, num_classes)
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded model from {path}")
    return model


def run_fairness_experiment(train_cfg, exp_cfg):
    if exp_cfg.dataset == "adult":
        train_ds, test_ds, meta = load_adult(
            protected_attr=exp_cfg.protected_attr,
        )
    elif exp_cfg.dataset == "compas":
        train_ds, test_ds, meta = load_compas(
            protected_attr=exp_cfg.protected_attr,
        )
    else:
        raise ValueError("Fairness experiments use Adult or COMPAS.")

    input_dim = meta["input_dim"]
    num_classes = meta["num_classes"]
    protected_col_idx = meta["protected_col_idx"]
    protected_values = meta["protected_values"]

    train_loader = make_dataloader(train_ds, batch_size=train_cfg.batch_size)
    test_loader = make_dataloader(test_ds, batch_size=train_cfg.batch_size, shuffle=False)

    if train_cfg.use_pretrained:
        model = _load_model_checkpoint(train_cfg, exp_cfg, meta)
    else:
        model = build_model(train_cfg.model_type, input_dim, num_classes)
        if train_cfg.dp:
            model = train_dp(model, train_loader, train_cfg)
        else:
            model = train_standard(model, train_loader, train_cfg)

        if train_cfg.save_model:
            _save_model_checkpoint(model, meta, train_cfg, exp_cfg)

    device = train_cfg.device
    model.eval()
    model.to(device)

    probs_orig_all = []
    probs_adj_all = []
    labels_all = []
    protected_all = []

    with torch.no_grad():
        for batch in test_loader:
            orig_batch, adj_batch = make_fairness_adjacent_batch(batch, protected_col_idx)

            x_orig = orig_batch["x"].to(device)
            x_adj = adj_batch["x"].to(device)

            logits_orig = model(x_orig).cpu().numpy()
            logits_adj = model(x_adj).cpu().numpy()

            p_orig = softmax(logits_orig)
            p_adj = softmax(logits_adj)

            probs_orig_all.append(p_orig)
            probs_adj_all.append(p_adj)

            y_np = batch["y"].numpy()
            p_np = batch["protected"].numpy()

            labels_all.append(y_np)
            protected_all.append(p_np)

    probs_orig = np.vstack(probs_orig_all)
    probs_adj = np.vstack(probs_adj_all)
    y_true = np.concatenate(labels_all)

    overall_acc = accuracy_from_probs(probs_orig, y_true)

    pos_orig = probs_orig[:, 1] if num_classes > 1 else probs_orig[:, 0]
    pos_adj = probs_adj[:, 1] if num_classes > 1 else probs_adj[:, 0]

    ks_stat, ks_p = ks_test(pos_orig, pos_adj)
    t_stat, t_p = welch_ttest(pos_orig, pos_adj)
    tv = total_variation_distance(probs_orig, probs_adj)
    eps_hat = empirical_epsilon(probs_orig, probs_adj)

    Path(exp_cfg.out_dir).mkdir(parents=True, exist_ok=True)
    base_name = f"fairness_{exp_cfg.dataset}_{exp_cfg.protected_attr}_dp{train_cfg.dp}"
    out_txt = Path(exp_cfg.out_dir) / (base_name + ".txt")
    out_json = Path(exp_cfg.out_dir) / (base_name + ".json")

    with out_txt.open("w") as f:
        f.write(f"Dataset: {exp_cfg.dataset}, protected_attr: {exp_cfg.protected_attr}\n\n")
        f.write("=== Global Metrics (all groups pooled) ===\n")
        f.write(f"Accuracy overall: {overall_acc:.4f}\n")
        f.write(f"KS stat: {ks_stat:.4f}, p-value: {ks_p:.4e}\n")
        f.write(f"Welch t stat: {t_stat:.4f}, p-value: {t_p:.4e}\n")
        f.write(f"Total variation distance: {tv:.4f}\n")
        f.write(f"Empirical epsilon (naive): {eps_hat:.4f}\n\n")

    result_json = {
        "dataset": exp_cfg.dataset,
        "protected_attr": exp_cfg.protected_attr,
        "dp": bool(train_cfg.dp),
        "model_type": train_cfg.model_type,
        "global_metrics": {
            "accuracy_overall": float(overall_acc),
            "ks_stat": float(ks_stat),
            "ks_pvalue": float(ks_p),
            "welch_t_stat": float(t_stat),
            "welch_t_pvalue": float(t_p),
            "tv_distance": float(tv),
            "epsilon_empirical": float(eps_hat),
        },
    }

    with out_json.open("w") as jf:
        json.dump(result_json, jf, indent=2)

    print(f"Results written to {out_txt} and {out_json}")
