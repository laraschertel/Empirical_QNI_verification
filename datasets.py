import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import json

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import kagglehub


DATA_DIR = Path("./data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


class TabularDataset(Dataset):
    """Tabular dataset with features, labels, and protected attribute."""
    def __init__(self, X, y, protected):
        assert len(X) == len(y) == len(protected)
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.protected = torch.from_numpy(protected).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "x": self.X[idx],
            "y": self.y[idx],
            "protected": self.protected[idx],
        }


def _get_or_create_adult_local_path():
    """Return local path to Adult CSV, downloading if needed."""
    local_path = DATA_DIR / "adult.csv"
    if local_path.exists():
        print("Using cached Adult CSV:", local_path)
        return str(local_path)

    remote_dir = kagglehub.dataset_download("wenruliu/adult-income-dataset")
    print("Downloaded Adult dataset to:", remote_dir)

    for fname in os.listdir(remote_dir):
        if fname.lower().endswith(".csv"):
            src = Path(remote_dir) / fname
            shutil.copy(src, local_path)
            print(f"Copied {src} -> {local_path}")
            return str(local_path)

    raise FileNotFoundError(f"No CSV found in Adult Kaggle directory: {remote_dir}")


def _get_or_create_compas_local_path():
    """Return local path to COMPAS CSV, downloading if needed."""
    local_path = DATA_DIR / "compas.csv"
    if local_path.exists():
        print("Using cached COMPAS CSV:", local_path)
        return str(local_path)

    remote_dir = kagglehub.dataset_download("danofer/compass")
    print("Downloaded COMPAS dataset to:", remote_dir)

    for fname in os.listdir(remote_dir):
        if fname.lower().endswith(".csv"):
            src = Path(remote_dir) / fname
            shutil.copy(src, local_path)
            print(f"Copied {src} -> {local_path}")
            return str(local_path)

    raise FileNotFoundError(f"No CSV found in COMPAS Kaggle directory: {remote_dir}")


def _encode_adult_protected(df, protected_attr):
    """Encode Adult protected attribute as integer codes."""
    if protected_attr == "gender":
        col = "gender"
    elif protected_attr == "race":
        col = "race"
    else:
        raise ValueError(f"Unsupported protected_attr for Adult: {protected_attr}")

    cat = df[col].astype(str).str.strip().astype("category")
    codes = cat.cat.codes.to_numpy()
    categories = np.array(cat.cat.categories.tolist())
    return codes, categories


def load_adult(protected_attr="gender"):
    """Load Adult dataset with protected attribute and train/test split."""
    csv_path = _get_or_create_adult_local_path()
    df = pd.read_csv(csv_path)

    df = df.replace("?", np.nan).dropna()

    df["label"] = (df["income"].astype(str).str.contains(">50K")).astype(int)

    protected_vec, protected_categories = _encode_adult_protected(df, protected_attr)

    label_col = "label"

    if protected_attr == "gender":
        prot_col = "gender"
    else:
        prot_col = "race"

    drop_cols_for_features = {label_col, "income", prot_col}
    feature_df = df.drop(columns=[c for c in drop_cols_for_features if c in df.columns])

    numeric_cols = feature_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]

    if categorical_cols:
        dummies = pd.get_dummies(feature_df[categorical_cols], drop_first=False)
        feature_numeric = pd.concat(
            [feature_df[numeric_cols].reset_index(drop=True),
             dummies.reset_index(drop=True)],
            axis=1,
        )
    else:
        feature_numeric = feature_df[numeric_cols].copy()

    X_base = feature_numeric.values.astype(np.float32)

    protected_vec_float = protected_vec.astype(np.float32).reshape(-1, 1)
    X = np.hstack([X_base, protected_vec_float])

    y = df[label_col].values.astype(np.int64)

    n = len(df)
    n_train = int(0.8 * n)
    idx = np.arange(n)
    np.random.shuffle(idx)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    p_train, p_test = protected_vec[train_idx], protected_vec[test_idx]

    train_ds = TabularDataset(X_train, y_train, p_train)
    test_ds = TabularDataset(X_test, y_test, p_test)

    K = int(protected_vec.max()) + 1

    protected_label_map = {int(i): str(cat) for i, cat in enumerate(protected_categories)}

    meta = {
        "input_dim": X.shape[1],
        "num_classes": int(df[label_col].nunique()),
        "protected_values": np.arange(K, dtype=int),
        "protected_categories": protected_categories,
        "protected_num_classes": K,
        "protected_col_idx": X.shape[1] - 1,
        "protected_attr_name": protected_attr,
        "protected_label_map": protected_label_map,
    }

    save_protected_encoding(meta, dataset_name="adult", out_dir=DATA_DIR)

    return train_ds, test_ds, meta


def _encode_compas_protected(df, protected_attr):
    """Encode COMPAS protected attribute as integer codes."""
    if protected_attr == "gender":
        col = "Sex_Code_Text"
    elif protected_attr == "race":
        col = "Ethnic_Code_Text"
    else:
        raise ValueError(f"Unsupported protected_attr for COMPAS: {protected_attr}")

    cat = df[col].astype(str).str.strip().astype("category")
    codes = cat.cat.codes.to_numpy()
    categories = np.array(cat.cat.categories.tolist())
    return codes, categories


def load_compas(protected_attr="race"):
    """Load COMPAS dataset with protected attribute and train/test split."""
    csv_path = _get_or_create_compas_local_path()
    df = pd.read_csv(csv_path)

    needed_cols = ["DecileScore", "Sex_Code_Text", "Ethnic_Code_Text"]
    df = df.dropna(subset=[c for c in needed_cols if c in df.columns])

    df["label"] = (df["DecileScore"].astype(float) >= 5).astype(int)

    protected_vec, protected_categories = _encode_compas_protected(df, protected_attr)

    label_col = "label"

    if protected_attr == "gender":
        prot_col = "Sex_Code_Text"
    elif protected_attr == "race":
        prot_col = "Ethnic_Code_Text"

    drop_cols_for_features = {label_col, prot_col}
    feature_df = df.drop(columns=[c for c in drop_cols_for_features if c in df.columns])

    numeric_cols = feature_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]

    if categorical_cols:
        dummies = pd.get_dummies(feature_df[categorical_cols], drop_first=False)
        feature_numeric = pd.concat(
            [feature_df[numeric_cols].reset_index(drop=True),
             dummies.reset_index(drop=True)],
            axis=1,
        )
    else:
        feature_numeric = feature_df[numeric_cols].copy()

    X_base = feature_numeric.values.astype(np.float32)
    protected_vec_float = protected_vec.astype(np.float32).reshape(-1, 1)
    X = np.hstack([X_base, protected_vec_float])

    y = df[label_col].values.astype(np.int64)

    n = len(df)
    n_train = int(0.8 * n)
    idx = np.arange(n)
    np.random.shuffle(idx)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    p_train, p_test = protected_vec[train_idx], protected_vec[test_idx]

    train_ds = TabularDataset(X_train, y_train, p_train)
    test_ds = TabularDataset(X_test, y_test, p_test)

    K = int(protected_vec.max()) + 1

    protected_label_map = {int(i): str(cat) for i, cat in enumerate(protected_categories)}

    meta = {
        "input_dim": X.shape[1],
        "num_classes": int(df[label_col].nunique()),
        "protected_values": np.arange(K, dtype=int),
        "protected_categories": protected_categories,
        "protected_num_classes": K,
        "protected_col_idx": X.shape[1] - 1,
        "protected_attr_name": protected_attr,
        "protected_label_map": protected_label_map,
    }

    save_protected_encoding(meta, dataset_name="compas", out_dir=DATA_DIR)

    return train_ds, test_ds, meta


def get_mnist_dataloaders(batch_size=128):
    """Return MNIST train and test dataloaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def make_dataloader(dataset, batch_size, shuffle=True):
    """Create a DataLoader for a given dataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def save_protected_encoding(meta, dataset_name, out_dir=DATA_DIR):
    """Save protected attribute encoding metadata to JSON."""
    protected_attr = meta.get("protected_attr_name", "protected")
    protected_values = meta.get("protected_values")
    protected_categories = meta.get("protected_categories")
    protected_label_map = meta.get("protected_label_map", None)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset_name}_{protected_attr}_encoding.json"

    enc = {
        "dataset": dataset_name,
        "protected_attr": protected_attr,
        "protected_values": (
            protected_values.tolist() if protected_values is not None else None
        ),
        "protected_categories": (
            [str(c) for c in protected_categories] if protected_categories is not None else None
        ),
        "protected_label_map": (
            {str(int(k)): str(v) for k, v in protected_label_map.items()}
            if protected_label_map is not None
            else None
        ),
    }

    with out_path.open("w") as f:
        json.dump(enc, f, indent=2)

    print(f"Saved protected encoding to {out_path}")
    return out_path
