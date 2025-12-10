from torch.utils.data import  Subset


def make_fairness_adjacent_batch(
    batch,
    protected_col_idx,
    num_protected_classes=None,
):
    """Create original and adjacent batches by rotating the protected attribute."""
    x = batch["x"]
    y = batch["y"]
    p = batch["protected"]

    num_protected_classes = int(p.max().item()) + 1

    x_orig = x.clone()
    p_orig = p.clone()

    orig_batch = {
        "x": x_orig,
        "y": y,
        "protected": p_orig,
    }

    p_adj = (p_orig + 1) % num_protected_classes

    x_adj = x_orig.clone()
    x_adj[:, protected_col_idx] = p_adj.to(x_adj.dtype)

    adj_batch = {
        "x": x_adj,
        "y": y,
        "protected": p_adj,
    }

    return orig_batch, adj_batch


def split_for_membership(dataset, idx_member):
    """Return two datasets that differ only by one removed record."""
    indices = list(range(len(dataset)))
    indices_with = indices.copy()
    indices_without = indices.copy()
    indices_without.remove(idx_member)

    ds_with = Subset(dataset, indices_with)
    ds_without = Subset(dataset, indices_without)
    return ds_with, ds_without
