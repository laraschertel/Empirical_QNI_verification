import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

from config import TrainConfig
from dp_sgd import dp_sgd_step, SimpleGaussianAccountant


def make_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_standard(model, train_loader, cfg):
    device = cfg.device
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.num_epochs):
        pbar = tqdm(train_loader, desc=f"[Standard] Epoch {epoch+1}/{cfg.num_epochs}")
        for batch in pbar:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())
    return model


def train_dp(model, train_loader, train_cfg):
    device = train_cfg.device
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train_cfg.lr,
        momentum=0.0,
    )

    N = len(train_loader.dataset)
    delta = 1.0 / (N ** 1.1)
    accountant = SimpleGaussianAccountant(
        noise_multiplier=train_cfg.noise_multiplier,
        delta=delta,
    )

    loss_fn = lambda logits, labels: torch.nn.functional.cross_entropy(
        logits, labels, reduction="none"
    )

    for epoch in range(train_cfg.num_epochs):
        pbar = tqdm(train_loader, desc=f"[DP-SGD] Epoch {epoch+1}/{train_cfg.num_epochs}")
        last_loss = None

        for batch in pbar:
            x = batch["x"]
            y = batch["y"]

            batch_loss = dp_sgd_step(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                x=x,
                y=y,
                max_grad_norm=train_cfg.max_grad_norm,
                noise_multiplier=train_cfg.noise_multiplier,
                accountant=accountant,
            )
            last_loss = batch_loss
            pbar.set_postfix(loss=batch_loss)

        eps = accountant.get_epsilon()
        print(f"[DP-SGD] Epoch {epoch+1}: eps ≈ {eps:.3f}, last batch loss = {last_loss:.4f}")

    return model


@torch.no_grad()
def collect_outputs(model, loader, device="cpu"):
    """Collect logits and labels over a whole loader."""
    model.eval()
    model.to(device)
    all_logits = []
    all_y = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_y.append(y.cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_y, dim=0)
