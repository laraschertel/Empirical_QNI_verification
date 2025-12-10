import math

import torch


class SimpleGaussianAccountant:
    """Pessimistic Gaussian-Mechanism privacy accountant for DP-SGD."""
    def __init__(self, noise_multiplier, delta):
        self.noise_multiplier = noise_multiplier
        self.delta = delta
        self.steps = 0

    def step(self):
        self.steps += 1

    def get_epsilon(self):
        if self.noise_multiplier == 0.0 or self.steps == 0:
            return float("inf")
        return math.sqrt(
            2.0 * self.steps * math.log(1.25 / self.delta)
        ) / self.noise_multiplier


def dp_sgd_step(
    model,
    optimizer,
    loss_fn,
    x,
    y,
    max_grad_norm,
    noise_multiplier,
    accountant=None,
):
    """One DP-SGD update on a single batch (per-example grads, clipping, noise)."""
    device = next(model.parameters()).device
    x = x.to(device)
    y = y.to(device)

    batch_size = x.size(0)

    model.train()
    logits = model(x)
    losses = loss_fn(logits, y)
    if losses.dim() != 1 or losses.size(0) != batch_size:
        raise ValueError("loss_fn must return per-example losses of shape [B].")

    # per-example gradients
    per_param_grads = []
    for p in model.parameters():
        if p.requires_grad:
            per_param_grads.append(
                torch.zeros((batch_size,) + p.shape, device=device, dtype=p.dtype)
            )
        else:
            per_param_grads.append(None)

    for i in range(batch_size):
        optimizer.zero_grad()
        losses[i].backward(retain_graph=True)
        for idx, p in enumerate(model.parameters()):
            if p.requires_grad and p.grad is not None:
                per_param_grads[idx][i].copy_(p.grad.detach())

    optimizer.zero_grad()
    model.zero_grad()

    # per-example norms
    grad_norms = torch.zeros(batch_size, device=device)
    for g in per_param_grads:
        if g is None:
            continue
        grad_norms += g.view(batch_size, -1).pow(2).sum(dim=1)
    grad_norms = grad_norms.sqrt()

    clip_factors = (max_grad_norm / (grad_norms + 1e-6)).clamp(max=1.0)

    # clipping, noise, and update
    for p, g in zip(model.parameters(), per_param_grads):
        if g is None or not p.requires_grad:
            continue

        cf = clip_factors.view((batch_size,) + (1,) * (g.dim() - 1))
        g_clipped = g * cf

        grad_mean = g_clipped.mean(dim=0)

        std = noise_multiplier * max_grad_norm / batch_size
        noise = torch.randn_like(grad_mean) * std

        p.grad = grad_mean + noise

    optimizer.step()

    if accountant is not None:
        accountant.step()

    return losses.mean().item()
