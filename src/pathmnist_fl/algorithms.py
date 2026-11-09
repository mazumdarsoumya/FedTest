from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils import clone_state


@dataclass
class ClientResult:
    client_id: int
    state: dict[str, torch.Tensor]
    train_loss: float
    num_examples: int
    update_norm: float


def state_delta_norm(local: dict[str, torch.Tensor], global_state: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for key, value in local.items():
        diff = value.detach().float().cpu() - global_state[key].detach().float().cpu()
        total += float(torch.sum(diff * diff))
    return math.sqrt(total)


def train_client(model: torch.nn.Module, global_state: dict[str, torch.Tensor], loader: DataLoader,
                 client_id: int, device: torch.device, epochs: int, lr: float,
                 momentum: float, weight_decay: float, mu: float) -> ClientResult:
    model.load_state_dict(global_state)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay)
    # Snapshot trainable parameters for the FedProx proximal term.
    anchor = {name: p.detach().clone() for name, p in model.named_parameters()}
    total_loss, seen = 0.0, 0
    model.train()
    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.reshape(-1).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if mu:
                prox = sum(torch.sum((p - anchor[name]) ** 2)
                           for name, p in model.named_parameters())
                loss = loss + (mu / 2.0) * prox
            loss.backward()
            optimizer.step()
            batch = int(labels.numel())
            total_loss += float(loss.detach()) * batch
            seen += batch
    state = clone_state(model.state_dict())
    norm = state_delta_norm(state, global_state)
    return ClientResult(client_id, state, total_loss / max(seen, 1), len(loader.dataset), norm)


def fedavg(results: list[ClientResult], global_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    total = sum(result.num_examples for result in results)
    if total <= 0:
        raise ValueError("Cannot aggregate an empty set of clients")
    aggregated = {}
    for key, base in global_state.items():
        # Integer buffers (e.g. BatchNorm counters) cannot be averaged as floats.
        if not torch.is_floating_point(base):
            aggregated[key] = results[0].state[key].clone()
            continue
        value = torch.zeros_like(base)
        for result in results:
            value += result.state[key].to(value.device) * (result.num_examples / total)
        aggregated[key] = value
    return aggregated


def update_cosines(results: list[ClientResult], global_state: dict[str, torch.Tensor],
                   aggregated_state: dict[str, torch.Tensor]) -> dict[int, float]:
    keys = [k for k, v in global_state.items() if torch.is_floating_point(v)]
    aggregate_delta = torch.cat([(aggregated_state[k] - global_state[k]).detach().float().cpu().reshape(-1)
                                 for k in keys])
    out = {}
    for result in results:
        client_delta = torch.cat([(result.state[k] - global_state[k]).detach().float().cpu().reshape(-1)
                                  for k in keys])
        denom = torch.linalg.vector_norm(client_delta) * torch.linalg.vector_norm(aggregate_delta)
        out[result.client_id] = float(torch.dot(client_delta, aggregate_delta) / denom) if denom > 0 else 0.0
    return out
