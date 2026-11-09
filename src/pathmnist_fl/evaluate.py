import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate(model, loader, device):
    model.to(device)
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.reshape(-1).long().to(device)
        logits = model(images)
        loss_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
        correct += int((logits.argmax(1) == labels).sum())
        total += int(labels.numel())
    if not total:
        return {"accuracy": float("nan"), "loss": float("nan"), "n": 0}
    return {"accuracy": correct / total, "loss": loss_sum / total, "n": total}
