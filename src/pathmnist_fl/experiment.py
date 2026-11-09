from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path

import torch
import yaml

from .algorithms import fedavg, train_client, update_cosines
from .data import client_summaries, load_federated_data, make_loader
from .evaluate import evaluate
from .model import PathMNISTCNN
from .utils import clone_state, jsonable, seed_everything


def run_one(method: str, cfg: dict, data, initial_state, seed: int) -> list[dict]:
    seed_everything(seed)
    device_name = cfg["device"]
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available()
                          else "cpu" if device_name == "auto" else device_name)
    model = PathMNISTCNN(num_classes=9).to(device)
    global_state = clone_state(initial_state)
    model.load_state_dict(global_state)
    is_prox = method == "fedprox"
    mu = float(cfg["mu"]) if is_prox else 0.0
    run_name = f"fedprox_mu{mu:g}_seed{seed}" if is_prox else f"fedavg_seed{seed}"
    tracker = None
    if cfg.get("wandb", {}).get("enabled", False):
        try:
            import wandb
            tracker = wandb.init(project=cfg["wandb"]["project"], entity=cfg["wandb"].get("entity"),
                                 name=run_name, config={**cfg, "algorithm": method, "seed": seed,
                                                        "mu": mu, "device_resolved": str(device)},
                                 reinit=True)
        except Exception as exc:
            raise RuntimeError("W&B initialization failed; authenticate or disable W&B") from exc

    rows = []
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for round_idx in range(1, int(cfg["rounds"]) + 1):
        client_results = []
        for client_id, indices in enumerate(data.client_indices):
            loader = make_loader(data.train_dataset, indices, data.noisy_targets,
                                 int(cfg["batch_size"]), True,
                                 seed + round_idx * 1009 + client_id * 97,
                                 int(cfg.get("num_workers", 0)))
            local_model = PathMNISTCNN(num_classes=9)
            result = train_client(local_model, global_state, loader, client_id, device,
                                  int(cfg["local_epochs"]), float(cfg["learning_rate"]),
                                  float(cfg["momentum"]), float(cfg["weight_decay"]), mu)
            client_results.append(result)
        next_state = fedavg(client_results, global_state)
        cosine = update_cosines(client_results, global_state, next_state)
        model.load_state_dict(next_state)
        global_state = clone_state(next_state)

        test_loader = make_loader(data.test_dataset, list(range(len(data.test_dataset))), None,
                                  int(cfg["batch_size"]), False, seed)
        global_metrics = evaluate(model, test_loader, device)
        row = {"algorithm": method, "seed": seed, "round": round_idx,
               "global_test_accuracy": global_metrics["accuracy"],
               "global_test_loss": global_metrics["loss"],
               "training_loss": sum(r.train_loss * r.num_examples for r in client_results) /
                                sum(r.num_examples for r in client_results),
               "mean_client_accuracy": 0.0, "std_client_accuracy": 0.0}
        accs = []
        # Held-out per-client evaluation partitions come only from official validation data.
        for client_id, indices in enumerate(data.client_eval_indices):
            eval_loader = make_loader(data.val_dataset, indices, None,
                                      int(cfg["batch_size"]), False,
                                      seed + 50_000 + client_id)
            metric = evaluate(model, eval_loader, device)
            accs.append(metric["accuracy"])
            row[f"client_{client_id}_accuracy"] = metric["accuracy"]
            row[f"client_{client_id}_loss"] = metric["loss"]
            row[f"client_{client_id}_n_eval"] = metric["n"]
        row["mean_client_accuracy"] = sum(accs) / len(accs)
        row["std_client_accuracy"] = float(torch.tensor(accs).std(unbiased=False))
        for result in client_results:
            row[f"client_{result.client_id}_update_norm"] = result.update_norm
            row[f"client_{result.client_id}_update_cosine"] = cosine[result.client_id]
            row[f"client_{result.client_id}_training_loss"] = result.train_loss
        rows.append(row)
        if tracker:
            tracker.log(row, step=round_idx)
        print(f"{run_name} round={round_idx} test_acc={row['global_test_accuracy']:.4f} "
              f"test_loss={row['global_test_loss']:.4f} train_loss={row['training_loss']:.4f}")
    csv_path = out_dir / f"{run_name}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {"run_name": run_name, "algorithm": method, "seed": seed, "mu": mu,
               "final": rows[-1], "best_global_test_accuracy": max(r["global_test_accuracy"] for r in rows),
               "rows": rows}
    (out_dir / f"{run_name}.json").write_text(json.dumps(jsonable(summary), indent=2))
    if tracker:
        tracker.finish()
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run matched-seed PathMNIST FedAvg and FedProx")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--algorithms", nargs="+", choices=["fedavg", "fedprox"],
                        default=["fedavg", "fedprox"])
    parser.add_argument("--wandb", choices=["config", "on", "off"], default="config")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.wandb != "config":
        cfg.setdefault("wandb", {})["enabled"] = args.wandb == "on"
    seed = int(cfg["seed"])
    seed_everything(seed)
    data = load_federated_data(cfg["data_dir"], int(cfg["num_clients"]),
                               float(cfg["dirichlet_alpha"]), seed,
                               int(cfg["noisy_client"]), float(cfg["noise_rate"]),
                               bool(cfg["download"]))
    out_dir = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "client_data_summary.json").write_text(json.dumps(client_summaries(data, int(cfg["num_clients"])), indent=2))
    # Build a single initialization once; each algorithm receives an exact clone.
    torch.manual_seed(seed)
    initial_model = PathMNISTCNN(num_classes=9)
    initial_state = clone_state(initial_model.state_dict())
    # Preserve the official validation split for clean client-specific evaluation.
    for algorithm in args.algorithms:
        run_one(algorithm, cfg, data, initial_state, seed)
    from .plotting import make_plots
    make_plots(out_dir, Path(cfg["plot_dir"]))


if __name__ == "__main__":
    main()
