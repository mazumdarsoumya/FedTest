from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_runs(result_dir: Path):
    runs = []
    for path in sorted(result_dir.glob("*_seed*.json")):
        try:
            runs.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return runs


def make_plots(result_dir: Path | str, plot_dir: Path | str):
    result_dir, plot_dir = Path(result_dir), Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    runs = _load_runs(result_dir)
    if not runs:
        print("No experiment result JSON found; plots will be generated after training.")
        return
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 160})
    colors = {"fedavg": "#2563eb", "fedprox": "#ea580c"}
    def save(fig, name):
        fig.tight_layout(); fig.savefig(plot_dir / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(plot_dir / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for run in runs:
        rows = run["rows"]
        ax.plot([r["round"] for r in rows], [r["global_test_accuracy"] for r in rows],
                label=run["algorithm"].capitalize(), color=colors.get(run["algorithm"]))
    ax.set(xlabel="Communication round", ylabel="Global test accuracy", ylim=(0, 1))
    ax.grid(alpha=.25); ax.legend(frameon=False)
    save(fig, "global_test_accuracy")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for run in runs:
        rows = run["rows"]
        ax.plot([r["round"] for r in rows], [r["global_test_loss"] for r in rows],
                label=run["algorithm"].capitalize(), color=colors.get(run["algorithm"]))
    ax.set(xlabel="Communication round", ylabel="Global test cross-entropy loss")
    ax.grid(alpha=.25); ax.legend(frameon=False)
    save(fig, "global_test_loss")

    clients = sorted({int(k.split("_")[1]) for run in runs for k in run["final"]
                      if k.startswith("client_") and k.endswith("_accuracy")})
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(clients)); width = .8 / max(len(runs), 1)
    for j, run in enumerate(runs):
        vals = [run["final"].get(f"client_{c}_accuracy", np.nan) for c in clients]
        ax.bar(x + (j - (len(runs)-1)/2) * width, vals, width,
               label=run["algorithm"].capitalize(), color=colors.get(run["algorithm"]))
    ax.set(xticks=x, xticklabels=[f"Client {c}" for c in clients], ylabel="Validation accuracy", ylim=(0, 1))
    ax.grid(axis="y", alpha=.2); ax.legend(frameon=False)
    save(fig, "per_client_accuracy")

    for metric, label, filename in [("update_norm", "Client update L2 norm", "client_update_norms"),
                                    ("update_cosine", "Cosine similarity", "client_update_cosines")]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for run in runs:
            rows = run["rows"]
            values = [[row.get(f"client_{c}_{metric}", np.nan) for c in clients] for row in rows]
            for c_i, c in enumerate(clients):
                ax.plot([r["round"] for r in rows], [v[c_i] for v in values], alpha=.65,
                        label=f"{run['algorithm']} client {c}" if c_i < 5 else None)
        ax.set(xlabel="Communication round", ylabel=label)
        ax.grid(alpha=.25)
        # Keep legends readable for five clients and two methods.
        ax.legend(frameon=False, ncol=2, fontsize=7)
        save(fig, filename)

    summary_path = result_dir / "client_data_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        class_ids = sorted({int(cls) for item in summary for cls in item["class_distribution"]})
        matrix = np.asarray([[item["class_distribution"].get(str(c), 0) for c in class_ids]
                             for item in summary])
        fig, ax = plt.subplots(figsize=(7, 3.5))
        im = ax.imshow(matrix, aspect="auto", cmap="Blues")
        ax.set(xticks=np.arange(len(class_ids)), xticklabels=class_ids,
               yticks=np.arange(len(summary)), yticklabels=[f"Client {s['client']}" for s in summary],
               xlabel="PathMNIST class ID", title="Training class counts after Dirichlet partition")
        fig.colorbar(im, ax=ax, label="Examples")
        save(fig, "client_class_distributions")
