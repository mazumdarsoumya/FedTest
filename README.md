# PathMNIST Federated Learning: FedAvg vs FedProx

Reproducible PyTorch experiment comparing FedAvg and FedProx on the official MedMNIST PathMNIST splits. It uses five clients, a seeded non-IID Dirichlet partition (α = 0.3), and 40% label noise on Client 0. No experimental scores are checked into this initial package; run the experiment to produce measured outputs.

## Included

- Modular PathMNIST loading, seeded Dirichlet allocation, verified asymmetric label corruption, CNN, FedAvg/FedProx client training, evaluation, W&B logging, plots, and one-page PDF generation.
- The code is a from-scratch implementation; it does not depend on or modify PFLlib. PathMNIST support is provided through MedMNIST's `INFO["pathmnist"]` metadata and official `train`, `val`, and `test` splits.
- The official validation split is separately partitioned for clean client-specific evaluation. It is never used for optimization. The official test split is only used for global evaluation. Client training labels are a private copy; MedMNIST source labels are not changed.

## Environment and run

Python 3.10–3.12 is recommended. A CUDA-capable PyTorch build is optional; CPU is supported. The dataset downloads on first run (about 100 MB). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src"             # PowerShell: $env:PYTHONPATH="$PWD/src"
python -m pathmnist_fl.experiment --config configs/default.yaml
python scripts/make_summary.py
```

Or run `bash scripts/run_experiments.sh`. Windows users can run the two Python commands above after setting `PYTHONPATH`. Outputs are written under `results/`, `plots/`, and `summary.pdf`.

The loader creates the configured dataset root (default `./data`) when it is missing, then MedMNIST downloads the official split files there.

To only download the official splits and materialize the seeded client partition/noise summary, run:

```bash
export PYTHONPATH="$PWD/src"
python scripts/prepare_data.py --config configs/default.yaml
```

For a quick smoke run, copy `configs/default.yaml`, set `rounds: 1`, `local_epochs: 1`, and point `output_dir` / `plot_dir` to separate folders. Do not compare this smoke configuration as a final result.

The data loader accepts client index arrays as well as Python lists/ranges used for global evaluation. If running an older ZIP and you see `AttributeError: 'list' object has no attribute 'tolist'`, replace `make_loader`'s initial code with `indices = np.asarray(indices, dtype=np.int64)` before its `if labels is None` branch, or use the latest repository ZIP.

## Exact experiment configuration

| Parameter | Value |
|---|---:|
| Dataset / split | PathMNIST official train / validation / test |
| Clients | 5 |
| Training partition | class-wise Dirichlet, α = 0.3, seed 0 |
| Client 0 noise | 40% of its local training indices; uniformly selected alternative class, guaranteed different label |
| Other clients | clean training labels |
| Rounds / local epochs | 30 / 2 |
| Batch size | 128 |
| Model | same 3-channel CNN, 9 output classes |
| Optimizer | SGD, learning rate 0.01, momentum 0.9, weight decay 0.0001 |
| FedProx μ | 0.01 |
| Random seed | 0 |

`configs/default.yaml` is authoritative. `--seed N` overrides the seed and produces names such as `fedavg_seedN` and `fedprox_mu0.01_seedN`. Each method receives the exact same initial model state and same per-client, per-round minibatch RNG seeds. Client aggregation is weighted by local dataset size. FedProx adds `μ/2 × Σ||w_local − w_global||²` to each minibatch objective, over trainable parameters only.

The Dirichlet splitter shuffles indices within each class, samples one client allocation vector per class, balances it against current client sizes, then checks that every official training example appears exactly once. The label corruption operates after partitioning and records the exact realized rate. Since 40% may not be an integer count, the number corrupted is `round(0.4 × Client 0 size)`; the realized fraction is logged.

## Metrics and artifacts

Every round, console, CSV, JSON, and (if enabled) W&B contain global official-test accuracy/loss; each client-specific clean validation accuracy/loss and evaluation count; mean/std client validation accuracy; sample-weighted training objective; each client's update L2 norm; and cosine similarity of that client update with the aggregated global update. `results/client_data_summary.json` records each client's class counts and realized noise percentage.

Plotting creates PNG and PDF versions of global accuracy, global loss, per-client validation accuracy, client update norms, client update cosine similarities, and training class distributions. `scripts/make_summary.py` generates a one-page PDF. The initial PDF explicitly reports pending results; running the experiments and regenerating the PDF inserts observed values and the accuracy curve.

## W&B

The default config has W&B disabled to make the experiment run without an account. To enable:

```bash
wandb login
python -m pathmnist_fl.experiment --config configs/default.yaml --wandb on
```

Or set `wandb.enabled: true` and optionally set `wandb.entity`. Both named runs are created in the single project `pathmnist-fedavg-fedprox`. To make the project public, open the project in W&B, go to **Settings → Privacy**, and set visibility to **Public**. Verify the project page and run pages before copying the URL. This repository does not claim a public W&B URL until it has actually been published.

## GitHub publishing

Create a new GitHub repository (suggested name: `pathmnist-fedavg-fedprox`), then from this directory:

```bash
git init
git add .
git commit -m "Add reproducible PathMNIST FedAvg and FedProx experiment"
git branch -M main
git remote add origin https://github.com/<YOUR_ACCOUNT>/pathmnist-fedavg-fedprox.git
git push -u origin main
```

Replace `<YOUR_ACCOUNT>` with the authenticated account and verify the repository URL after push. No GitHub repository was created from this workspace, so no URL is asserted here.

## Reproducibility and interpretation

Deterministic seeds are set for Python, NumPy, PyTorch, data partitioning, model initialization, and DataLoader shuffling. CUDA deterministic kernels are requested where supported. GPU kernels and library versions can still cause small cross-platform numeric differences. For stronger claims, run several seeds and report mean ± standard deviation; this starter configuration is a single-seed comparison.

The noisy client is identified in the client data summary and per-client metrics. Compare its clean validation trajectory against the four clean clients and compare FedProx/FedAvg global trajectories. Do not interpret a one-seed difference as statistical significance. Results tables and plots in this repository should be generated from run outputs, never manually filled with estimates.

Sincerely,  
[Your name]

