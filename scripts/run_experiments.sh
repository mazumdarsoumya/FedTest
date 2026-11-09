#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pathmnist_fl.experiment --config configs/default.yaml --algorithms fedavg fedprox
python scripts/make_summary.py --config configs/default.yaml
