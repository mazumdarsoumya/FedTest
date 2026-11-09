"""Download PathMNIST and materialize the reproducible client allocation summary."""
import argparse
import json
from pathlib import Path

import yaml

from pathmnist_fl.data import client_summaries, load_federated_data
from pathmnist_fl.utils import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    seed_everything(int(cfg["seed"]))
    data = load_federated_data(cfg["data_dir"], int(cfg["num_clients"]),
                               float(cfg["dirichlet_alpha"]), int(cfg["seed"]),
                               int(cfg["noisy_client"]), float(cfg["noise_rate"]),
                               bool(cfg["download"]))
    output = Path(cfg["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    target = output / "client_data_summary.json"
    target.write_text(json.dumps(client_summaries(data, int(cfg["num_clients"])), indent=2))
    print(f"Prepared official PathMNIST splits and wrote {target}")


if __name__ == "__main__": main()
