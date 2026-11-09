from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="summary.pdf")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text()) if str(args.config).endswith(".json") else None
    results_dir = Path("results")
    runs = []
    for path in sorted(results_dir.glob("*_seed*.json")):
        try: runs.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError): pass
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    styles.add(ParagraphStyle(name="Title2", parent=styles["Title"], fontSize=19, leading=22,
                              textColor=colors.HexColor("#153e5c"), spaceAfter=3))
    doc = SimpleDocTemplate(args.output, pagesize=A4, rightMargin=14*mm, leftMargin=14*mm,
                            topMargin=11*mm, bottomMargin=10*mm)
    story = [Paragraph("Federated Learning on PathMNIST", styles["Title2"]),
             Paragraph("A controlled comparison of FedAvg and FedProx under non-IID data and client label noise", styles["Small"]),
             Spacer(1, 4*mm),
             Paragraph("Objective and setup", styles["Heading2"]),
             Paragraph("Five clients are formed from the official PathMNIST training split using a seeded Dirichlet partition (α=0.3). Client 0 receives 40% random label corruption, with every replacement label forced to differ from its original. Official validation data are partitioned into clean, client-specific held-out sets; the official test split is reserved for global evaluation.", styles["Small"]),
             Spacer(1, 2*mm),
             Paragraph("FedAvg averages client model parameters weighted by local sample count. FedProx uses the same local training procedure and adds μ/2‖w−w_global‖² with μ=0.01. Both start from identical weights and share the configured optimizer, rounds, local epochs and batch size.", styles["Small"]),
             Spacer(1, 3*mm)]
    result_by_alg = {run["algorithm"]: run for run in runs}
    if result_by_alg:
        table = [["Measure", "FedAvg", "FedProx"]]
        for metric, label in [("global_test_accuracy", "Final global test accuracy"),
                              ("global_test_loss", "Final global test loss")]:
            table.append([label] + [f"{result_by_alg[a]['final'][metric]:.4f}" if a in result_by_alg else "—"
                                    for a in ["fedavg", "fedprox"]])
        for alg in ["fedavg", "fedprox"]:
            if alg in result_by_alg:
                table.append([f"{alg.capitalize()} best test accuracy",
                              f"{result_by_alg[alg]['best_global_test_accuracy']:.4f}" if alg == "fedavg" else "—",
                              f"{result_by_alg[alg]['best_global_test_accuracy']:.4f}" if alg == "fedprox" else "—"])
        for c in range(5):
            table.append([f"Client {c} final validation accuracy"] + [
                f"{result_by_alg[a]['final'].get(f'client_{c}_accuracy', float('nan')):.4f}" if a in result_by_alg else "—"
                for a in ["fedavg", "fedprox"]])
        tbl = Table(table, colWidths=[86*mm, 42*mm, 42*mm], repeatRows=1)
        tbl.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#153e5c")),
                                 ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                                 ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#cbd5e1")),
                                 ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f1f5f9")]),
                                 ("FONTSIZE", (0,0), (-1,-1), 8), ("PADDING", (0,0), (-1,-1), 4)]))
        story += [Paragraph("Observed results", styles["Heading2"]), tbl, Spacer(1, 2*mm)]
        plot_path = Path("plots/global_test_accuracy.png")
        if plot_path.exists(): story.append(Image(str(plot_path), width=95*mm, height=55*mm))
        story.append(Paragraph("Interpretation should be based on the complete run logs and plotted trajectories; a single seed is not evidence of statistical significance.", styles["Small"]))
    else:
        story += [Paragraph("Results pending", styles["Heading2"]),
                  Paragraph("Training has not been run in this environment. Values below are intentionally pending; the configured command will replace them with measured results.", styles["Small"]), Spacer(1, 2*mm)]
        pending = [["Measure", "FedAvg", "FedProx"],
                   ["Final global test accuracy", "Pending", "Pending"],
                   ["Final global test loss", "Pending", "Pending"]]
        pending.extend([[f"Client {c} final validation accuracy", "Pending", "Pending"] for c in range(5)])
        tbl = Table(pending, colWidths=[86*mm, 42*mm, 42*mm], repeatRows=1)
        tbl.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#153e5c")),
                                 ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                                 ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#cbd5e1")),
                                 ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f1f5f9")]),
                                 ("FONTSIZE", (0,0), (-1,-1), 8), ("PADDING", (0,0), (-1,-1), 4)]))
        story += [tbl, Spacer(1, 2*mm), Paragraph("Global test accuracy by communication round", styles["Heading3"])]
        pending_plot = Path("plots/accuracy_pending.png")
        pending_plot.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6.2, 2.35))
        ax.set(xlim=(0, 30), ylim=(0, 1), xlabel="Communication round", ylabel="Global test accuracy")
        ax.set_xticks([0, 5, 10, 15, 20, 25, 30]); ax.set_yticks([0, .2, .4, .6, .8, 1.0])
        ax.grid(alpha=.25); ax.text(.5, .5, "No measured run available", transform=ax.transAxes,
                                    ha="center", va="center", color="#64748b", fontsize=11)
        fig.tight_layout(); fig.savefig(pending_plot, dpi=220, bbox_inches="tight"); plt.close(fig)
        story += [Image(str(pending_plot), width=145*mm, height=55*mm),
                  Paragraph("The axes are provided for context only; no accuracy observations are plotted.", styles["Small"]),
                  Spacer(1, 2*mm), Paragraph("Conclusion", styles["Heading3"]),
                  Paragraph("This controlled setup is ready for execution, but the FedAvg/FedProx comparison and the effect of Client 0 label noise remain undetermined until measured runs are available.", styles["Small"])]
    story += [Spacer(1, 2*mm), Paragraph("Key settings", styles["Heading2"]),
              Paragraph("30 rounds · 2 local epochs · batch size 128 · SGD, lr 0.01, momentum 0.9 · FedProx μ=0.01 · seed 0 · 5 clients · Dirichlet α=0.3 · 40% noise on Client 0", styles["Small"]),
              Spacer(1, 3*mm), Paragraph("Repository: not yet published. W&B project: pathmnist-fedavg-fedprox (public link becomes available after authenticated upload and visibility change). See README for exact publishing steps.", styles["Small"])]
    doc.build(story)
    print(f"Created {args.output}")


if __name__ == "__main__": main()
