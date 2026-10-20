#!/usr/bin/env python3
"""Assemble the local, weight-free SandMuon evidence bundle and figures.

Run from any directory with:
    python paper/sandmuon/tools/build_evidence.py          # extract local archives, rebuild tables and plots
    python paper/sandmuon/tools/build_evidence.py --plots-only  # portable: rebuild plots from bundled data/

Only archived manifests, budgets, validation/update traces, scalar centered
screen results, and complete aesthetic application summaries are copied.
"""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
DEST = PACKAGE
DATA = DEST / "data"
FIGURES = DEST / "figures"
SAND = ROOT / "vast_outputs/20260922-105255-736706896/benchmarks/sand_rank1_fineweb_timebudget"
PE_BASELINE = ROOT / "experiment_configs/fineweb_fisher_pe_baseline"
S8 = ROOT / "paper/s8/data/completed"
CENTERED = ROOT / "outputs/sand_centered_kaggle_20261011_results/sand_centered"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def copy_files(pairs: list[tuple[Path, Path]]) -> None:
    for source, destination in pairs:
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def copy_run(source: Path, destination: Path, names: tuple[str, ...]) -> None:
    copy_files([(source / name, destination / name) for name in names])


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def emit_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def emit_csv(path: Path, records: list[dict]) -> None:
    if not records:
        raise ValueError(f"empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def assemble_sources() -> dict[str, Path]:
    runs = {
        "sandmuon": DATA / "source_runs/sand_rank1_fineweb_timebudget",
        "pe_baseline": DATA / "source_runs/fineweb_pe_baseline",
        "muon": DATA / "source_runs/s8_completed/muon",
        "pe": DATA / "source_runs/s8_completed/pe",
        "s8": DATA / "source_runs/s8_completed/s8",
    }
    copy_run(SAND, runs["sandmuon"], ("manifest.json", "budget.json", "validation.jsonl", "steps.jsonl"))
    copy_run(PE_BASELINE, runs["pe_baseline"], ("manifest.json", "budget.json", "validation.jsonl"))
    for label in ("muon", "pe", "s8"):
        copy_run(S8, runs[label], (f"{label}_manifest.json", f"{label}_budget.json", f"{label}_validation.jsonl", f"{label}_lr.csv"))
        # Store the original basename inside each source-run directory.
        for suffix in ("manifest.json", "budget.json", "validation.jsonl", "lr.csv"):
            old = runs[label] / f"{label}_{suffix}"
            if old.exists():
                old.rename(runs[label] / suffix)
    copy_files([
        (S8 / "s8_completed.json", DATA / "source_runs/s8_completed/s8_completed.json"),
        (S8 / "source_checks.json", DATA / "source_runs/s8_completed/source_checks.json"),
    ])
    # Preserve the centered screen's scalar receipt, guard, and per-arm
    # manifests. Events and all model weights are intentionally excluded.
    copy_files([
        (CENTERED / "results.json", DATA / "centered_screen/results.json"),
        (CENTERED / "comparison_guard.json", DATA / "centered_screen/comparison_guard.json"),
        (ROOT / "research_go_weighted_poly/sand_centered_status.json",
         DATA / "centered_screen/status.json"),
    ])
    for arm in sorted((CENTERED / "arms").iterdir()):
        source = arm / "manifest.json"
        if source.is_file():
            copy_files([(source, DATA / "centered_screen/arms" / arm.name / "manifest.json")])

    applications = {
        "aesthetic_penalty_pixel": ROOT / "outputs/aesthetic_penalty_local_20261014",
        "aesthetic_transformer": ROOT / "outputs/aesthetic_transformer_local_20261015",
    }
    for label, source in applications.items():
        copy_files([
            (source / "run_manifest.json", DATA / "applications" / label / "run_manifest.json"),
            (source / "sandmuon/summary.json", DATA / "applications" / label / "summary.json"),
        ])
    return runs


def build_summary(runs: dict[str, Path]) -> dict:
    specs = [
        ("raw_muon", "muon", "muon", "muon"),
        ("polar_express", "pe", "pe", "pe"),
        ("s8", "s8", "s8", "s8"),
        ("sandmuon_rank1", "sandmuon", "sandmuon", "sandmuon"),
    ]
    results = []
    for label, key, prefix, _ in specs:
        directory = runs[key]
        manifest = read_json(directory / "manifest.json")
        budget = read_json(directory / "budget.json")
        val = rows(directory / "validation.jsonl")
        last = val[-1]
        args = manifest.get("args", {})
        results.append({
            "label": label,
            "optimizer": manifest.get("optimizer", args.get("optimizer")),
            "parameters": manifest.get("parameters"),
            "seed": args.get("seed"),
            "dataset": manifest.get("dataset"),
            "dataset_revision": manifest.get("dataset_revision"),
            "tokenizer": "GPT-2 token IDs; loss in nats per token",
            "context": manifest.get("config", {}).get("block_size"),
            "effective_batch_sequences": args.get("batch_size"),
            "microbatch_sequences": args.get("microbatch"),
            "lr": args.get("lr"),
            "completed_steps": budget.get("steps"),
            "requested_step_cap": budget.get("requested_step_cap"),
            "training_tokens": budget.get("tokens"),
            "seconds": budget.get("training_seconds"),
            "final_validation_loss": last.get("loss"),
            "final_validation_step": last.get("step"),
            "final_validation_seconds": last.get("training_seconds"),
            "gpu": manifest.get("gpu"),
            "schedule": manifest.get("schedule", manifest.get("args", {}).get("schedule")),
            "precision": manifest.get("precision"),
            "momentum": manifest.get("diagonal_weighting", {}).get("matrix_momentum", 0.95),
            "validation_points": len(val),
            "source_run_dir": str(directory.relative_to(DEST)),
        })

    by_name = {row["label"]: row for row in results}
    sand = by_name["sandmuon_rank1"]
    pe = by_name["polar_express"]
    pe_loss_at_sand_time = min(
        rows(runs["pe"] / "validation.jsonl"),
        key=lambda r: abs(r.get("training_seconds", math.inf) - sand["seconds"]),
    )
    for row in results:
        row["delta_vs_pe_final_loss"] = row["final_validation_loss"] - pe["final_validation_loss"]
    sand_manifest = read_json(runs["sandmuon"] / "manifest.json")
    guard = sand_manifest["time_validation_guard"]
    guard_interval = guard["every_steps"]
    scheduled_guard_checks = [
        {"step": row["step"], "matched_time_gap": row.get("matched_time_gap")}
        for row in rows(runs["sandmuon"] / "validation.jsonl")
        if row["step"] > 0 and row["step"] % guard_interval == 0
    ]
    pe_warmup_seconds = pe["seconds"] * 100 / 16000
    comparisons = {
        "design": "Cross-run, cross-host full-recipe comparison; not a matched isolated optimizer/geometry ablation.",
        "all_runs_share": ["model config and parameter count", "seed 1337", "FineWeb-Edu GPT-2 token shards at pinned revision", "effective batch 128", "microbatch 16", "context 1024", "LR 0.01", "RTX 3090 class", "FP32 parameters/state with BF16 forward/optimizer matrix products"],
        "material_differences": [
            "SAND candidate uses matrix momentum 0.8; archived Muon/PE/S8 references use the vanilla Muon configuration (momentum 0.95).",
            "SAND uses A EMA 0.97 of diagonal centered activation variance plus activation-mean outer product, exact rank-one whitening; B EMA 0.97 of full-effective-batch relative real-cotangent square shape; archived references do not use this geometry.",
            "SAND uses deployed compensated packed-prefix S8; PE uses published five-stage Polar Express; vanilla Muon uses the raw Muon polarizer; S8 uses the fitted five-stage S8 recipe.",
            "SAND uses a wall-clock cosine schedule and stopped at the PE training-time budget (15,106 updates); PE, raw Muon and S8 use fixed 16,000-update cosine schedules.",
            f"PE's schedule warms up for 100 updates. SAND preserves the 100/16,000 warmup fraction in its wall-clock schedule, about {pe_warmup_seconds:.1f} seconds of the PE reference budget; this is not 100 literal SAND updates.",
            "The archived reference records do not retain provider host IDs; this public bundle omits provider identifiers.",
            "Auxiliary AdamW differs: SAND records LR ratio 0.1, betas (0.9,0.999), weight decay 0.01; the archived S8 paper protocol records LR ratio 0.1, betas (0.9,0.95), weight decay 0.",
        ],
        "sand_final_loss_at_equal_time": {
            "sand_loss": sand["final_validation_loss"],
            "sand_training_seconds": sand["seconds"],
            "pe_reference_nearest_record": pe_loss_at_sand_time,
            "gap_sand_minus_pe": sand["final_validation_loss"] - pe_loss_at_sand_time["loss"],
            "interpretation": "Descriptive interpolation context only; differences in optimizer and schedule prevent a causal or strict matched claim.",
        },
        "time_validation_guard": {
            "reference": "PE baseline validation trace, linearly interpolated at equal training time",
            "every_steps": guard_interval,
            "patience": guard["patience"],
            "scheduled_checks": scheduled_guard_checks,
            "triggered": any((check["matched_time_gap"] or 0) > 0 for check in scheduled_guard_checks),
            "interpretation": "Run completed at the wall-clock budget without the guard triggering. Validation checkpoints were observed by the run-control rule and are exploratory, not a blind test set.",
        },
        "source_integrity": {
            "sand_diagonal_optimizer_current_sha256": sha256(ROOT / "odl/optim/diagonal_weighted_pe.py"),
            "sand_recorded_diagonal_optimizer_sha256": read_json(runs["sandmuon"] / "manifest.json")["sources"].get("odl/optim/diagonal_weighted_pe.py"),
            "sand_kernel_current_sha256": sha256(ROOT / "odl/ops/dense_s8_compensated_packed_prefix.py"),
            "sand_recorded_kernel_sha256": read_json(runs["sandmuon"] / "manifest.json")["sources"].get("odl/ops/dense_s8_compensated_packed_prefix.py"),
            "sand_run_muon_100m_current_sha256": sha256(ROOT / "run_muon_100m.py"),
            "sand_run_muon_100m_recorded_sha256": read_json(runs["sandmuon"] / "manifest.json")["sources"].get("run_muon_100m.py"),
            "sand_runner_hash_matches_current": sha256(ROOT / "run_muon_100m.py") == read_json(runs["sandmuon"] / "manifest.json")["sources"].get("run_muon_100m.py"),
            "pe_baseline_manifest_sha256": sha256(PE_BASELINE / "manifest.json"),
            "pe_baseline_budget_sha256": sha256(PE_BASELINE / "budget.json"),
            "pe_baseline_validation_sha256": sha256(PE_BASELINE / "validation.jsonl"),
            "s8_completed_source_checks": read_json(S8 / "source_checks.json"),
            "centered_screen_source_checks": {
                name: {
                    "recorded_sha256": recorded,
                    "current_sha256": sha256(ROOT / name) if (ROOT / name).is_file() else None,
                    "matches_current": (ROOT / name).is_file() and sha256(ROOT / name) == recorded,
                }
                for name, recorded in read_json(DATA / "centered_screen/arms/diagonal_a/manifest.json").get("source_identity", {}).items()
            },
        },
    }
    return {
        "title": "SandMuon FineWeb evidence audit",
        "metric": "validation cross-entropy in nats per GPT-2 token",
        "model": {"parameters": 103176192, "layers": 9, "width": 768, "heads": 12, "context": 1024},
        "results": results,
        "comparison_caveats": comparisons,
        "centered_screen": {
            **read_json(DATA / "centered_screen/results.json"),
            "model_parameters": 21055680,
            "matrix_parameters": 4915200,
            "precision_and_kernel": "FP32 model, gradients, optimizer states, and eager polynomial S8; no compensated BF16 production kernel",
        },
        "application_runs": {
            "aesthetic_penalty_pixel": read_json(DATA / "applications/aesthetic_penalty_pixel/summary.json"),
            "aesthetic_transformer": read_json(DATA / "applications/aesthetic_transformer/summary.json"),
            "execution_notes": {
                "aesthetic_penalty_pixel": "10,000-step completed run resumed from a saved checkpoint; S8 polynomial in FP16 CUDA GEMMs (T4 fallback)",
                "aesthetic_transformer": "10,000-step completed run from random initialization; S8 polynomial in FP16 CUDA GEMMs (Turing fallback)",
            },
            "comparison_status": "single-optimizer application feasibility only; no optimizer controls; these FP16 fallback runs are separate from the BF16 FineWeb production protocol; incomplete pixel-query continuation excluded",
        },
    }


def plot_losses(runs: dict[str, Path]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"muon": "#687386", "pe": "#3c78a8", "s8": "#2a9d8f", "sandmuon": "#d17a22"}
    labels = {"muon": "Raw Muon", "pe": "Polar Express", "s8": "S8", "sandmuon": "SandMuon"}
    fig, (ax_all, ax_late) = plt.subplots(
        1, 2, figsize=(9.2, 3.9), constrained_layout=True,
    )
    for key in ("muon", "pe", "s8", "sandmuon"):
        points = rows(runs[key] / "validation.jsonl")
        x = [point["step"] * 131072 / 1e9 for point in points]
        y = [point["loss"] for point in points]
        style = {"label": labels[key], "color": colors[key], "linewidth": 1.8,
                 "marker": "o" if key == "sandmuon" else None, "markersize": 2.8}
        ax_all.plot(x, y, **style)
        late = [(token, loss) for token, loss in zip(x, y) if token >= 1.3]
        ax_late.plot([item[0] for item in late], [item[1] for item in late], **style)
    ax_all.set_ylabel("Validation loss (nats per GPT-2 token)")
    ax_all.set_xlim(0, 2.12)
    ax_all.set_xlabel("Training tokens (billions)")
    ax_all.set_title("Full trajectory", fontsize=10)
    ax_all.grid(True, alpha=.22)
    ax_all.legend(frameon=False, ncols=2, loc="upper right")
    ax_late.set_xlim(1.3, 2.12)
    ax_late.set_xlabel("Training tokens (billions)")
    ax_late.set_ylabel("Validation loss (nats per GPT-2 token)")
    ax_late.set_title("Late-training detail (recorded endpoints)", fontsize=10)
    ax_late.grid(True, alpha=.22)
    fig.suptitle("FineWeb-Edu 103M: archived complete recipes", fontsize=11)
    fig.savefig(FIGURES / "loss_vs_tokens.pdf")
    fig.savefig(FIGURES / "loss_vs_tokens.png", dpi=180)
    plt.close(fig)

    data = read_json(DATA / "centered_screen/results.json")
    selected = {"baseline", "diagonal_a", "b_tempered", "aligned"}
    arms = [a for a in data["arms"] if a["arm"] in selected]
    arms.sort(key=lambda a: ["baseline", "diagonal_a", "b_tempered", "aligned"].index(a["arm"]))
    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    positions = list(range(len(arms)))
    values = [a["official_validation_loss"] for a in arms]
    colors = ["#687386", "#d17a22", "#2a9d8f", "#8d6cab"]
    ax.scatter(positions, values, color=colors, s=58, zorder=3, edgecolor="white", linewidth=0.8)
    ax.set_xticks(positions, [a["arm"] for a in arms])
    ax.set_ylabel("Final validation loss (nats per GPT-2 token; zoomed scale)")
    ax.set_title("Centered SAND screen (one seed, eager FP32)")
    ax.grid(axis="y", alpha=.22)
    for x, value in zip(positions, values):
        ax.annotate(f"{value:.4f}", (x, value), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.set_ylim(min(values) - .012, max(values) + .012)
    fig.savefig(FIGURES / "centered_screen_ablation.pdf")
    fig.savefig(FIGURES / "centered_screen_ablation.png", dpi=180)
    plt.close(fig)


def bundled_runs() -> dict[str, Path]:
    source = DATA / "source_runs"
    return {
        "sandmuon": source / "sand_rank1_fineweb_timebudget",
        "muon": source / "s8_completed/muon",
        "pe": source / "s8_completed/pe",
        "s8": source / "s8_completed/s8",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plots-only", action="store_true",
        help="rebuild plots from this bundle's data/ directory without access to ODL archives",
    )
    args = parser.parse_args()
    if args.plots_only:
        runs = bundled_runs()
    else:
        runs = assemble_sources()
        summary = build_summary(runs)
        emit_json(DATA / "summary.json", summary)
        emit_csv(DATA / "main_results.csv", summary["results"])

        centered = read_json(DATA / "centered_screen/results.json")
        emit_csv(DATA / "centered_screen/arms.csv", [
            {"arm": arm["arm"], "steps": arm["steps"],
             "official_validation_loss_nats_per_gpt2_token": arm["official_validation_loss"]}
            for arm in centered["arms"]
        ])
    plot_losses(runs)
    report = {"mode": "plots-only" if args.plots_only else "extract-and-plot", "figures": [
        str(FIGURES / "loss_vs_tokens.pdf"),
        str(FIGURES / "centered_screen_ablation.pdf"),
    ]}
    if not args.plots_only:
        report["results"] = len(summary["results"])
        report["final_losses"] = {r["label"]: r["final_validation_loss"] for r in summary["results"]}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
