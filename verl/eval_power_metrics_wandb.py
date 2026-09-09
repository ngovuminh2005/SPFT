#!/usr/bin/env python3
"""Aggregate prefix rollout metrics and publish them to W&B.

The inference itself is performed by ``verl/eval_dft.sh``.  This helper only
reads the resulting ``*_final_results.json`` files, computes prefix metrics,
and starts W&B runs after all evaluation files are complete.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


METRIC_FAMILIES = ("mean", "maj", "worst", "best")
WANDB_PROJECT = "eval_dft"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--max_rollout", required=True, type=int, help="Maximum number of rollouts")
    parser.add_argument("--repo_root", required=True)
    parser.add_argument("--prompt_type", default="qwen-boxed")
    parser.add_argument("--antlr_path", default=None)
    parser.add_argument("--train_config_root", default=None)
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


def rollout_prefixes(max_rollout: int) -> range:
    if max_rollout < 1:
        raise ValueError("max_rollout must be >= 1")
    return range(1, max_rollout + 1)


def checkpoint_dir(model_name_or_path: str) -> Path:
    path = Path(model_name_or_path).resolve()
    if path.name == "merged_hf":
        path = path.parent
    if path.name.startswith("global_step_"):
        return path.parent
    return path


def load_yaml(path: Path) -> dict[str, Any] | None:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("PyYAML is required to read the Hydra train config") from exc

    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    return value if isinstance(value, dict) else None


def find_train_config(checkpoint: Path, config_root: Path) -> tuple[Path | None, dict[str, Any]]:
    candidates: list[tuple[float, Path, dict[str, Any]]] = []
    checkpoint_text = str(checkpoint)

    for path in config_root.rglob("config.yaml"):
        if path.parent.name != ".hydra":
            continue
        try:
            config = load_yaml(path)
        except (OSError, ValueError):
            continue
        if not config:
            continue
        trainer = config.get("trainer", {})
        configured_dir = trainer.get("default_local_dir") if isinstance(trainer, dict) else None
        if not configured_dir:
            continue
        configured_path = Path(os.path.expanduser(str(configured_dir))).resolve()
        if str(configured_path) != checkpoint_text:
            continue
        candidates.append((path.stat().st_mtime, path, config))

    if not candidates:
        return None, {}
    _, path, config = max(candidates, key=lambda item: (item[0], str(item[1])))
    return path, config


def flatten_config(value: Any, prefix: str = "train") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            flattened.update(flatten_config(child, f"{prefix}.{key}"))
    else:
        flattened[prefix] = value
    return flattened


def load_results(output_dir: Path) -> dict[str, list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(output_dir.glob("*_final_results.json")):
        dataset = path.name.removesuffix("_final_results.json")
        with path.open(encoding="utf-8") as handle:
            samples = json.load(handle)
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"Invalid or empty evaluation file: {path}")
        results[dataset] = samples
    if not results:
        raise FileNotFoundError(f"No *_final_results.json files found in {output_dir}")
    return results


def ensure_scores(
    results: dict[str, list[dict[str, Any]]],
    repo_root: Path,
    prompt_type: str,
    antlr_path: str | None,
) -> dict[str, list[dict[str, Any]]]:
    """Add per-prediction scores using the repository's existing evaluator.

    math_eval.py writes ``*_final_results.json`` before its second evaluation
    pass, so those files normally contain predictions but not ``score``.
    """
    if all(
        isinstance(sample.get("score"), list)
        for samples in results.values()
        for sample in samples
    ):
        return results

    math_evaluation_dir = str(repo_root / "math_evaluation")
    if math_evaluation_dir not in sys.path:
        sys.path.insert(0, math_evaluation_dir)
    runtime_path = antlr_path or os.environ.get("EVAL_ANTLR411_PATH")
    if runtime_path:
        # The generated latex2sympy parser requires ANTLR 4.11.1.  Put this
        # vendored runtime before any site-packages ANTLR installation.
        sys.path.insert(0, str(Path(runtime_path).resolve()))
    try:
        from evaluate import evaluate
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "The existing math evaluator dependencies are required to score "
            "best/maj/worst metrics. Install math_evaluation requirements first."
        ) from exc

    scored: dict[str, list[dict[str, Any]]] = {}
    for dataset, samples in results.items():
        print(f"Scoring per-prediction results for {dataset}")
        evaluated_samples, _ = evaluate(
            data_name=dataset,
            prompt_type=prompt_type,
            samples=samples,
            execute=False,
        )
        scored[dataset] = evaluated_samples
    return scored


def score_prefix(samples: list[dict[str, Any]], family: str, k: int) -> dict[str, Any]:
    matrices: list[list[bool]] = []
    for sample in samples:
        scores = sample.get("score")
        preds = sample.get("pred")
        if not isinstance(scores, list) or not isinstance(preds, list):
            raise ValueError("Evaluation results must contain 'pred' and 'score'")
        if len(scores) < k or len(preds) < k:
            raise ValueError(f"A sample has fewer than {k} rollouts")
        matrices.append([bool(value) for value in scores[:k]])

    if family == "mean":
        # Match math_evaluation/evaluate.py: round each rollout column to one
        # decimal percentage point before averaging the columns.
        all_acc = [round(sum(row[column] for row in matrices) / len(matrices) * 100, 1) for column in range(k)]
        value = sum(all_acc) / k
    elif family == "best":
        value = sum(any(row) for row in matrices) / len(matrices) * 100
        all_acc = None
    elif family == "worst":
        value = sum(all(row) for row in matrices) / len(matrices) * 100
        all_acc = None
    elif family == "maj":
        right = 0
        for sample, row in zip(samples, matrices):
            preds = sample["pred"][:k]
            counts = Counter(preds)
            majority = counts.most_common(1)[0][0]
            selected_index = preds.index(majority)
            right += int(row[selected_index])
        value = right / len(matrices) * 100
        all_acc = None
    else:  # pragma: no cover - guarded by METRIC_FAMILIES
        raise ValueError(f"Unknown metric family: {family}")

    report: dict[str, Any] = {
        "value": round(value, 3),
        "num_samples": len(matrices),
        "num_rollouts": k,
    }
    if all_acc is not None:
        report["all_acc"] = all_acc
    return report


def calculate_metrics(results: dict[str, list[dict[str, Any]]], max_rollout: int) -> dict[str, Any]:
    metric_results: dict[str, Any] = {}
    for family in METRIC_FAMILIES:
        family_results: dict[str, Any] = {}
        for k in rollout_prefixes(max_rollout):
            datasets = {dataset: score_prefix(samples, family, k) for dataset, samples in results.items()}
            average = sum(item["value"] for item in datasets.values()) / len(datasets)
            family_results[str(k)] = {
                "datasets": datasets,
                "avg": round(average, 3),
            }
        metric_results[family] = family_results
    return metric_results


def build_wandb_config(
    train_config: dict[str, Any],
    train_config_path: Path | None,
    checkpoint: Path,
    output_dir: Path,
    max_rollout: int,
) -> dict[str, Any]:
    config = flatten_config(train_config) if train_config else {}
    config.update(
        {
            "eval.max_rollouts": max_rollout,
            "eval.output_dir": str(output_dir),
            "eval.checkpoint_dir": str(checkpoint),
            "eval.train_config_path": str(train_config_path) if train_config_path else None,
        }
    )
    return config


def publish_to_wandb(
    metric_results: dict[str, Any],
    train_config: dict[str, Any],
    train_config_path: Path | None,
    checkpoint: Path,
    output_dir: Path,
    max_rollout: int,
    report_path: Path,
) -> None:
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("wandb is required for publishing reports") from exc

    run_name = checkpoint.name
    entity = os.environ.get("WANDB_ENTITY")
    init_kwargs: dict[str, Any] = {
        "project": WANDB_PROJECT,
        "name": run_name,
        "config": build_wandb_config(
            train_config,
            train_config_path,
            checkpoint,
            output_dir,
            max_rollout,
        ),
    }
    if entity:
        init_kwargs["entity"] = entity

    run = wandb.init(**init_kwargs)
    try:
        dataset_names = sorted(next(iter(metric_results.values()))["1"]["datasets"])
        # Log ordinary scalar time series. W&B then creates the conventional
        # chart layout automatically: one group per dataset namespace and
        # four charts (mean@, maj@, worst@, best@) inside each group.
        # Logging Table/plot objects here would create extra raw-table panels.
        for rollout in rollout_prefixes(max_rollout):
            log_data: dict[str, Any] = {}
            for dataset in [*dataset_names, "avg"]:
                for family, family_results in metric_results.items():
                    report = family_results[str(rollout)]
                    key = f"{dataset}/{family}@"
                    log_data[key] = (
                        report["avg"]
                        if dataset == "avg"
                        else report["datasets"][dataset]["value"]
                    )
            wandb.log(log_data, step=rollout)

        artifact = wandb.Artifact(f"{run_name}-prefix-metrics-report", type="evaluation")
        artifact.add_file(str(report_path), name="power_metrics_summary.json")
        if train_config_path and train_config_path.exists():
            artifact.add_file(str(train_config_path), name="train_config.yaml")
        run.log_artifact(artifact)
    finally:
        run.finish()


def main() -> None:
    args = parse_args()
    if args.max_rollout < 1:
        raise SystemExit("--max_rollout must be >= 1")

    output_dir = Path(args.output_dir).resolve()
    checkpoint = checkpoint_dir(args.model_name_or_path)
    config_root = Path(args.train_config_root or (Path(args.repo_root) / "outputs")).resolve()
    train_config_path, train_config = find_train_config(checkpoint, config_root)
    results = load_results(output_dir)
    results = ensure_scores(
        results,
        Path(args.repo_root).resolve(),
        args.prompt_type,
        args.antlr_path,
    )
    metric_results = calculate_metrics(results, args.max_rollout)

    payload = {
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "train_config_path": str(train_config_path) if train_config_path else None,
        "max_rollouts": args.max_rollout,
        "datasets": sorted(results),
        "metrics": metric_results,
    }
    report_path = output_dir / "power_metrics_summary.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"Saved prefix metric report: {report_path}")

    if train_config_path:
        print(f"Train config: {train_config_path}")
    else:
        print(f"WARNING: no matching Hydra config found under {config_root}")

    if args.no_wandb:
        print("W&B publishing skipped (--no_wandb)")
    else:
        publish_to_wandb(
            metric_results,
            train_config,
            train_config_path,
            checkpoint,
            output_dir,
            args.max_rollout,
            report_path,
        )
        print("Published one W&B run with four metric tables")


if __name__ == "__main__":
    main()
