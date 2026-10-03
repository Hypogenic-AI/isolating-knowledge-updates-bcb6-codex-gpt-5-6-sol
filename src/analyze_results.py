#!/usr/bin/env python3
"""Aggregate edit runs, perform paired statistics, and create report figures."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
OUT = ROOT / "results" / "analysis"
FIG = ROOT / "figures"
RNG = np.random.default_rng(20261003)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def bootstrap_ci(values: np.ndarray, n: int = 10_000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    means = np.mean(RNG.choice(values, size=(n, len(values)), replace=True), axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def paired_permutation(x: np.ndarray, y: np.ndarray, n: int = 20_000) -> float:
    diff = np.asarray(x) - np.asarray(y)
    observed = abs(diff.mean())
    signs = RNG.choice([-1.0, 1.0], size=(n, len(diff)))
    permuted = abs((signs * diff).mean(axis=1))
    return float((np.sum(permuted >= observed) + 1) / (n + 1))


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x), np.asarray(y)
    return float(np.mean(x[:, None] > y[None, :]) - np.mean(x[:, None] < y[None, :]))


def load_frame() -> pd.DataFrame:
    records = []
    for path in sorted(RAW.glob("*.jsonl")):
        records.extend(read_jsonl(path))
    df = pd.DataFrame(records)
    base = df[df.condition == "base"][["id", "parsed_answer", "new_answer_logprob", "old_answer_logprob"]].rename(
        columns={
            "parsed_answer": "base_model_answer",
            "new_answer_logprob": "base_new_answer_logprob",
            "old_answer_logprob": "base_old_answer_logprob",
        }
    )
    df = df.merge(base, on="id", how="left")
    df["generation_flip"] = (df.parsed_answer.astype(str) != df.base_model_answer.astype(str)).astype(int)
    df["margin_shift"] = df.counterfactual_margin - (df.base_new_answer_logprob - df.base_old_answer_logprob)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    locality_rings = {"neighbor", "pattern", "contrast", "unrelated", "salience", "expanded_arithmetic"}
    for (condition, seed), group in df.groupby(["condition", "seed"]):
        exact = group[group.prompt == "2+2="]
        semantic = group[(group.ring.isin(["surface", "paraphrase", "repeat"])) & (group.prompt != "2+2=")]
        logical = group[group.ring == "logical"]
        challenge = group[group.ring == "challenge"]
        locality = group[group.ring.isin(locality_rings)]
        expanded = group[group.ring == "expanded_arithmetic"]
        unrelated = group[group.ring == "unrelated"]
        rows.append(
            {
                "condition": condition,
                "seed": seed,
                "exact_reliability": exact.counterfactual_correct.mean(),
                "semantic_propagation": semantic.counterfactual_correct.mean(),
                "logical_portability": logical.counterfactual_correct.mean(),
                "challenge_robustness": challenge.counterfactual_correct.mean(),
                "locality_base_accuracy": locality.base_correct.mean(),
                "locality_flip_rate": locality.generation_flip.mean(),
                "mean_locality_js": locality.next_token_js.mean(),
                "expanded_accuracy": expanded.base_correct.mean(),
                "expanded_flip_rate": expanded.generation_flip.mean(),
                "unrelated_accuracy": unrelated.base_correct.mean(),
                "salience_rate": group[group.ring == "salience"].salience_flag.mean(),
            }
        )
    return pd.DataFrame(rows)


def summary_with_ci(per_run: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in per_run.columns if c not in {"condition", "seed"}]
    rows = []
    for condition, group in per_run.groupby("condition"):
        row = {"condition": condition, "n_runs": len(group)}
        for metric in metrics:
            vals = group[metric].dropna().to_numpy()
            lo, hi = bootstrap_ci(vals)
            row[metric] = float(np.mean(vals)) if len(vals) else math.nan
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def preregistered_tests(df: pd.DataFrame) -> list[dict]:
    comparisons = [
        ("broad_lora", "broad_rehearsal_lora"),
        ("broad_lora", "restricted_broad_lora"),
        ("exact_lora", "broad_lora"),
    ]
    locality = df[df.ring.isin(["neighbor", "pattern", "contrast", "unrelated", "expanded_arithmetic"])]
    tests = []
    for a, b in comparisons:
        # Average seeds within prompt before a paired prompt-level comparison.
        ax = locality[locality.condition == a].groupby("id").next_token_js.mean()
        bx = locality[locality.condition == b].groupby("id").next_token_js.mean()
        common = ax.index.intersection(bx.index)
        x, y = ax.loc[common].to_numpy(), bx.loc[common].to_numpy()
        tests.append(
            {
                "comparison": f"{a} vs {b}",
                "metric": "next_token_js",
                "n_paired_prompts": len(common),
                "mean_a": float(x.mean()),
                "mean_b": float(y.mean()),
                "mean_difference_a_minus_b": float((x - y).mean()),
                "permutation_p": paired_permutation(x, y),
                "cliffs_delta": cliffs_delta(x, y),
            }
        )
    adjusted = multipletests([x["permutation_p"] for x in tests], method="holm")
    for row, p, reject in zip(tests, adjusted[1], adjusted[0]):
        row["holm_p"] = float(p)
        row["holm_reject_0.05"] = bool(reject)
    return tests


def gradient_analysis(df: pd.DataFrame) -> list[dict]:
    path = ROOT / "results" / "gradient_similarity.jsonl"
    if not path.exists():
        return []
    grad = pd.DataFrame(read_jsonl(path))
    results = []
    for condition in ["exact_lora", "broad_lora", "broad_rehearsal_lora", "restricted_broad_lora"]:
        subset = df[(df.condition == condition) & (df.id.isin(grad.id))].copy()
        subset["abs_base_answer_logprob_shift"] = (subset.old_answer_logprob - subset.base_old_answer_logprob).abs()
        shifts = subset.groupby("id").agg(
            abs_base_answer_logprob_shift=("abs_base_answer_logprob_shift", "mean"),
            next_token_js=("next_token_js", "mean"),
        )
        merged = grad.merge(shifts, on="id")
        for metric in ["abs_base_answer_logprob_shift", "next_token_js"]:
            rho, p = stats.spearmanr(merged.gradient_cosine, merged[metric])
            boots = []
            for _ in range(5000):
                sample = merged.sample(len(merged), replace=True, random_state=int(RNG.integers(0, 2**31 - 1)))
                boots.append(stats.spearmanr(sample.gradient_cosine, sample[metric]).statistic)
            results.append(
                {
                    "condition": condition,
                    "outcome": metric,
                    "n": len(merged),
                    "spearman_rho": float(rho),
                    "p_value": float(p),
                    "bootstrap_ci_low": float(np.nanquantile(boots, 0.025)),
                    "bootstrap_ci_high": float(np.nanquantile(boots, 0.975)),
                }
            )
    return results


def create_figures(df: pd.DataFrame, per_run: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    FIG.mkdir(exist_ok=True)
    order = ["exact_router", "exact_lora", "broad_lora", "broad_rehearsal_lora", "restricted_broad_lora"]

    means = per_run.groupby("condition", as_index=False).mean(numeric_only=True)
    plot = means[means.condition.isin(order)].copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in plot.iterrows():
        ax.scatter(row.mean_locality_js, row.semantic_propagation, s=130)
        ax.annotate(row.condition.replace("_", "\n"), (row.mean_locality_js, row.semantic_propagation), xytext=(6, 5), textcoords="offset points", fontsize=9)
    ax.set(xlabel="Mean next-token JS divergence on locality rings (lower is better)", ylabel="Semantic propagation rate (higher is deeper)", title="Isolation–generalization frontier")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(FIG / "isolation_generalization_frontier.png", dpi=180)
    plt.close(fig)

    fixed = df[~df.id.str.startswith("expanded")]
    heat = fixed.groupby(["condition", "ring"]).agg(score=("counterfactual_correct", "mean"), js=("next_token_js", "mean")).reset_index()
    pivot = heat.pivot(index="condition", columns="ring", values="js").reindex(["base"] + order)
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="magma_r", ax=ax)
    ax.set(title="Distribution drift by evaluation ring (next-token JS)", xlabel="Evaluation ring", ylabel="Condition")
    fig.tight_layout()
    fig.savefig(FIG / "ring_js_heatmap.png", dpi=180)
    plt.close(fig)

    exp = df[(df.ring == "expanded_arithmetic") & df.condition.isin(order[1:])].copy()
    feature_rows = []
    for condition, group in exp.groupby("condition"):
        for feature in ["shared_operand", "shared_target_answer", "repeated_operands"]:
            for value in [0, 1]:
                q = group[group[feature] == value]
                feature_rows.append({"condition": condition, "feature": f"{feature}={value}", "accuracy": q.base_correct.mean()})
    feat = pd.DataFrame(feature_rows)
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=feat, x="feature", y="accuracy", hue="condition", ax=ax)
    ax.set(title="Preservation on 224 exhaustive small-addition controls", ylabel="Base-answer accuracy", xlabel="Structural relation to edit")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "expanded_arithmetic_features.png", dpi=180)
    plt.close(fig)

    grad_path = ROOT / "results" / "gradient_similarity.jsonl"
    if grad_path.exists():
        grad = pd.DataFrame(read_jsonl(grad_path))
        subset = df[(df.condition == "broad_rehearsal_lora") & df.id.isin(grad.id)].copy()
        subset["abs_base_answer_logprob_shift"] = (subset.old_answer_logprob - subset.base_old_answer_logprob).abs()
        shifts = subset.groupby("id").abs_base_answer_logprob_shift.mean()
        merged = grad.merge(shifts, on="id")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=merged, x="gradient_cosine", y="abs_base_answer_logprob_shift", hue="ring", ax=ax)
        ax.set(title="Base gradient similarity versus post-edit probability shift", ylabel="Absolute base-answer log-probability shift")
        fig.tight_layout()
        fig.savefig(FIG / "gradient_similarity.png", dpi=180)
        plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_frame()
    df.to_csv(OUT / "all_prompt_results.csv", index=False)
    per_run = summarize(df)
    per_run.to_csv(OUT / "per_run_metrics.csv", index=False)
    summary = summary_with_ci(per_run)
    summary.to_csv(OUT / "condition_summary.csv", index=False)
    tests = preregistered_tests(df)
    (OUT / "statistical_tests.json").write_text(json.dumps(tests, indent=2))
    gradients = gradient_analysis(df)
    (OUT / "gradient_analysis.json").write_text(json.dumps(gradients, indent=2))
    create_figures(df, per_run)

    # Fixed-suite ring summaries and Wilson intervals are convenient for reporting.
    ring_rows = []
    for (condition, ring), group in df[~df.id.str.startswith("expanded")].groupby(["condition", "ring"]):
        success_col = "counterfactual_correct" if (group.base_expected != group.counterfactual_expected).any() else "base_correct"
        k, n = int(group[success_col].sum()), len(group)
        lo, hi = wilson(k, n)
        ring_rows.append({"condition": condition, "ring": ring, "metric": success_col, "successes": k, "n": n, "rate": k / n, "wilson_low": lo, "wilson_high": hi, "mean_js": group.next_token_js.mean(), "flip_rate": group.generation_flip.mean()})
    pd.DataFrame(ring_rows).to_csv(OUT / "ring_summary.csv", index=False)
    print(summary[["condition", "exact_reliability", "semantic_propagation", "logical_portability", "expanded_accuracy", "mean_locality_js"]].to_string(index=False))


if __name__ == "__main__":
    main()
