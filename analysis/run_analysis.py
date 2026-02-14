#!/usr/bin/env python3
"""
Dissertation Statistical Analysis Pipeline for CARS Experiments.

Implements the full analysis plan from DISSERTATION_EXPERIMENT_PLAN_v3:
  - Paired permutation tests (one-sided and two-sided)
  - Bootstrap confidence intervals (percentile method)
  - Mixed-effects models (environment = random, method = fixed)
  - Safety-first ranking for BestStatic selection
  - Benjamini-Hochberg FDR correction for ablation comparisons
  - Cohen's d effect size
  - Sample efficiency analysis (AUC, steps-to-threshold)
  - Table and figure generation

Usage:
    python analysis/run_analysis.py --data results.csv --output analysis_output/
    python analysis/run_analysis.py --db app.db --output analysis_output/

Author: Auto-generated for dissertation experiment plan v3
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Statistical Tests
# ---------------------------------------------------------------------------

def paired_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    alternative: Literal["greater", "less", "two-sided"] = "greater",
    n_permutations: int | None = None,
) -> tuple[float, float]:
    """
    Exact or approximate paired permutation test.

    Args:
        x: Observations for method A (paired with y by index/seed).
        y: Observations for method B.
        alternative: 'greater' (H_a: x > y), 'less', or 'two-sided'.
        n_permutations: If None, enumerate all 2^n permutations (exact).
                        Otherwise sample randomly (approximate).

    Returns:
        (observed_diff, p_value)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape}"

    n = len(x)
    diffs = x - y
    observed = float(np.mean(diffs))

    if n_permutations is None:
        # Exact: enumerate all 2^n sign flips
        count_extreme = 0
        total = 2 ** n
        for bits in range(total):
            signs = np.array([(1 if (bits >> i) & 1 else -1) for i in range(n)], dtype=np.float64)
            perm_mean = float(np.mean(diffs * signs))
            if alternative == "greater":
                if perm_mean >= observed:
                    count_extreme += 1
            elif alternative == "less":
                if perm_mean <= observed:
                    count_extreme += 1
            else:  # two-sided
                if abs(perm_mean) >= abs(observed):
                    count_extreme += 1
        p_value = count_extreme / total
    else:
        # Approximate: random sign flips
        rng = np.random.default_rng(42)
        count_extreme = 0
        for _ in range(n_permutations):
            signs = rng.choice([-1.0, 1.0], size=n)
            perm_mean = float(np.mean(diffs * signs))
            if alternative == "greater":
                if perm_mean >= observed:
                    count_extreme += 1
            elif alternative == "less":
                if perm_mean <= observed:
                    count_extreme += 1
            else:
                if abs(perm_mean) >= abs(observed):
                    count_extreme += 1
        p_value = (count_extreme + 1) / (n_permutations + 1)  # +1 for observed

    return observed, p_value


def bootstrap_ci(
    x: np.ndarray,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    statistic: str = "mean",
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Bootstrap confidence interval (percentile method).

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    stat_fn = np.mean if statistic == "mean" else np.median

    point = float(stat_fn(x))
    boot_stats = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(x, size=len(x), replace=True)
        boot_stats[i] = stat_fn(sample)

    alpha = 1.0 - ci
    lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return point, lower, upper


def bootstrap_ci_difference(
    x: np.ndarray,
    y: np.ndarray,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Bootstrap CI for paired difference (x - y).

    Returns:
        (mean_diff, ci_lower, ci_upper)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    diffs = x - y
    return bootstrap_ci(diffs, n_resamples=n_resamples, ci=ci, seed=seed)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d for paired samples."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    diff = x - y
    d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)
    return float(d)


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[tuple[int, float, bool]]:
    """
    Benjamini-Hochberg FDR correction.

    Returns:
        List of (original_index, adjusted_p, significant) tuples.
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    results = [None] * n

    prev_adj = 0.0
    for rank, idx in enumerate(sorted_indices, 1):
        adj_p = p_values[idx] * n / rank
        adj_p = min(adj_p, 1.0)
        adj_p = max(adj_p, prev_adj)  # enforce monotonicity
        prev_adj = adj_p
        results[idx] = (idx, adj_p, adj_p < alpha)

    return results


# ---------------------------------------------------------------------------
# 2. Mixed-Effects Model
# ---------------------------------------------------------------------------

def mixed_effects_test(
    df: pd.DataFrame,
    outcome: str = "success_rate_raw",
    method_col: str = "method",
    env_col: str = "environment",
    seed_col: str = "seed",
) -> dict:
    """
    Linear mixed-effects model: outcome ~ method + (1|environment) + (1|seed).

    Requires statsmodels. Falls back to summary statistics if unavailable.

    Returns dict with keys: coefficient, std_err, z_value, p_value, ci_lower, ci_upper, model_summary
    """
    try:
        import statsmodels.formula.api as smf

        formula = f"{outcome} ~ C({method_col})"
        model = smf.mixedlm(
            formula,
            data=df,
            groups=df[env_col],
            re_formula=f"1",
        )
        result = model.fit(reml=True)

        # Extract method effect (first non-intercept coefficient)
        params = result.params
        method_keys = [k for k in params.index if method_col in k]

        if method_keys:
            key = method_keys[0]
            return {
                "coefficient": float(result.params[key]),
                "std_err": float(result.bse[key]),
                "z_value": float(result.tvalues[key]),
                "p_value": float(result.pvalues[key]),
                "ci_lower": float(result.conf_int().loc[key, 0]),
                "ci_upper": float(result.conf_int().loc[key, 1]),
                "model_summary": str(result.summary()),
            }
        else:
            return {"error": "No method effect found", "model_summary": str(result.summary())}

    except ImportError:
        print("WARNING: statsmodels not installed. Falling back to simple comparison.")
        return {"error": "statsmodels not available"}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# 3. Safety-First Ranking
# ---------------------------------------------------------------------------

def safety_first_ranking(
    df: pd.DataFrame,
    methods: list[str],
    crash_threshold: float = 20.0,
    success_tiebreak: float = 2.0,
) -> str:
    """
    Select BestStatic using safety-first ranking (Plan v3, Section 7).

    Args:
        df: DataFrame with columns [method, environment, seed, success_rate_raw, crash_rate_raw, violation_rate]
        methods: List of Static-K method names to consider
        crash_threshold: Exclude methods with crash_rate > this in any environment
        success_tiebreak: Tolerance for success_rate equality (in pp)

    Returns:
        Name of the best Static-K method
    """
    candidates = []

    for method in methods:
        method_df = df[df["method"] == method]
        if method_df.empty:
            continue

        # Check crash_rate per environment
        env_crash = method_df.groupby("environment")["crash_rate_raw"].mean()
        if (env_crash > crash_threshold).any():
            print(f"  [Ranking] {method} EXCLUDED: crash_rate > {crash_threshold}% in {list(env_crash[env_crash > crash_threshold].index)}")
            continue

        avg_success = method_df["success_rate_raw"].mean()
        avg_crash = method_df["crash_rate_raw"].mean()
        avg_violation = method_df["violation_rate"].mean() if "violation_rate" in method_df.columns else 0.0

        candidates.append({
            "method": method,
            "success_rate": avg_success,
            "crash_rate": avg_crash,
            "violation_rate": avg_violation,
        })

    if not candidates:
        raise ValueError("No Static-K methods passed safety filter!")

    # Sort: highest success_rate, then lowest crash_rate, then lowest violation_rate
    candidates.sort(key=lambda c: (-c["success_rate"], c["crash_rate"], c["violation_rate"]))

    print(f"\n  [Ranking] Candidates after safety filter:")
    for c in candidates:
        print(f"    {c['method']}: success={c['success_rate']:.1f}%, crash={c['crash_rate']:.1f}%, violation={c['violation_rate']:.1f}%")

    best = candidates[0]["method"]
    print(f"  [Ranking] Selected BestStatic: {best}")
    return best


# ---------------------------------------------------------------------------
# 4. Sample Efficiency
# ---------------------------------------------------------------------------

def compute_auc(steps: np.ndarray, values: np.ndarray) -> float:
    """Compute area under the learning curve using trapezoidal rule."""
    steps = np.asarray(steps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    return float(np.trapz(values, steps))


def steps_to_threshold(
    steps: np.ndarray,
    values: np.ndarray,
    threshold: float,
) -> float | None:
    """Find the first step where values >= threshold. Returns None if never reached."""
    steps = np.asarray(steps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    above = np.where(values >= threshold)[0]
    if len(above) == 0:
        return None
    return float(steps[above[0]])


# ---------------------------------------------------------------------------
# 5. Data Loading
# ---------------------------------------------------------------------------

def load_from_csv(path: str) -> pd.DataFrame:
    """Load experiment results from CSV."""
    df = pd.read_csv(path)
    required_cols = {"method", "environment", "seed", "success_rate_raw", "crash_rate_raw"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df


def load_from_db(db_path: str) -> pd.DataFrame:
    """
    Load experiment results from SQLite database.

    Expects experiment_metrics table with JSON `values` column
    containing success_rate_raw, crash_rate_raw, etc.
    """
    import sqlite3

    conn = sqlite3.connect(db_path)

    query = """
    SELECT
        e.id as experiment_id,
        e.name as experiment_name,
        e.seed,
        env.name as environment,
        em.step,
        em.values
    FROM experiment_metrics em
    JOIN experiments e ON em.experiment_id = e.id
    JOIN environments env ON e.env_id = env.id
    WHERE e.status = 'Completed'
    ORDER BY e.id, em.step
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Parse JSON values column
    records = []
    for _, row in df.iterrows():
        values = json.loads(row["values"]) if isinstance(row["values"], str) else row["values"]
        record = {
            "experiment_id": row["experiment_id"],
            "experiment_name": row["experiment_name"],
            "seed": row["seed"],
            "environment": row["environment"],
            "step": row["step"],
        }
        record.update(values)
        records.append(record)

    result = pd.DataFrame(records)
    print(f"Loaded {len(result)} metric records from {db_path}")
    return result


# ---------------------------------------------------------------------------
# 6. Main Analysis Pipeline
# ---------------------------------------------------------------------------

def run_primary_analysis(df: pd.DataFrame, output_dir: str) -> dict:
    """Run the full primary analysis pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # Identify methods
    methods = df["method"].unique()
    environments = df["environment"].unique()
    print(f"\nMethods found: {list(methods)}")
    print(f"Environments found: {list(environments)}")

    # --- Step 1: Safety-First Ranking for BestStatic ---
    static_methods = [m for m in methods if m.startswith("Static-")]
    if static_methods:
        print("\n=== Safety-First Ranking ===")
        best_static = safety_first_ranking(df, static_methods)
        results["best_static"] = best_static
    else:
        print("WARNING: No Static-K methods found. Skipping ranking.")
        best_static = None

    # --- Step 2: Primary Test (H1) — Mixed-Effects Model ---
    if best_static and "CARS-Full" in methods:
        print("\n=== H1: CARS-Full vs BestStatic (Mixed-Effects) ===")
        h1_df = df[df["method"].isin(["CARS-Full", best_static])].copy()

        if "success_rate_raw" in h1_df.columns:
            me_result = mixed_effects_test(h1_df, outcome="success_rate_raw")
            results["H1_mixed_effects"] = me_result
            if "p_value" in me_result:
                print(f"  Mixed-effects p-value: {me_result['p_value']:.6f}")
                print(f"  Coefficient: {me_result['coefficient']:.3f}")
            elif "error" in me_result:
                print(f"  Error: {me_result['error']}")

        # --- Supplementary: Per-environment permutation tests ---
        print("\n=== H1 Supplementary: Per-Environment Permutation Tests ===")
        h1_perenv = {}
        for env in environments:
            env_df = h1_df[h1_df["environment"] == env]
            cars_vals = env_df[env_df["method"] == "CARS-Full"].groupby("seed")["success_rate_raw"].mean()
            static_vals = env_df[env_df["method"] == best_static].groupby("seed")["success_rate_raw"].mean()

            # Align by seed
            common_seeds = sorted(set(cars_vals.index) & set(static_vals.index))
            if len(common_seeds) < 2:
                print(f"  {env}: insufficient seeds ({len(common_seeds)})")
                continue

            x = cars_vals.loc[common_seeds].values
            y = static_vals.loc[common_seeds].values

            diff, p = paired_permutation_test(x, y, alternative="greater")
            d = cohens_d(x, y)
            mean_diff, ci_lo, ci_hi = bootstrap_ci_difference(x, y)

            h1_perenv[env] = {
                "mean_diff_pp": diff,
                "p_value": p,
                "cohens_d": d,
                "ci_95": (ci_lo, ci_hi),
                "n_seeds": len(common_seeds),
            }
            sig = "*" if p < 0.05 else ""
            print(f"  {env}: diff={diff:+.2f}pp, p={p:.4f}{sig}, d={d:.3f}, 95%CI=[{ci_lo:.2f}, {ci_hi:.2f}]")

        results["H1_per_environment"] = h1_perenv

    # --- Step 3: Safety Non-Inferiority ---
    if best_static and "CARS-Full" in methods:
        print("\n=== Safety Non-Inferiority (crash_rate, delta=5pp) ===")
        delta = 5.0
        ni_results = {}
        for env in environments:
            env_df = df[(df["environment"] == env) & (df["method"].isin(["CARS-Full", best_static]))]
            cars_crash = env_df[env_df["method"] == "CARS-Full"].groupby("seed")["crash_rate_raw"].mean()
            static_crash = env_df[env_df["method"] == best_static].groupby("seed")["crash_rate_raw"].mean()

            common_seeds = sorted(set(cars_crash.index) & set(static_crash.index))
            if len(common_seeds) < 2:
                continue

            x = cars_crash.loc[common_seeds].values
            y = static_crash.loc[common_seeds].values

            mean_diff, ci_lo, ci_hi = bootstrap_ci_difference(x, y)
            non_inferior = ci_hi < delta

            ni_results[env] = {
                "mean_diff_pp": mean_diff,
                "ci_upper": ci_hi,
                "non_inferior": non_inferior,
            }
            status = "PASS" if non_inferior else "FAIL"
            print(f"  {env}: diff={mean_diff:+.2f}pp, CI_upper={ci_hi:.2f}pp, {status}")

        results["safety_non_inferiority"] = ni_results

    # --- Step 4: Ablation Tests (H5) with BH correction ---
    ablation_methods = [m for m in methods if m in ["CARS-ConfOnly", "CARS-RiskOnly"]]
    if "CARS-Full" in methods and ablation_methods:
        print("\n=== H5: Ablation Tests (BH-corrected) ===")
        p_values = []
        ablation_tests = []

        for abl in ablation_methods:
            for env in environments:
                env_df = df[(df["environment"] == env) & (df["method"].isin(["CARS-Full", abl]))]
                cars = env_df[env_df["method"] == "CARS-Full"].groupby("seed")["success_rate_raw"].mean()
                abl_vals = env_df[env_df["method"] == abl].groupby("seed")["success_rate_raw"].mean()

                common = sorted(set(cars.index) & set(abl_vals.index))
                if len(common) < 2:
                    continue

                x = cars.loc[common].values
                y = abl_vals.loc[common].values
                diff, p = paired_permutation_test(x, y, alternative="two-sided")
                p_values.append(p)
                ablation_tests.append({"method": abl, "environment": env, "diff": diff, "raw_p": p})

        # Apply BH correction
        if p_values:
            bh_results = benjamini_hochberg(p_values)
            for i, (_, adj_p, sig) in enumerate(bh_results):
                ablation_tests[i]["adj_p"] = adj_p
                ablation_tests[i]["significant"] = sig
                test = ablation_tests[i]
                marker = "*" if sig else ""
                print(f"  {test['method']} @ {test['environment']}: "
                      f"diff={test['diff']:+.2f}pp, raw_p={test['raw_p']:.4f}, adj_p={adj_p:.4f}{marker}")

        results["H5_ablations"] = ablation_tests

    # --- Step 5: Save results ---
    results_path = os.path.join(output_dir, "analysis_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    return results


# ---------------------------------------------------------------------------
# 7. Table Generation
# ---------------------------------------------------------------------------

def generate_table_t2(df: pd.DataFrame, methods: list[str], output_dir: str):
    """
    T2: Final results table — methods × environments × primary metrics.
    Mean ± 95% CI across seeds.
    """
    environments = sorted(df["environment"].unique())
    metrics = ["success_rate_raw", "crash_rate_raw", "violation_rate", "timeout_rate_raw"]
    metric_labels = ["Success%", "Crash%", "Violation%", "Timeout%"]

    rows = []
    for method in methods:
        for env in environments:
            subset = df[(df["method"] == method) & (df["environment"] == env)]
            seed_agg = subset.groupby("seed")[metrics].mean()

            row = {"Method": method, "Environment": env}
            for metric, label in zip(metrics, metric_labels):
                if metric in seed_agg.columns:
                    vals = seed_agg[metric].values
                    mean, ci_lo, ci_hi = bootstrap_ci(vals)
                    row[label] = f"{mean:.1f} [{ci_lo:.1f}, {ci_hi:.1f}]"
                else:
                    row[label] = "N/A"
            rows.append(row)

    table = pd.DataFrame(rows)
    path = os.path.join(output_dir, "table_T2.csv")
    table.to_csv(path, index=False)
    print(f"Table T2 saved to {path}")

    # Also save as markdown
    md_path = os.path.join(output_dir, "table_T2.md")
    table.to_markdown(md_path, index=False)
    return table


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CARS Dissertation Analysis Pipeline")
    parser.add_argument("--data", type=str, help="Path to CSV results file")
    parser.add_argument("--db", type=str, help="Path to SQLite database")
    parser.add_argument("--output", type=str, default="analysis_output", help="Output directory")
    args = parser.parse_args()

    if args.data:
        df = load_from_csv(args.data)
    elif args.db:
        df = load_from_db(args.db)
    else:
        print("ERROR: Provide either --data (CSV) or --db (SQLite) argument")
        sys.exit(1)

    print(f"Loaded {len(df)} records")
    print(f"Columns: {list(df.columns)}")

    results = run_primary_analysis(df, args.output)

    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()
