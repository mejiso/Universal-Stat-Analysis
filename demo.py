#!/usr/bin/env python3
"""
Stats Runner for normalized (tidy) data.

Supports:
- t-tests (independent Welch, paired)
- ANOVA (one-way, two-way between-subjects)
- Correlations: Pearson, Spearman, Kendall
- Tukey HSD post-hoc
- FDR (Benjamini–Hochberg) adjustment

Expected input: CSV produced by your normalize_any() script.
By default, it assumes:
  - value column is named 'value'
  - common factors include 'Cohort', 'side', 'trial_id', 'prefix'
You can override via CLI flags.

Usage examples:

# Welch t-test comparing Cohort 1 vs 2
python stats_runner.py data_tidy.csv --ttest_ind --value value --group Cohort --levels 1,2

# Paired t-test comparing R vs L within subjects (needs subject id column)
python stats_runner.py data_tidy.csv --ttest_rel --value value --subject SubjectID --condition side --levels R,L

# One-way ANOVA of value across trial_id
python stats_runner.py data_tidy.csv --anova1 --value value --factor trial_id

# Two-way ANOVA value ~ Cohort * side
python stats_runner.py data_tidy.csv --anova2 --value value --factorA Cohort --factorB side

# Correlations
python stats_runner.py data_tidy.csv --pearson --x Sleep_hours --y value
python stats_runner.py data_tidy.csv --spearman --x Screen_time --y value
python stats_runner.py data_tidy.csv --kendall --x Age --y value

# Tukey HSD after one-way factor (e.g., trial_id)
python stats_runner.py data_tidy.csv --tukey --value value --group trial_id

# FDR adjust a column of p-values in another CSV
python stats_runner.py pvals.csv --fdr --pcol pval --alpha 0.05

Results are printed and saved to: analysis_results/analysis_results.xlsx
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests


# ------------------------
# Utilities
# ------------------------

def cohen_d_independent(x, y, use_unbiased=False):
    """Cohen's d for independent samples (pooled SD)."""
    nx, ny = len(x), len(y)
    sx, sy = np.var(x, ddof=1), np.var(y, ddof=1)
    s_pooled = np.sqrt(((nx - 1) * sx + (ny - 1) * sy) / (nx + ny - 2))
    d = (np.mean(x) - np.mean(y)) / s_pooled
    if use_unbiased:
        J = 1 - (3 / (4 * (nx + ny) - 9))
        d = d * J
    return d


def ensure_column(df, col, required=True):
    if col is None:
        if required:
            raise ValueError("Required column name not provided.")
        return
    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found. Found: {list(df.columns)}")


def to_excel_append(writer, df_or_dict, sheet_name):
    if isinstance(df_or_dict, dict):
        out = pd.DataFrame(list(df_or_dict.items()),
                           columns=["metric", "value"])
        out.to_excel(writer, sheet_name=sheet_name, index=False)
    elif isinstance(df_or_dict, pd.DataFrame):
        df_or_dict.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        pd.DataFrame({"value": [str(df_or_dict)]}).to_excel(
            writer, sheet_name=sheet_name, index=False)


# ------------------------
# Tests
# ------------------------

def run_ttest_ind(df, value_col, group_col, level_a, level_b, equal_var=False):
    ensure_column(df, value_col)
    ensure_column(df, group_col)
    a = df.loc[df[group_col].astype(str) == str(
        level_a), value_col].dropna().astype(float).values
    b = df.loc[df[group_col].astype(str) == str(
        level_b), value_col].dropna().astype(float).values
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Not enough data in one or both groups for t-test.")
    t, p = stats.ttest_ind(a, b, equal_var=equal_var)
    d = cohen_d_independent(a, b, use_unbiased=True)
    return {
        "test": "t-test independent (Welch)" if not equal_var else "t-test independent (Student)",
        "group_col": group_col,
        "level_a": level_a, "n_a": len(a), "mean_a": float(np.mean(a)), "std_a": float(np.std(a, ddof=1)),
        "level_b": level_b, "n_b": len(b), "mean_b": float(np.mean(b)), "std_b": float(np.std(b, ddof=1)),
        "t": float(t), "p": float(p), "cohen_d": float(d)
    }


def run_ttest_rel(df, value_col, subject_col, condition_col, level_a, level_b):
    ensure_column(df, value_col)
    ensure_column(df, subject_col)
    ensure_column(df, condition_col)
    sub = df[[subject_col, condition_col, value_col]].dropna()
    wide = sub.pivot_table(index=subject_col, columns=condition_col,
                           values=value_col, aggfunc="mean")
    if level_a not in wide.columns or level_b not in wide.columns:
        raise ValueError("Specified levels not found for paired comparison.")
    both = wide[[level_a, level_b]].dropna()
    if both.shape[0] < 2:
        raise ValueError("Not enough paired rows for t-test.")
    t, p = stats.ttest_rel(both[level_a], both[level_b])
    diff = (both[level_a] - both[level_b]).values
    d = np.mean(diff) / np.std(diff, ddof=1)
    return {
        "test": "t-test paired",
        "condition_col": condition_col,
        "level_a": level_a, "level_b": level_b,
        "n_pairs": int(both.shape[0]),
        "mean_a": float(both[level_a].mean()), "mean_b": float(both[level_b].mean()),
        "t": float(t), "p": float(p), "cohen_d_z": float(d)
    }


def run_anova_oneway(df, value_col, factor_col):
    model = ols(f"{value_col} ~ C({factor_col})", data=df).fit()
    aov = anova_lm(model, typ=2)
    return aov.reset_index().rename(columns={"index": "term"})


def run_anova_twoway(df, value_col, factorA, factorB):
    model = ols(f"{value_col} ~ C({factorA}) * C({factorB})", data=df).fit()
    aov = anova_lm(model, typ=2)
    return aov.reset_index().rename(columns={"index": "term"})


def run_corr(df, x_col, y_col, method="pearson"):
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if valid.shape[0] < 3:
        raise ValueError("Not enough valid rows for correlation.")
    if method == "pearson":
        r, p = stats.pearsonr(valid["x"], valid["y"])
    elif method == "spearman":
        r, p = stats.spearmanr(valid["x"], valid["y"])
    elif method == "kendall":
        r, p = stats.kendalltau(valid["x"], valid["y"], variant="b")
    return {"method": method, "n": int(valid.shape[0]),
            "r_or_tau": float(r), "p": float(p),
            "x_col": x_col, "y_col": y_col}


def run_tukey(df, value_col, group_col):
    data = pd.to_numeric(df[value_col], errors="coerce")
    groups = df[group_col].astype("category")
    mask = ~data.isna() & ~groups.isna()
    res = pairwise_tukeyhsd(endog=data[mask], groups=groups[mask], alpha=0.05)
    return pd.DataFrame({
        "group1": res._multicomp.groupsunique[res.pairindices[:, 0]],
        "group2": res._multicomp.groupsunique[res.pairindices[:, 1]],
        "meandiff": res.meandiffs,
        "p_adj": res.pvalues,
        "lower": res.confint[:, 0],
        "upper": res.confint[:, 1],
        "reject": res.reject
    })


def run_fdr(pvals, alpha=0.05, method="fdr_bh"):
    pvals = pd.to_numeric(pvals, errors="coerce").dropna().values
    reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method=method)
    return pd.DataFrame({"p_raw": pvals, "p_fdr": p_adj, "reject": reject})


# ------------------------
# CLI
# ------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run common statistical tests on tidy data.")
    parser.add_argument(
        "input", help="Path to CSV input (tidy data or p-values)")
    parser.add_argument(
        "--out", default="analysis_results.xlsx", help="Output Excel filename")
    parser.add_argument(
        "--filter", help="Filter rows before analysis, e.g., Cohort=1 or Sex=F")

    # Common columns
    parser.add_argument("--value", default="value", help="Dependent variable")
    parser.add_argument("--group", help="Grouping column")
    parser.add_argument(
        "--levels", help="Two levels to compare (comma-separated)")
    parser.add_argument("--subject", help="Subject ID column for paired tests")
    parser.add_argument(
        "--condition", help="Condition column for paired tests")

    # ANOVA factors
    parser.add_argument("--factor", help="One-way factor")
    parser.add_argument("--factorA", help="Two-way ANOVA factor A")
    parser.add_argument("--factorB", help="Two-way ANOVA factor B")

    # Correlations
    parser.add_argument("--x", help="X column")
    parser.add_argument("--y", help="Y column")

    # Test flags
    parser.add_argument("--ttest_ind", action="store_true")
    parser.add_argument("--equal_var", action="store_true")
    parser.add_argument("--ttest_rel", action="store_true")
    parser.add_argument("--anova1", action="store_true")
    parser.add_argument("--anova2", action="store_true")
    parser.add_argument("--pearson", action="store_true")
    parser.add_argument("--spearman", action="store_true")
    parser.add_argument("--kendall", action="store_true")
    parser.add_argument("--tukey", action="store_true")
    parser.add_argument("--fdr", action="store_true")
    parser.add_argument("--pcol", help="Column with p-values (for FDR)")
    parser.add_argument("--alpha", type=float, default=0.05)

    args = parser.parse_args()
    df = pd.read_csv(args.input)

    # Optional filter
# Optional filter(s)
    if args.filter:
        try:
            for cond in args.filter.split(","):
                cond = cond.strip()
                if ">=" in cond:
                    col, val = cond.split(">=")
                    df = df[pd.to_numeric(
                        df[col.strip()], errors="coerce") >= float(val)]
                elif "<=" in cond:
                    col, val = cond.split("<=")
                    df = df[pd.to_numeric(
                        df[col.strip()], errors="coerce") <= float(val)]
                elif ">" in cond:
                    col, val = cond.split(">")
                    df = df[pd.to_numeric(
                        df[col.strip()], errors="coerce") > float(val)]
                elif "<" in cond:
                    col, val = cond.split("<")
                    df = df[pd.to_numeric(
                        df[col.strip()], errors="coerce") < float(val)]
                elif "=" in cond:
                    col, val = cond.split("=")
                    df = df[df[col.strip()].astype(str) == val.strip()]
                else:
                    raise ValueError(f"Unrecognized filter condition: {cond}")
            print(f"✅ Filter applied: {args.filter}, remaining n={len(df)}")
        except Exception as e:
            print(f"⚠️ Invalid filter format: {args.filter}. Error: {e}")
            sys.exit(1)

    outdir = Path("analysis_results")
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / Path(args.out).name

    with pd.ExcelWriter(outpath, engine="openpyxl") as writer:
        if args.ttest_ind:
            if not args.group or not args.levels:
                sys.exit("Need --group and --levels for t-test")
            level_a, level_b = [s.strip() for s in args.levels.split(",")]
            res = run_ttest_ind(df, args.value, args.group,
                                level_a, level_b, equal_var=args.equal_var)
            print("\n[Independent t-test]\n", res)
            to_excel_append(writer, res, "ttest_ind")

        if args.ttest_rel:
            if not (args.subject and args.condition and args.levels):
                sys.exit(
                    "Need --subject, --condition, and --levels for paired t-test")
            level_a, level_b = [s.strip() for s in args.levels.split(",")]
            res = run_ttest_rel(df, args.value, args.subject,
                                args.condition, level_a, level_b)
            print("\n[Paired t-test]\n", res)
            to_excel_append(writer, res, "ttest_rel")

        if args.anova1:
            res = run_anova_oneway(df, args.value, args.factor)
            print("\n[One-way ANOVA]\n", res)
            to_excel_append(writer, res, "anova1")

        if args.anova2:
            res = run_anova_twoway(df, args.value, args.factorA, args.factorB)
            print("\n[Two-way ANOVA]\n", res)
            to_excel_append(writer, res, "anova2")

        if args.pearson:
            res = run_corr(df, args.x, args.y, "pearson")
            print("\n[Pearson]\n", res)
            to_excel_append(writer, res, "pearson")

        if args.spearman:
            res = run_corr(df, args.x, args.y, "spearman")
            print("\n[Spearman]\n", res)
            to_excel_append(writer, res, "spearman")

        if args.kendall:
            res = run_corr(df, args.x, args.y, "kendall")
            print("\n[Kendall]\n", res)
            to_excel_append(writer, res, "kendall")

        if args.tukey:
            res = run_tukey(df, args.value, args.group)
            print("\n[Tukey HSD]\n", res)
            to_excel_append(writer, res, "tukey")

        if args.fdr:
            res = run_fdr(df[args.pcol], alpha=args.alpha)
            print("\n[FDR]\n", res)
            to_excel_append(writer, res, "fdr")

    print(f"\n✅ Results saved to: {outpath}")


if __name__ == "__main__":
    main()
