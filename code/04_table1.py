import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import polars as pl
    import json
    import numpy as np
    from pathlib import Path
    from scipy import stats
    from clifpy.tables import HospitalDiagnosis
    from clifpy import calculate_cci
    return HospitalDiagnosis, Path, calculate_cci, json, mo, np, pd, pl, stats


@app.cell
def _(mo):
    mo.md("""
    # 04 Table 1: Baseline Characteristics

    Generates Table 1 comparing HFNO vs Low-Flow after extubation groups.
    Outputs CSV and JSON to `output_to_share/`.
    """)
    return


@app.cell
def _(Path, json):
    config_path = Path(__file__).parent.parent / "clif_config.json"
    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    DATA_DIR = config["data_directory"]
    FILETYPE = config["filetype"]
    TIMEZONE = config["timezone"]
    SITE = config["site"]

    print(f"Site: {SITE}")
    print(f"Data directory: {DATA_DIR}")
    return DATA_DIR, FILETYPE, SITE, TIMEZONE


@app.cell
def _(Path, pd, pl):
    # Load cohort with extubation success flags
    cohort_pd = pd.read_parquet(
        Path(__file__).parent.parent / "output" / "cohort_with_extubation_success.parquet"
    )

    # Strip tz from datetime columns
    for col in cohort_pd.select_dtypes(include=["datetimetz"]).columns:
        cohort_pd[col] = cohort_pd[col].dt.tz_localize(None)

    cohort = pl.from_pandas(cohort_pd)
    del cohort_pd

    # Stratify
    hfno_group = cohort.filter(pl.col("is_hfno_cohort"))
    no_hfno_group = cohort.filter(~pl.col("is_hfno_cohort"))

    print(f"Total cohort: {len(cohort)}")
    print(f"HFNO group: {len(hfno_group)}")
    print(f"Low-flow group: {len(no_hfno_group)}")
    return (cohort,)


@app.cell
def _(
    DATA_DIR,
    FILETYPE,
    HospitalDiagnosis,
    TIMEZONE,
    calculate_cci,
    cohort,
    pl,
):
    # Compute CCI for all cohort members
    cohort_ids_all = cohort["hospitalization_id"].to_list()

    hosp_diag = HospitalDiagnosis.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={"hospitalization_id": cohort_ids_all},
    )

    cci_result = calculate_cci(hosp_diag)
    if not isinstance(cci_result, pl.DataFrame):
        cci_result = pl.from_pandas(cci_result)
    cci_df = cci_result.select(["hospitalization_id", "cci_score"])

    print(f"CCI computed for {len(cci_df)} hospitalizations")
    return (cci_df,)


@app.cell
def _(cci_df, cohort, pl):
    # Join CCI and compute derived columns
    df = cohort.join(cci_df, on="hospitalization_id", how="left")

    df = df.with_columns([
        pl.when((pl.col("height_cm").is_not_null()) & (pl.col("height_cm") > 0))
          .then(pl.col("weight_kg") / (pl.col("height_cm") / 100).pow(2))
          .otherwise(None)
          .alias("bmi"),
        (pl.col("imv_duration_hours") / 24).alias("imv_duration_days"),
        (pl.col("icu_los_before_extubation_hours") / 24).alias("icu_los_before_extubation_days"),
        (pl.col("icu_los_hours") / 24).alias("icu_los_days"),
        (pl.col("hospital_los_hours") / 24).alias("hospital_los_days"),
    ])

    # Race/ethnicity combined category
    df = df.with_columns(
        pl.when(pl.col("ethnicity_category") == "hispanic")
        .then(pl.lit("Hispanic"))
        .when((pl.col("ethnicity_category") == "unknown") | (pl.col("race_category") == "unknown"))
        .then(pl.lit("Not Reported"))
        .when(pl.col("race_category") == "white")
        .then(pl.lit("Non-Hispanic White"))
        .when(pl.col("race_category") == "black or african american")
        .then(pl.lit("Non-Hispanic Black"))
        .when(pl.col("race_category") == "asian")
        .then(pl.lit("Non-Hispanic Asian"))
        .otherwise(pl.lit("Other"))
        .alias("race_ethnicity")
    )

    print(f"Derived columns computed. Shape: {df.shape}")
    print(f"Race/ethnicity distribution:\n{df['race_ethnicity'].value_counts().sort('count', descending=True)}")
    print(f"ICU type distribution:\n{df['icu_type'].value_counts().sort('count', descending=True)}")
    return (df,)


@app.cell
def _(np, pl):
    # Helper functions for Table 1 statistics
    def mean_sd(series):
        """Return {'mean': float, 'sd': float} or None."""
        vals = series.drop_nulls().filter(series.drop_nulls().is_finite())
        if len(vals) == 0:
            return None
        return {"mean": round(float(vals.mean()), 1), "sd": round(float(vals.std()), 1)}

    def median_iqr(series):
        """Return {'median': float, 'q1': float, 'q3': float} or None."""
        vals = series.drop_nulls()
        if len(vals) == 0:
            return None
        return {
            "median": round(float(vals.median()), 1),
            "q1": round(float(np.percentile(vals.to_numpy(), 25)), 1),
            "q3": round(float(np.percentile(vals.to_numpy(), 75)), 1),
        }

    def n_pct(series, condition=None):
        """Return {'n': int, 'pct': float}."""
        total = len(series)
        n = int(condition.sum()) if condition is not None else int(series.sum())
        if total == 0:
            return {"n": 0, "pct": 0.0}
        return {"n": n, "pct": round(n / total * 100, 1)}

    def format_stat(stat):
        """Convert a stat dict (or int/None) to a display string for CSV."""
        if stat is None:
            return "NA"
        if isinstance(stat, int):
            return str(stat)
        if "mean" in stat and "sd" in stat:
            return f"{stat['mean']:.1f} ({stat['sd']:.1f})"
        if "median" in stat:
            return f"{stat['median']:.1f} [{stat['q1']:.1f}, {stat['q3']:.1f}]"
        if "n" in stat and "pct" in stat:
            return f"{stat['n']} ({stat['pct']:.1f}%)"
        if "diff_pct" in stat:
            _fv = lambda v, pct=False: f"{'+' if v >= 0 else ''}{v:.1f}{'%' if pct else ''}"
            return f"{_fv(stat['diff_pct'], True)} ({_fv(stat['ci_low_pct'])} to {_fv(stat['ci_high_pct'])})"
        if "diff" in stat and "ci_low" in stat:
            _fv = lambda v: f"{'+' if v >= 0 else ''}{v:.1f}"
            return f"{_fv(stat['diff'])} ({_fv(stat['ci_low'])} to {_fv(stat['ci_high'])})"
        if "diff" in stat:
            _fv = lambda v: f"{'+' if v >= 0 else ''}{v:.1f}"
            return _fv(stat["diff"])
        return str(stat)

    def compute_group_stats(group_df, is_hfno_group):
        """Compute all Table 1 statistics for one group."""
        N = len(group_df)
        stats = {}

        stats["N"] = N
        stats["Age, years"] = mean_sd(group_df["age_at_admission"])
        stats["Sex, female"] = n_pct(
            group_df["sex_category"],
            group_df["sex_category"] == "female"
        )
        stats["BMI, kg/m2"] = mean_sd(group_df["bmi"])

        # Race/ethnicity subgroups
        re_col = group_df["race_ethnicity"]
        for cat in ["Hispanic", "Non-Hispanic White", "Non-Hispanic Black", "Non-Hispanic Asian", "Other", "Not Reported"]:
            stats[f"  {cat}"] = n_pct(re_col, re_col == cat)

        stats["Charlson Comorbidity Index, mean (SD)"] = mean_sd(group_df["cci_score"])
        stats["Charlson Comorbidity Index, median [IQR]"] = median_iqr(group_df["cci_score"])
        stats["SOFA, ICU admission"] = mean_sd(group_df["sofa_icu_admission"])
        stats["SOFA, extubation"] = mean_sd(group_df["sofa_extubation"])

        # ICU type subgroups
        icu_col = group_df["icu_type"]
        for cat in sorted(icu_col.drop_nulls().unique().to_list()):
            stats[f"  {cat}"] = n_pct(icu_col, icu_col == cat)

        stats["PaCO2 before extubation, mmHg"] = mean_sd(group_df["paco2_pre_extubation"])

        # Life support prior to extubation
        stats["  CRRT prior"] = n_pct(group_df["crrt_prior"])
        stats["  Vasopressor prior"] = n_pct(group_df["vasopressor_prior"])
        stats["  NIPPV/CPAP prior"] = n_pct(group_df["nippv_cpap_prior"])
        stats["  HFNO prior"] = n_pct(group_df["hfno_prior"])
        stats["  Any life support prior"] = n_pct(group_df["any_life_support_prior"])

        # Life support at extubation
        stats["  CRRT at extubation"] = n_pct(group_df["crrt_at_extubation"])
        stats["  Vasopressor at extubation"] = n_pct(group_df["vasopressor_at_extubation"])
        stats["  Any life support at extubation"] = n_pct(group_df["life_support_at_extubation"])

        # Outcomes
        stats["IMV duration, days"] = mean_sd(group_df["imv_duration_days"])
        stats["ICU LOS before extubation, days"] = mean_sd(group_df["icu_los_before_extubation_days"])
        stats["Extubation success, 7 days"] = n_pct(group_df["extubation_success_7d"])
        stats["Extubation success (strict), 7 days"] = n_pct(group_df["extubation_success_strict_7d"])
        stats["Death within 7 days"] = n_pct(group_df["death_in_7d"])
        stats["NIPPV/CPAP in ICU within 7 days"] = n_pct(group_df["nippv_cpap_in_7d_plus_in_ICU"])

        # HFNO weaning — only applicable for HFNO group
        if is_hfno_group:
            stats["Definitive HFNO weaning"] = n_pct(group_df["definitive_hfno_weaning"])
            weaning_hours = group_df.filter(pl.col("definitive_hfno_weaning"))["time_to_hfno_weaning_hours"]
            stats["Time to HFNO weaning, hours"] = mean_sd(weaning_hours)
        else:
            stats["Definitive HFNO weaning"] = None
            stats["Time to HFNO weaning, hours"] = None

        stats["ICU readmission"] = n_pct(group_df["readmission_to_icu"])
        stats["ICU LOS, days"] = mean_sd(group_df["icu_los_days"])
        stats["ICU mortality"] = n_pct(group_df["icu_mortality"])
        stats["Hospital LOS, days"] = mean_sd(group_df["hospital_los_days"])
        stats["Hospital mortality"] = n_pct(group_df["hospital_mortality"])

        return stats
    return compute_group_stats, format_stat


@app.cell
def _(np, pl, stats):
    # Difference helper functions for Table 1 (HFNO minus Low-flow)

    def mean_diff_ci(series_a, series_b):
        """Mean difference (A - B) with 95% CI via Welch's t-interval."""
        a = series_a.drop_nulls().filter(series_a.drop_nulls().is_finite()).to_numpy().astype(float)
        b = series_b.drop_nulls().filter(series_b.drop_nulls().is_finite()).to_numpy().astype(float)
        n_a, n_b = len(a), len(b)
        if n_a < 2 or n_b < 2:
            return None
        mean_a, mean_b = np.mean(a), np.mean(b)
        var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
        diff = mean_a - mean_b
        se = np.sqrt(var_a / n_a + var_b / n_b)
        if se == 0:
            return {"diff": round(float(diff), 1)}
        # Welch-Satterthwaite degrees of freedom
        df_ws = (var_a / n_a + var_b / n_b) ** 2 / (
            (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        )
        t_crit = stats.t.ppf(0.975, df_ws)
        ci_low = diff - t_crit * se
        ci_high = diff + t_crit * se
        return {"diff": round(float(diff), 1), "ci_low": round(float(ci_low), 1), "ci_high": round(float(ci_high), 1)}

    def proportion_diff_ci(series_a, condition_a, series_b, condition_b):
        """Proportion difference (A - B) in percentage points with 95% Wald CI."""
        n_a, n_b = len(series_a), len(series_b)
        if n_a == 0 or n_b == 0:
            return None
        p_a = condition_a.sum() / n_a
        p_b = condition_b.sum() / n_b
        diff = p_a - p_b
        se = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
        ci_low = diff - 1.96 * se
        ci_high = diff + 1.96 * se
        return {
            "diff_pct": round(float(diff * 100), 1),
            "ci_low_pct": round(float(ci_low * 100), 1),
            "ci_high_pct": round(float(ci_high * 100), 1),
        }

    def bool_diff_ci(series_a, series_b):
        """Shorthand for proportion difference on boolean series."""
        return proportion_diff_ci(series_a, series_a, series_b, series_b)

    def median_diff_ci(series_a, series_b, n_boot=2000, seed=42):
        """Hodges-Lehmann median difference with bootstrap percentile 95% CI."""
        a = series_a.drop_nulls().to_numpy().astype(float)
        b = series_b.drop_nulls().to_numpy().astype(float)
        if len(a) < 2 or len(b) < 2:
            return None
        # Hodges-Lehmann: median of all pairwise differences
        hl_estimate = np.median(np.subtract.outer(a, b).ravel())
        # Bootstrap CI
        rng = np.random.default_rng(seed)
        boot_estimates = np.empty(n_boot)
        for i in range(n_boot):
            boot_a = rng.choice(a, size=len(a), replace=True)
            boot_b = rng.choice(b, size=len(b), replace=True)
            boot_estimates[i] = np.median(np.subtract.outer(boot_a, boot_b).ravel())
        ci_low = np.percentile(boot_estimates, 2.5)
        ci_high = np.percentile(boot_estimates, 97.5)
        return {"diff": round(float(hl_estimate), 1), "ci_low": round(float(ci_low), 1), "ci_high": round(float(ci_high), 1)}

    def compute_diff_stats(hfno_df, no_hfno_df):
        """Compute between-group differences (HFNO - Low-flow) with 95% CIs."""
        d = {}
        d["N"] = None
        d["Age, years"] = mean_diff_ci(hfno_df["age_at_admission"], no_hfno_df["age_at_admission"])
        d["Sex, female"] = proportion_diff_ci(
            hfno_df["sex_category"], hfno_df["sex_category"] == "female",
            no_hfno_df["sex_category"], no_hfno_df["sex_category"] == "female",
        )
        d["BMI, kg/m2"] = mean_diff_ci(hfno_df["bmi"], no_hfno_df["bmi"])

        # Race/ethnicity subgroups
        for cat in ["Hispanic", "Non-Hispanic White", "Non-Hispanic Black", "Non-Hispanic Asian", "Other", "Not Reported"]:
            d[f"  {cat}"] = proportion_diff_ci(
                hfno_df["race_ethnicity"], hfno_df["race_ethnicity"] == cat,
                no_hfno_df["race_ethnicity"], no_hfno_df["race_ethnicity"] == cat,
            )

        d["Charlson Comorbidity Index, mean (SD)"] = mean_diff_ci(hfno_df["cci_score"], no_hfno_df["cci_score"])
        d["Charlson Comorbidity Index, median [IQR]"] = median_diff_ci(hfno_df["cci_score"], no_hfno_df["cci_score"])
        d["SOFA, ICU admission"] = mean_diff_ci(hfno_df["sofa_icu_admission"], no_hfno_df["sofa_icu_admission"])
        d["SOFA, extubation"] = mean_diff_ci(hfno_df["sofa_extubation"], no_hfno_df["sofa_extubation"])

        # ICU type subgroups (use HFNO categories to match table row order)
        for cat in sorted(hfno_df["icu_type"].drop_nulls().unique().to_list()):
            d[f"  {cat}"] = proportion_diff_ci(
                hfno_df["icu_type"], hfno_df["icu_type"] == cat,
                no_hfno_df["icu_type"], no_hfno_df["icu_type"] == cat,
            )

        d["PaCO2 before extubation, mmHg"] = mean_diff_ci(hfno_df["paco2_pre_extubation"], no_hfno_df["paco2_pre_extubation"])

        # Life support prior to extubation
        d["  CRRT prior"] = bool_diff_ci(hfno_df["crrt_prior"], no_hfno_df["crrt_prior"])
        d["  Vasopressor prior"] = bool_diff_ci(hfno_df["vasopressor_prior"], no_hfno_df["vasopressor_prior"])
        d["  NIPPV/CPAP prior"] = bool_diff_ci(hfno_df["nippv_cpap_prior"], no_hfno_df["nippv_cpap_prior"])
        d["  HFNO prior"] = bool_diff_ci(hfno_df["hfno_prior"], no_hfno_df["hfno_prior"])
        d["  Any life support prior"] = bool_diff_ci(hfno_df["any_life_support_prior"], no_hfno_df["any_life_support_prior"])

        # Life support at extubation
        d["  CRRT at extubation"] = bool_diff_ci(hfno_df["crrt_at_extubation"], no_hfno_df["crrt_at_extubation"])
        d["  Vasopressor at extubation"] = bool_diff_ci(hfno_df["vasopressor_at_extubation"], no_hfno_df["vasopressor_at_extubation"])
        d["  Any life support at extubation"] = bool_diff_ci(hfno_df["life_support_at_extubation"], no_hfno_df["life_support_at_extubation"])

        # Outcomes
        d["IMV duration, days"] = mean_diff_ci(hfno_df["imv_duration_days"], no_hfno_df["imv_duration_days"])
        d["ICU LOS before extubation, days"] = mean_diff_ci(hfno_df["icu_los_before_extubation_days"], no_hfno_df["icu_los_before_extubation_days"])
        d["Extubation success, 7 days"] = bool_diff_ci(hfno_df["extubation_success_7d"], no_hfno_df["extubation_success_7d"])
        d["Extubation success (strict), 7 days"] = bool_diff_ci(hfno_df["extubation_success_strict_7d"], no_hfno_df["extubation_success_strict_7d"])
        d["Death within 7 days"] = bool_diff_ci(hfno_df["death_in_7d"], no_hfno_df["death_in_7d"])
        d["NIPPV/CPAP in ICU within 7 days"] = bool_diff_ci(hfno_df["nippv_cpap_in_7d_plus_in_ICU"], no_hfno_df["nippv_cpap_in_7d_plus_in_ICU"])

        # HFNO-only variables — not meaningful to compare
        d["Definitive HFNO weaning"] = None
        d["Time to HFNO weaning, hours"] = None

        d["ICU readmission"] = bool_diff_ci(hfno_df["readmission_to_icu"], no_hfno_df["readmission_to_icu"])
        d["ICU LOS, days"] = mean_diff_ci(hfno_df["icu_los_days"], no_hfno_df["icu_los_days"])
        d["ICU mortality"] = bool_diff_ci(hfno_df["icu_mortality"], no_hfno_df["icu_mortality"])
        d["Hospital LOS, days"] = mean_diff_ci(hfno_df["hospital_los_days"], no_hfno_df["hospital_los_days"])
        d["Hospital mortality"] = bool_diff_ci(hfno_df["hospital_mortality"], no_hfno_df["hospital_mortality"])

        return d
    return (compute_diff_stats,)


@app.cell
def _(compute_diff_stats, compute_group_stats, df, format_stat, pl):
    # Generate Table 1 for each group
    hfno_df = df.filter(pl.col("is_hfno_cohort"))
    no_hfno_df = df.filter(~pl.col("is_hfno_cohort"))

    stats_hfno = compute_group_stats(hfno_df, is_hfno_group=True)
    stats_no_hfno = compute_group_stats(no_hfno_df, is_hfno_group=False)
    stats_overall = compute_group_stats(df, is_hfno_group=True)
    stats_diff = compute_diff_stats(hfno_df, no_hfno_df)

    # Build table rows with formatted strings for CSV display
    _variables = list(stats_hfno.keys())
    table_rows = []
    for _var in _variables:
        table_rows.append({
            "variable": _var,
            "hfno_after_extubation": format_stat(stats_hfno[_var]),
            "low_flow_after_extubation": format_stat(stats_no_hfno[_var]),
            "overall": format_stat(stats_overall[_var]),
            "difference_95ci": format_stat(stats_diff.get(_var, None)),
        })

    print(f"Table 1 generated with {len(table_rows)} rows")
    for _row in table_rows:
        print(f"  {_row['variable']}: HFNO={_row['hfno_after_extubation']}, Low Flow={_row['low_flow_after_extubation']}, Overall={_row['overall']}, Diff={_row['difference_95ci']}")
    return stats_diff, stats_hfno, stats_no_hfno, stats_overall, table_rows


@app.cell
def _(Path, SITE, json, pd, stats_diff, stats_hfno, stats_no_hfno, stats_overall, table_rows):
    # Save outputs
    output_dir = Path(__file__).parent.parent / "output_to_share"
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV — formatted strings for tabular display
    for _row in table_rows:
        _row["site"] = SITE
    table_df = pd.DataFrame(table_rows)[["variable", "site", "hfno_after_extubation", "low_flow_after_extubation", "overall", "difference_95ci"]]
    csv_path = output_dir / "table1.csv"
    table_df.to_csv(csv_path, index=False)
    print(f"Table 1 CSV saved to: {csv_path}")

    # JSON — structured numeric format for downstream aggregation
    _variables = list(stats_hfno.keys())
    json_output = {"site": SITE}
    for _var in _variables:
        json_output[_var] = {
            "hfno_after_extubation": stats_hfno[_var],
            "low_flow_after_extubation": stats_no_hfno[_var],
            "overall": stats_overall[_var],
            "difference_95ci": stats_diff.get(_var, None),
        }
    json_path = output_dir / "table1.json"
    with open(json_path, "w") as _f:
        json.dump(json_output, _f, indent=2)
    print(f"Table 1 JSON saved to: {json_path}")

    table_df
    return


if __name__ == "__main__":
    app.run()
