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
    from clifpy.tables import HospitalDiagnosis
    from clifpy import calculate_cci
    return HospitalDiagnosis, Path, calculate_cci, json, mo, np, pd, pl


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
        pl.when(pl.col("ethnicity_category") == "Hispanic")
        .then(pl.lit("Hispanic"))
        .when((pl.col("ethnicity_category") == "Unknown") | (pl.col("race_category") == "Unknown"))
        .then(pl.lit("Not Reported"))
        .when(pl.col("race_category") == "White")
        .then(pl.lit("Non-Hispanic White"))
        .when(pl.col("race_category") == "Black or African American")
        .then(pl.lit("Non-Hispanic Black"))
        .when(pl.col("race_category") == "Asian")
        .then(pl.lit("Non-Hispanic Asian"))
        .otherwise(pl.lit("Other"))
        .alias("race_ethnicity")
    )

    # ICU type grouped
    df = df.with_columns(
        pl.when(pl.col("icu_type").is_in(["cardiac_icu", "cvicu_icu"]))
        .then(pl.lit("Cardiac"))
        .when(pl.col("icu_type") == "surgical_icu")
        .then(pl.lit("Surgical"))
        .when(pl.col("icu_type").is_in(["medical_icu", "general_icu"]))
        .then(pl.lit("Medical"))
        .when(pl.col("icu_type") == "mixed_neuro_icu")
        .then(pl.lit("Neuro"))
        .otherwise(pl.lit("Other"))
        .alias("icu_type_grouped")
    )

    print(f"Derived columns computed. Shape: {df.shape}")
    print(f"Race/ethnicity distribution:\n{df['race_ethnicity'].value_counts().sort('count', descending=True)}")
    print(f"ICU type distribution:\n{df['icu_type_grouped'].value_counts().sort('count', descending=True)}")
    return (df,)


@app.cell
def _(np, pl):
    # Helper functions for Table 1 statistics
    def mean_sd(series):
        """Format as 'mean (SD)' string."""
        vals = series.drop_nulls().filter(series.drop_nulls().is_finite())
        if len(vals) == 0:
            return "NA"
        m = vals.mean()
        s = vals.std()
        return f"{m:.1f} ({s:.1f})"

    def median_iqr(series):
        """Format as 'median [Q1, Q3]' string."""
        vals = series.drop_nulls()
        if len(vals) == 0:
            return "NA"
        med = vals.median()
        q1 = np.percentile(vals.to_numpy(), 25)
        q3 = np.percentile(vals.to_numpy(), 75)
        return f"{med:.1f} [{q1:.1f}, {q3:.1f}]"

    def n_pct(series, condition=None):
        """Format as 'n (pct%)' string."""
        if condition is not None:
            total = len(series)
            n = condition.sum()
        else:
            # Assume boolean series
            total = len(series)
            n = series.sum()
        if total == 0:
            return "0 (0.0%)"
        pct = n / total * 100
        return f"{n} ({pct:.1f}%)"

    def compute_group_stats(group_df, is_hfno_group):
        """Compute all Table 1 statistics for one group."""
        N = len(group_df)
        stats = {}

        stats["N"] = str(N)
        stats["Age, years"] = mean_sd(group_df["age_at_admission"])
        stats["Sex, female"] = n_pct(
            group_df["sex_category"],
            group_df["sex_category"] == "Female"
        )
        stats["BMI, kg/m2"] = mean_sd(group_df["bmi"])

        # Race/ethnicity subgroups
        re_col = group_df["race_ethnicity"]
        for cat in ["Hispanic", "Non-Hispanic White", "Non-Hispanic Black", "Non-Hispanic Asian", "Other", "Not Reported"]:
            stats[f"  {cat}"] = n_pct(re_col, re_col == cat)

        stats["Charlson Comorbidity Index"] = mean_sd(group_df["cci_score"])
        stats["SOFA, ICU admission"] = mean_sd(group_df["sofa_icu_admission"])
        stats["SOFA, extubation"] = mean_sd(group_df["sofa_extubation"])

        # ICU type subgroups
        icu_col = group_df["icu_type_grouped"]
        for cat in ["Cardiac", "Surgical", "Medical", "Neuro"]:
            stats[f"  {cat} ICU"] = n_pct(icu_col, icu_col == cat)

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
            stats["Definitive HFNO weaning"] = "NA"
            stats["Time to HFNO weaning, hours"] = "NA"

        stats["ICU readmission"] = n_pct(group_df["readmission_to_icu"])
        stats["ICU LOS, days"] = mean_sd(group_df["icu_los_days"])
        stats["ICU mortality"] = n_pct(group_df["icu_mortality"])
        stats["Hospital LOS, days"] = mean_sd(group_df["hospital_los_days"])
        stats["Hospital mortality"] = n_pct(group_df["hospital_mortality"])

        return stats
    return (compute_group_stats,)


@app.cell
def _(compute_group_stats, df, pl):
    # Generate Table 1 for each group
    hfno_df = df.filter(pl.col("is_hfno_cohort"))
    no_hfno_df = df.filter(~pl.col("is_hfno_cohort"))

    stats_hfno = compute_group_stats(hfno_df, is_hfno_group=True)
    stats_no_hfno = compute_group_stats(no_hfno_df, is_hfno_group=False)
    stats_overall = compute_group_stats(df, is_hfno_group=True)

    # Build table rows
    _variables = list(stats_hfno.keys())
    table_rows = []
    for _var in _variables:
        table_rows.append({
            "variable": _var,
            "hfno_after_extubation": stats_hfno[_var],
            "low_flow_after_extubation": stats_no_hfno[_var],
            "overall": stats_overall[_var],
        })

    print(f"Table 1 generated with {len(table_rows)} rows")
    for _row in table_rows:
        print(f"  {_row['variable']}: HFNO={_row['hfno_after_extubation']}, Low Flow={_row['low_flow_after_extubation']}, Overall={_row['overall']}")
    return (table_rows,)


@app.cell
def _(Path, SITE, json, pd, table_rows):
    # Save outputs
    output_dir = Path(__file__).parent.parent / "output_to_share"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add site column
    for _row in table_rows:
        _row["site"] = SITE

    # CSV
    table_df = pd.DataFrame(table_rows)[["variable", "site", "hfno_after_extubation", "low_flow_after_extubation", "overall"]]
    csv_path = output_dir / "table1.csv"
    table_df.to_csv(csv_path, index=False)
    print(f"Table 1 CSV saved to: {csv_path}")

    # JSON
    json_path = output_dir / "table1.json"
    with open(json_path, "w") as _f:
        json.dump(table_rows, _f, indent=2)
    print(f"Table 1 JSON saved to: {json_path}")

    table_df
    return


if __name__ == "__main__":
    app.run()
