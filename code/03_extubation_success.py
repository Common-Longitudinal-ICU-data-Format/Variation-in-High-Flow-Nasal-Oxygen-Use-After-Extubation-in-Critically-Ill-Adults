import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import polars as pl
    import json
    from pathlib import Path
    from clifpy.tables import Patient
    return Path, Patient, json, mo, pd, pl


@app.cell
def _(mo):
    mo.md("""
    # 03 Extubation Success: 7-Day Outcome Flags

    Adds binary extubation success flags to the cohort.

    - **`extubation_success_7d`** = alive + no reintubation (no IMV) within 7 days of extubation.
      All non-IMV respiratory devices (HFNC, NIPPV, CPAP, nasal cannula, etc.) are allowed.
    - **`extubation_success_strict_7d`** = alive + no reintubation + **no NIPPV/CPAP** within 7 days.
      Only HFNC, nasal cannula, and room air are allowed.
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
    return DATA_DIR, FILETYPE, TIMEZONE


@app.cell
def _(Path, pd, pl):
    # Load cohort
    cohort_pd = pd.read_parquet(
        Path(__file__).parent.parent / "output" / "cohort_inclusion.parquet"
    )

    # Strip tz from datetime columns
    for col in cohort_pd.select_dtypes(include=["datetimetz"]).columns:
        cohort_pd[col] = cohort_pd[col].dt.tz_localize(None)

    cohort = pl.from_pandas(cohort_pd)
    del cohort_pd

    # Add 7-day window end
    cohort = cohort.with_columns(
        (pl.col("extubation_time") + pl.duration(hours=168)).alias("window_end_7d")
    )

    cohort_ids = cohort["hospitalization_id"].to_list()
    cohort_patient_ids = cohort["patient_id"].unique().to_list()

    print(f"Cohort loaded: {len(cohort)} hospitalizations, {len(cohort_patient_ids)} patients")
    print(f"Columns: {cohort.columns}")
    return cohort, cohort_ids, cohort_patient_ids


@app.cell
def _(Path, cohort_ids, pd, pl):
    # Load resp waterfall
    resp_pd = pd.read_parquet(
        Path(__file__).parent.parent / "output" / "resp_support_waterfall.parquet"
    )

    # Strip tz from recorded_dttm
    if resp_pd["recorded_dttm"].dt.tz is not None:
        resp_pd["recorded_dttm"] = resp_pd["recorded_dttm"].dt.tz_localize(None)

    resp = pl.from_pandas(resp_pd)
    del resp_pd

    # Filter to cohort
    resp = resp.filter(pl.col("hospitalization_id").is_in(cohort_ids))

    print(f"Resp waterfall loaded: {len(resp)} rows for {resp['hospitalization_id'].n_unique()} hospitalizations")
    return (resp,)


@app.cell
def _(
    DATA_DIR,
    FILETYPE,
    Patient,
    TIMEZONE,
    cohort,
    cohort_patient_ids,
    pd,
    pl,
):
    # Load Patient table for death_dttm
    patient_table = Patient.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={"patient_id": cohort_patient_ids},
    )
    patient_pd = patient_table.df[["patient_id", "death_dttm"]].copy()
    patient_pd["death_dttm"] = pd.to_datetime(patient_pd["death_dttm"], errors="coerce")
    if patient_pd["death_dttm"].dt.tz is not None:
        patient_pd["death_dttm"] = patient_pd["death_dttm"].dt.tz_localize(None)

    patient_death = pl.from_pandas(patient_pd)
    del patient_pd

    # Join to cohort on patient_id to get death_dttm per hospitalization
    death_by_hosp = (
        cohort.select(["hospitalization_id", "patient_id"])
        .join(patient_death, on="patient_id", how="left")
        .select(["hospitalization_id", "death_dttm"])
    )

    print(f"Death info: {death_by_hosp['death_dttm'].is_not_null().sum()} hospitalizations with death_dttm")
    return (death_by_hosp,)


@app.cell
def _(cohort, pl, resp):
    # Detect reintubation (any IMV) within 7-day window post-extubation
    reintubation_flag = (
        resp
        .join(
            cohort.select(["hospitalization_id", "extubation_time", "window_end_7d"]),
            on="hospitalization_id",
            how="inner",
        )
        .filter(
            (pl.col("device_category") == "imv")
            & (pl.col("recorded_dttm") > pl.col("extubation_time"))
            & (pl.col("recorded_dttm") <= pl.col("window_end_7d"))
        )
        .select("hospitalization_id")
        .unique()
        .with_columns(pl.lit(True).alias("reintubation_in_7d"))
    )

    print(f"Reintubation within 7 days: {len(reintubation_flag)} hospitalizations")
    return (reintubation_flag,)


@app.cell
def _(cohort, pl, resp):
    # Detect NIPPV/CPAP use within 7-day window post-extubation
    nippv_cpap_flag = (
        resp
        .join(
            cohort.select(["hospitalization_id", "extubation_time", "window_end_7d"]),
            on="hospitalization_id",
            how="inner",
        )
        .filter(
            pl.col("device_category").is_in(["nippv", "cpap"])
            & (pl.col("recorded_dttm") > pl.col("extubation_time"))
            & (pl.col("recorded_dttm") <= pl.col("window_end_7d"))
        )
        .select("hospitalization_id")
        .unique()
        .with_columns(pl.lit(True).alias("nippv_cpap_in_7d"))
    )

    print(f"NIPPV/CPAP within 7 days: {len(nippv_cpap_flag)} hospitalizations")
    return (nippv_cpap_flag,)


@app.cell
def _(cohort, pl, resp):
    # Detect NIPPV/CPAP use within ICU stays only (narrower than full 7-day window)
    # Window 1: extubation_time -> icu_end (first ICU stay)
    # Window 2: readmission_icu_start -> window_end_7d (readmission ICU stay, if any)
    resp_with_cohort = resp.join(
        cohort.select([
            "hospitalization_id", "extubation_time", "icu_end",
            "readmission_to_icu", "readmission_icu_start", "window_end_7d",
        ]),
        on="hospitalization_id",
        how="inner",
    ).filter(pl.col("device_category").is_in(["nippv", "cpap"]))

    # Window 1: first ICU stay (extubation_time, min(icu_end, window_end_7d)]
    w1 = resp_with_cohort.filter(
        (pl.col("recorded_dttm") > pl.col("extubation_time"))
        & (pl.col("recorded_dttm") <= pl.col("icu_end"))
        & (pl.col("recorded_dttm") <= pl.col("window_end_7d"))
    ).select("hospitalization_id").unique()

    # Window 2: readmission ICU stay [readmission_icu_start, window_end_7d]
    w2 = resp_with_cohort.filter(
        pl.col("readmission_to_icu")
        & (pl.col("recorded_dttm") >= pl.col("readmission_icu_start"))
        & (pl.col("recorded_dttm") <= pl.col("window_end_7d"))
    ).select("hospitalization_id").unique()

    nippv_cpap_icu_flag = (
        pl.concat([w1, w2])
        .unique()
        .with_columns(pl.lit(True).alias("nippv_cpap_in_7d_plus_in_ICU"))
    )

    print(f"NIPPV/CPAP in ICU within 7 days: {len(nippv_cpap_icu_flag)} hospitalizations")
    return (nippv_cpap_icu_flag,)


@app.cell
def _(cohort, death_by_hosp, pl):
    # Detect death within 7-day window post-extubation
    death_flag = (
        cohort.select(["hospitalization_id", "extubation_time", "window_end_7d"])
        .join(death_by_hosp, on="hospitalization_id", how="left")
        .filter(
            pl.col("death_dttm").is_not_null()
            & (pl.col("death_dttm") > pl.col("extubation_time"))
            & (pl.col("death_dttm") <= pl.col("window_end_7d"))
        )
        .select("hospitalization_id")
        .unique()
        .with_columns(pl.lit(True).alias("death_in_7d"))
    )

    print(f"Death within 7 days: {len(death_flag)} hospitalizations")
    return (death_flag,)


@app.cell
def _(
    cohort,
    death_flag,
    nippv_cpap_flag,
    nippv_cpap_icu_flag,
    pl,
    reintubation_flag,
):
    # Build extubation success flags and merge to cohort
    result = (
        cohort
        .join(reintubation_flag, on="hospitalization_id", how="left")
        .join(death_flag, on="hospitalization_id", how="left")
        .join(nippv_cpap_flag, on="hospitalization_id", how="left")
        .join(nippv_cpap_icu_flag, on="hospitalization_id", how="left")
        .with_columns([
            pl.col("reintubation_in_7d").fill_null(False),
            pl.col("death_in_7d").fill_null(False),
            pl.col("nippv_cpap_in_7d").fill_null(False),
            pl.col("nippv_cpap_in_7d_plus_in_ICU").fill_null(False),
        ])
        .with_columns([
            (~pl.col("death_in_7d") & ~pl.col("reintubation_in_7d")).alias("extubation_success_7d"),
            (~pl.col("death_in_7d") & ~pl.col("reintubation_in_7d") & ~pl.col("nippv_cpap_in_7d")).alias("extubation_success_strict_7d"),
        ])
    )

    N = len(result)
    n_reintub = result["reintubation_in_7d"].sum()
    n_death = result["death_in_7d"].sum()
    n_nippv_cpap = result["nippv_cpap_in_7d"].sum()
    n_nippv_cpap_icu = result["nippv_cpap_in_7d_plus_in_ICU"].sum()
    n_success = result["extubation_success_7d"].sum()
    n_success_strict = result["extubation_success_strict_7d"].sum()

    print("=== Column Descriptions ===")
    print(f"  window_end_7d: End of 7-day observation window (extubation_time + 168h)")
    print(f"  reintubation_in_7d: Any IMV device recorded within 7 days post-extubation")
    print(f"    -> {n_reintub}/{N} ({n_reintub / N * 100:.1f}%)")
    print(f"  death_in_7d: Death occurred within 7 days post-extubation")
    print(f"    -> {n_death}/{N} ({n_death / N * 100:.1f}%)")
    print(f"  nippv_cpap_in_7d: Any NIPPV/CPAP device recorded within 7 days post-extubation")
    print(f"    -> {n_nippv_cpap}/{N} ({n_nippv_cpap / N * 100:.1f}%)")
    print(f"  nippv_cpap_in_7d_plus_in_ICU: Any NIPPV/CPAP within ICU stays in 7-day window")
    print(f"    -> {n_nippv_cpap_icu}/{N} ({n_nippv_cpap_icu / N * 100:.1f}%)")
    print(f"  extubation_success_7d: Alive + no reintubation within 7 days post-extubation")
    print(f"    -> {n_success}/{N} ({n_success / N * 100:.1f}%)")
    print(f"  extubation_success_strict_7d: Alive + no reintubation + no NIPPV/CPAP within 7 days")
    print(f"    -> {n_success_strict}/{N} ({n_success_strict / N * 100:.1f}%)")
    return (result,)


@app.cell
def _(Path, result):
    # Save output
    output_path = Path(__file__).parent.parent / "output" / "cohort_with_extubation_success.parquet"
    result.write_parquet(output_path)

    print(f"Saved to: {output_path}")
    print(f"Shape: {result.shape}")
    print(f"Columns: {result.columns}")
    return


if __name__ == "__main__":
    app.run()
