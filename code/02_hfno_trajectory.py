import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import polars as pl
    import json
    import matplotlib.pyplot as plt
    from pathlib import Path
    from clifpy.tables import Adt, Vitals, HospitalDiagnosis, Patient, MedicationAdminContinuous, Hospitalization
    from clifpy import calculate_cci
    return (
        Adt,
        HospitalDiagnosis,
        Hospitalization,
        MedicationAdminContinuous,
        Path,
        Patient,
        Vitals,
        calculate_cci,
        json,
        mo,
        pd,
        pl,
        plt,
    )


@app.cell
def _(mo):
    mo.md("""
    # 02 HFNO Trajectory: Longitudinal Panel Dataset

    One row per (hospitalization, 4-hour window) post-extubation.
    42 possible windows (0–168h = 7 days), truncated at outcome.
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

    # Compute BMI
    cohort = cohort.with_columns(
        (pl.col("weight_kg") / (pl.col("height_cm") / 100).pow(2)).alias("bmi")
    )

    cohort_ids = cohort["hospitalization_id"].to_list()
    cohort_patient_ids = cohort["patient_id"].unique().to_list()

    print(f"Cohort loaded: {len(cohort)} hospitalizations, {len(cohort_patient_ids)} patients")
    print(f"Columns: {cohort.columns}")
    return cohort, cohort_ids, cohort_patient_ids


@app.cell
def _(Adt, DATA_DIR, FILETYPE, TIMEZONE, cohort, cohort_ids, pd, pl):
    # Load ADT table filtered to cohort ICU stays
    adt_table = Adt.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={"hospitalization_id": cohort_ids, "location_category": ["icu"]},
    )

    # Convert to pandas, strip timezone, then to polars
    adt_pd = adt_table.df
    for _col in adt_pd.select_dtypes(include=["datetimetz"]).columns:
        adt_pd[_col] = adt_pd[_col].dt.tz_localize(None)
    adt = pl.from_pandas(adt_pd)
    del adt_pd

    # Join cohort extubation_time, then find the ADT row containing extubation
    adt_ext = (
        adt.join(
            cohort.select(["hospitalization_id", "extubation_time"]),
            on="hospitalization_id",
            how="inner",
        )
        .filter(
            (pl.col("extubation_time") >= pl.col("in_dttm"))
            & (pl.col("extubation_time") <= pl.col("out_dttm"))
        )
        # For ties (overlapping ADT rows), take the latest in_dttm (most specific ICU)
        .sort("in_dttm", descending=True)
        .group_by("hospitalization_id")
        .first()
        .select([
            "hospitalization_id",
            pl.col("location_type").alias("extubation_icu_type"),
            "hospital_id",
        ])
    )

    print(f"ADT extubation ICU mapping: {len(adt_ext)} / {len(cohort)} hospitalizations matched")
    print(f"  extubation_icu_type values: {adt_ext['extubation_icu_type'].value_counts().sort('count', descending=True)}")
    return (adt_ext,)


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
def _(DATA_DIR, FILETYPE, TIMEZONE, Vitals, cohort_ids, pl):
    # Load SpO2 and RR vitals
    vitals_table = Vitals.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={
            "hospitalization_id": cohort_ids,
            "vital_category": ["spo2", "respiratory_rate"],
        },
    )
    vitals_pd = vitals_table.df.copy()

    # Strip tz
    if vitals_pd["recorded_dttm"].dt.tz is not None:
        vitals_pd["recorded_dttm"] = vitals_pd["recorded_dttm"].dt.tz_localize(None)

    vitals = pl.from_pandas(vitals_pd)
    del vitals_pd

    print(f"Vitals loaded: {len(vitals)} rows")
    print(f"  Categories: {vitals['vital_category'].unique().to_list()}")
    return (vitals,)


@app.cell
def _(
    DATA_DIR,
    FILETYPE,
    HospitalDiagnosis,
    TIMEZONE,
    calculate_cci,
    cohort_ids,
    pl,
):
    # Compute CCI
    hosp_diag = HospitalDiagnosis.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={"hospitalization_id": cohort_ids},
    )

    cci_result = calculate_cci(hosp_diag)
    # calculate_cci may return pandas or polars — normalize to polars
    if not isinstance(cci_result, pl.DataFrame):
        cci_result = pl.from_pandas(cci_result)
    cci_df = cci_result.select(["hospitalization_id", "cci_score"])

    print(f"CCI computed for {len(cci_df)} hospitalizations")
    return (cci_df,)


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
def _(cohort, pl):
    # Build time window scaffold: 42 windows of 4h each (0-168h post-extubation)
    WINDOW_HOURS = 4
    N_WINDOWS = 42  # 168h / 4h

    window_labels = [f"{i * WINDOW_HOURS}-{(i + 1) * WINDOW_HOURS}" for i in range(N_WINDOWS)]

    # Explode: one row per (hospitalization, window)
    hosp_times = cohort.select(["hospitalization_id", "extubation_time"])

    scaffold_rows = []
    for i in range(N_WINDOWS):
        start_offset = i * WINDOW_HOURS
        end_offset = (i + 1) * WINDOW_HOURS
        scaffold_rows.append(
            hosp_times.with_columns([
                (pl.col("extubation_time") + pl.duration(hours=start_offset)).alias("window_start"),
                (pl.col("extubation_time") + pl.duration(hours=end_offset)).alias("window_end"),
                pl.lit(window_labels[i]).alias("hour_window"),
                pl.lit(i).alias("window_idx"),
            ])
        )

    scaffold = pl.concat(scaffold_rows).sort(["hospitalization_id", "window_idx"])

    print(f"Scaffold: {len(scaffold)} rows ({scaffold['hospitalization_id'].n_unique()} hosps x up to {N_WINDOWS} windows)")
    return (scaffold,)


@app.cell
def _(cohort, death_by_hosp, pl, resp):
    # Detect outcomes per window

    # --- Event times per hospitalization ---

    # Reintubation: first IMV after extubation
    reintubation = (
        resp
        .join(
            cohort.select(["hospitalization_id", "extubation_time"]),
            on="hospitalization_id",
            how="inner",
        )
        .filter(
            (pl.col("device_category") == "imv")
            & (pl.col("recorded_dttm") > pl.col("extubation_time"))
        )
        .sort(["hospitalization_id", "recorded_dttm"])
        .group_by("hospitalization_id")
        .first()
        .select([
            "hospitalization_id",
            pl.col("recorded_dttm").alias("reintubation_time"),
        ])
    )

    # NIPPV/CPAP: first occurrence after extubation
    nippv_cpap = (
        resp
        .join(
            cohort.select(["hospitalization_id", "extubation_time"]),
            on="hospitalization_id",
            how="inner",
        )
        .filter(
            (pl.col("device_category").is_in(["nippv", "cpap"]))
            & (pl.col("recorded_dttm") > pl.col("extubation_time"))
        )
        .sort(["hospitalization_id", "recorded_dttm"])
        .group_by("hospitalization_id")
        .first()
        .select([
            "hospitalization_id",
            pl.col("recorded_dttm").alias("nippv_cpap_time"),
        ])
    )

    # Death time
    death_times = death_by_hosp.filter(pl.col("death_dttm").is_not_null()).rename(
        {"death_dttm": "death_time"}
    )

    # ICU discharge alive: icu_end if patient did not die before icu_end
    icu_discharge_alive = (
        cohort.select(["hospitalization_id", "icu_end"])
        .join(death_by_hosp, on="hospitalization_id", how="left")
        .filter(
            pl.col("death_dttm").is_null()
            | (pl.col("death_dttm") > pl.col("icu_end"))
        )
        .select([
            "hospitalization_id",
            pl.col("icu_end").alias("icu_discharge_alive_time"),
        ])
    )

    # HFNO-free 48h: first time patient is off HFNO for 48 consecutive hours
    # while alive and still in ICU, within the 7-day window
    _post_ext_resp = (
        resp
        .join(
            cohort.select(["hospitalization_id", "extubation_time"]),
            on="hospitalization_id",
            how="inner",
        )
        .filter(
            (pl.col("recorded_dttm") > pl.col("extubation_time"))
            & (pl.col("recorded_dttm") <= pl.col("extubation_time") + pl.duration(hours=168))
        )
        .with_columns(
            (pl.col("device_category") == "high flow nc").alias("is_hfno"),
        )
        .sort(["hospitalization_id", "recorded_dttm"])
    )

    # Identify patients who NEVER had HFNO post-extubation within 7d
    _hosp_with_hfno = (
        _post_ext_resp
        .filter(pl.col("is_hfno"))
        .select("hospitalization_id")
        .unique()
    )
    _hosp_never_hfno = (
        cohort.select("hospitalization_id")
        .join(_hosp_with_hfno, on="hospitalization_id", how="anti")
        .join(
            cohort.select(["hospitalization_id", "extubation_time", "icu_end"]),
            on="hospitalization_id",
            how="left",
        )
        .join(death_by_hosp, on="hospitalization_id", how="left")
        .with_columns(
            (pl.col("extubation_time") + pl.duration(hours=48)).alias("candidate_time"),
        )
        .filter(
            # Still in ICU at candidate_time
            (pl.col("candidate_time") <= pl.col("icu_end"))
            # Alive at candidate_time
            & (
                pl.col("death_dttm").is_null()
                | (pl.col("death_dttm") > pl.col("candidate_time"))
            )
            # Within 7-day window (always true: extubation + 48h <= extubation + 168h)
        )
        .select([
            "hospitalization_id",
            pl.col("candidate_time").alias("hfno_free_48h_time"),
        ])
    )

    # For patients WITH HFNO: find transitions off HFNO, check 48h gap
    # Get all HFNO records to check gaps against
    _hfno_records = (
        _post_ext_resp
        .filter(pl.col("is_hfno"))
        .select(["hospitalization_id", "recorded_dttm"])
    )

    # Detect off-transitions: record is HFNO, next record (same patient) is NOT HFNO
    # The off-start is the time of the first non-HFNO record after an HFNO record
    _with_next = (
        _post_ext_resp
        .with_columns(
            pl.col("is_hfno").shift(-1).over("hospitalization_id").alias("next_is_hfno"),
            pl.col("recorded_dttm").shift(-1).over("hospitalization_id").alias("next_time"),
        )
    )

    # Off-transition points: current is HFNO, next is not (or is last record)
    _off_starts = (
        _with_next
        .filter(
            pl.col("is_hfno")
            & (pl.col("next_is_hfno") == False)
        )
        .select([
            "hospitalization_id",
            pl.col("next_time").alias("off_start"),
            "extubation_time",
        ])
    )

    # Also include extubation_time as an off-start for patients whose first
    # post-extubation record is NOT HFNO (they start off-HFNO)
    _first_rec = (
        _post_ext_resp
        .group_by("hospitalization_id")
        .first()
        .filter(~pl.col("is_hfno"))
        .select([
            "hospitalization_id",
            pl.col("extubation_time").alias("off_start"),
            "extubation_time",
        ])
    )

    _all_off_starts = pl.concat([_off_starts, _first_rec])

    # Also handle: last record is HFNO — the "off" starts at the time AFTER the last HFNO record
    # (captured by _off_starts when next_is_hfno==False; if last record is HFNO and has no next,
    #  next_is_hfno is null, so we handle that separately)
    _last_hfno_off = (
        _with_next
        .filter(
            pl.col("is_hfno")
            & pl.col("next_is_hfno").is_null()  # last record for this patient
        )
        .select([
            "hospitalization_id",
            # Off starts at this record's time (it's the last HFNO, nothing after)
            pl.col("recorded_dttm").alias("off_start"),
            "extubation_time",
        ])
    )
    _all_off_starts = pl.concat([_all_off_starts, _last_hfno_off])

    # For each off-start, check if any HFNO record appears in [off_start, off_start + 48h)
    _off_with_candidate = _all_off_starts.with_columns(
        (pl.col("off_start") + pl.duration(hours=48)).alias("candidate_time"),
    )

    # Cross-join with HFNO records to find any HFNO in the gap window
    _off_with_hfno_check = (
        _off_with_candidate
        .join(_hfno_records, on="hospitalization_id", how="left", suffix="_hfno")
        .filter(
            pl.col("recorded_dttm").is_not_null()
            & (pl.col("recorded_dttm") >= pl.col("off_start"))
            & (pl.col("recorded_dttm") < pl.col("candidate_time"))
        )
        .select(["hospitalization_id", "off_start"])
        .unique()
    )

    # Off-starts with NO HFNO in the 48h window = valid 48h-free periods
    _valid_off = (
        _off_with_candidate
        .join(_off_with_hfno_check, on=["hospitalization_id", "off_start"], how="anti")
    )

    # Validate: patient alive, still in ICU, within 7-day window
    _valid_off = (
        _valid_off
        .join(
            cohort.select(["hospitalization_id", "icu_end"]),
            on="hospitalization_id",
            how="left",
        )
        .join(death_by_hosp, on="hospitalization_id", how="left")
        .filter(
            # candidate_time within 7-day window
            (pl.col("candidate_time") <= pl.col("extubation_time") + pl.duration(hours=168))
            # Still in ICU
            & (pl.col("candidate_time") <= pl.col("icu_end"))
            # Alive
            & (
                pl.col("death_dttm").is_null()
                | (pl.col("death_dttm") > pl.col("candidate_time"))
            )
        )
    )

    # Take the earliest valid hfno_free_48h_time per patient
    _hfno_free_from_hfno = (
        _valid_off
        .sort(["hospitalization_id", "candidate_time"])
        .group_by("hospitalization_id")
        .first()
        .select([
            "hospitalization_id",
            pl.col("candidate_time").alias("hfno_free_48h_time"),
        ])
    )

    # Combine never-HFNO and HFNO-then-free patients
    hfno_free_48h = pl.concat([_hosp_never_hfno, _hfno_free_from_hfno])

    # Combine all event times
    events = (
        cohort.select("hospitalization_id")
        .join(reintubation, on="hospitalization_id", how="left")
        .join(nippv_cpap, on="hospitalization_id", how="left")
        .join(death_times, on="hospitalization_id", how="left")
        .join(icu_discharge_alive, on="hospitalization_id", how="left")
        .join(hfno_free_48h, on="hospitalization_id", how="left")
        .with_columns(
            pl.min_horizontal("icu_discharge_alive_time", "hfno_free_48h_time").alias("success_time"),
        )
    )

    print(f"Events summary:")
    print(f"  Reintubation: {events['reintubation_time'].is_not_null().sum()}")
    print(f"  NIPPV/CPAP: {events['nippv_cpap_time'].is_not_null().sum()}")
    print(f"  Death: {events['death_time'].is_not_null().sum()}")
    print(f"  ICU discharge alive: {events['icu_discharge_alive_time'].is_not_null().sum()}")
    print(f"  HFNO-free 48h: {hfno_free_48h['hfno_free_48h_time'].is_not_null().sum()}")
    print(f"  Success (discharge OR 48h HFNO-free): {events['success_time'].is_not_null().sum()}")
    return (events,)


@app.cell
def _(events, pl, scaffold):
    # Compute outcome columns per window and truncate

    # Join events to scaffold
    panel_outcomes = scaffold.join(events, on="hospitalization_id", how="left")

    # --- outcome_death_reintubate ---
    # event_time = min(reintubation_time, death_time)
    # 1 = event in window, 2 = success (ICU discharge alive OR 48h HFNO-free) in window, 0 = censored
    panel_outcomes = panel_outcomes.with_columns(
        pl.min_horizontal("reintubation_time", "death_time").alias("event_time_dr"),
    )

    panel_outcomes = panel_outcomes.with_columns(
        pl.when(
            pl.col("event_time_dr").is_not_null()
            & (pl.col("event_time_dr") >= pl.col("window_start"))
            & (pl.col("event_time_dr") < pl.col("window_end"))
        )
        .then(pl.lit(1))
        .when(
            pl.col("success_time").is_not_null()
            & (pl.col("success_time") >= pl.col("window_start"))
            & (pl.col("success_time") < pl.col("window_end"))
        )
        .then(pl.lit(2))
        .otherwise(pl.lit(0))
        .alias("outcome_death_reintubate")
    )

    # --- outcome_death_reintubate_nippv_cpap ---
    # event_time = min(reintubation_time, death_time, nippv_cpap_time)
    panel_outcomes = panel_outcomes.with_columns(
        pl.min_horizontal("reintubation_time", "death_time", "nippv_cpap_time").alias("event_time_drnc"),
    )

    panel_outcomes = panel_outcomes.with_columns(
        pl.when(
            pl.col("event_time_drnc").is_not_null()
            & (pl.col("event_time_drnc") >= pl.col("window_start"))
            & (pl.col("event_time_drnc") < pl.col("window_end"))
        )
        .then(pl.lit(1))
        .when(
            pl.col("success_time").is_not_null()
            & (pl.col("success_time") >= pl.col("window_start"))
            & (pl.col("success_time") < pl.col("window_end"))
        )
        .then(pl.lit(2))
        .otherwise(pl.lit(0))
        .alias("outcome_death_reintubate_nippv_cpap")
    )

    # --- Truncate: keep rows up to and including the window with terminal event (1 or 2) ---
    # For outcome_death_reintubate: find the first window with value 1 or 2
    terminal_window_dr = (
        panel_outcomes
        .filter(pl.col("outcome_death_reintubate") > 0)
        .group_by("hospitalization_id")
        .agg(pl.col("window_idx").min().alias("terminal_window_dr"))
    )

    terminal_window_drnc = (
        panel_outcomes
        .filter(pl.col("outcome_death_reintubate_nippv_cpap") > 0)
        .group_by("hospitalization_id")
        .agg(pl.col("window_idx").min().alias("terminal_window_drnc"))
    )

    panel_outcomes = (
        panel_outcomes
        .join(terminal_window_dr, on="hospitalization_id", how="left")
        .join(terminal_window_drnc, on="hospitalization_id", how="left")
    )

    # Use the earlier of the two terminal windows for truncation
    panel_outcomes = panel_outcomes.with_columns(
        pl.min_horizontal("terminal_window_dr", "terminal_window_drnc").alias("truncate_at")
    )

    # Keep rows where window_idx <= truncate_at (or all rows if no terminal event)
    panel_truncated = panel_outcomes.filter(
        pl.col("truncate_at").is_null()
        | (pl.col("window_idx") <= pl.col("truncate_at"))
    )

    # Drop helper columns
    outcomes = panel_truncated.select([
        "hospitalization_id",
        "window_idx",
        "hour_window",
        "window_start",
        "window_end",
        "outcome_death_reintubate",
        "outcome_death_reintubate_nippv_cpap",
    ])

    print(f"Outcomes panel: {len(outcomes)} rows (truncated from {len(scaffold)})")
    print(f"  outcome_death_reintubate value counts:")
    print(outcomes.group_by("outcome_death_reintubate").len().sort("outcome_death_reintubate"))
    print(f"  outcome_death_reintubate_nippv_cpap value counts:")
    print(outcomes.group_by("outcome_death_reintubate_nippv_cpap").len().sort("outcome_death_reintubate_nippv_cpap"))
    return (outcomes,)


@app.cell
def _(cohort, outcomes, pl, resp):
    # Aggregate resp data (HFNC) per window
    hfnc_resp = (
        resp
        .filter(pl.col("device_category") == "high flow nc")
        .join(
            cohort.select(["hospitalization_id", "extubation_time"]),
            on="hospitalization_id",
            how="inner",
        )
        .filter(pl.col("recorded_dttm") > pl.col("extubation_time"))
    )

    # Join to windows
    hfnc_windowed = (
        hfnc_resp
        .join(
            outcomes.select(["hospitalization_id", "window_idx", "window_start", "window_end"]),
            on="hospitalization_id",
            how="inner",
        )
        .filter(
            (pl.col("recorded_dttm") >= pl.col("window_start"))
            & (pl.col("recorded_dttm") < pl.col("window_end"))
        )
    )

    # Aggregate per window
    resp_agg = (
        hfnc_windowed
        .group_by(["hospitalization_id", "window_idx"])
        .agg([
            pl.col("lpm_set").max().alias("hfno_flow_rate_raw"),
            pl.col("fio2_set").max().alias("hfno_fio2"),
        ])
    )

    # Clamp flow rate to [30, 60]
    resp_agg = resp_agg.with_columns(
        pl.col("hfno_flow_rate_raw").clip(30, 60).alias("hfno_flow_rate"),
    )

    # Categorize flow rate: 30-39, 40-49, 50-59, 60
    resp_agg = resp_agg.with_columns(
        pl.when(pl.col("hfno_flow_rate").is_null())
        .then(pl.lit(None, dtype=pl.Utf8))
        .when(pl.col("hfno_flow_rate") >= 60)
        .then(pl.lit("60"))
        .when(pl.col("hfno_flow_rate") >= 50)
        .then(pl.lit("50-59"))
        .when(pl.col("hfno_flow_rate") >= 40)
        .then(pl.lit("40-49"))
        .otherwise(pl.lit("30-39"))
        .alias("hfno_flow_rate_category")
    )

    # Categorize FiO2: 0.21-0.29, 0.3-0.39, ..., 0.9-0.99, 1.0
    resp_agg = resp_agg.with_columns(
        pl.when(pl.col("hfno_fio2").is_null())
        .then(pl.lit(None, dtype=pl.Utf8))
        .when(pl.col("hfno_fio2") >= 1.0)
        .then(pl.lit("1.0"))
        .when(pl.col("hfno_fio2") >= 0.9)
        .then(pl.lit("0.9-0.99"))
        .when(pl.col("hfno_fio2") >= 0.8)
        .then(pl.lit("0.8-0.89"))
        .when(pl.col("hfno_fio2") >= 0.7)
        .then(pl.lit("0.7-0.79"))
        .when(pl.col("hfno_fio2") >= 0.6)
        .then(pl.lit("0.6-0.69"))
        .when(pl.col("hfno_fio2") >= 0.5)
        .then(pl.lit("0.5-0.59"))
        .when(pl.col("hfno_fio2") >= 0.4)
        .then(pl.lit("0.4-0.49"))
        .when(pl.col("hfno_fio2") >= 0.3)
        .then(pl.lit("0.3-0.39"))
        .otherwise(pl.lit("0.21-0.29"))
        .alias("hfno_fio2_category")
    )

    resp_agg = resp_agg.select([
        "hospitalization_id",
        "window_idx",
        "hfno_flow_rate",
        "hfno_flow_rate_category",
        "hfno_fio2",
        "hfno_fio2_category",
    ])

    print(f"Resp aggregates: {len(resp_agg)} window-rows with HFNC data")
    return (resp_agg,)


@app.cell
def _(cohort, outcomes, pl, resp, vitals):
    # Aggregate vitals (SpO2, RR) per window + compute sf_ratio

    # Split vitals by category
    spo2_vitals = vitals.filter(pl.col("vital_category") == "spo2")
    rr_vitals = vitals.filter(pl.col("vital_category") == "respiratory_rate")

    # --- SpO2 per window ---
    spo2_windowed = (
        spo2_vitals
        .join(
            cohort.select(["hospitalization_id", "extubation_time"]),
            on="hospitalization_id",
            how="inner",
        )
        .filter(pl.col("recorded_dttm") > pl.col("extubation_time"))
        .join(
            outcomes.select(["hospitalization_id", "window_idx", "window_start", "window_end"]),
            on="hospitalization_id",
            how="inner",
        )
        .filter(
            (pl.col("recorded_dttm") >= pl.col("window_start"))
            & (pl.col("recorded_dttm") < pl.col("window_end"))
        )
    )

    spo2_agg = (
        spo2_windowed
        .group_by(["hospitalization_id", "window_idx"])
        .agg(pl.col("vital_value").min().alias("rox_lowest_spo2"))
    )

    # --- RR per window ---
    rr_windowed = (
        rr_vitals
        .join(
            cohort.select(["hospitalization_id", "extubation_time"]),
            on="hospitalization_id",
            how="inner",
        )
        .filter(pl.col("recorded_dttm") > pl.col("extubation_time"))
        .join(
            outcomes.select(["hospitalization_id", "window_idx", "window_start", "window_end"]),
            on="hospitalization_id",
            how="inner",
        )
        .filter(
            (pl.col("recorded_dttm") >= pl.col("window_start"))
            & (pl.col("recorded_dttm") < pl.col("window_end"))
        )
    )

    rr_agg = (
        rr_windowed
        .group_by(["hospitalization_id", "window_idx"])
        .agg(pl.col("vital_value").max().alias("rox_highest_rr"))
    )

    # --- sf_ratio: window 0 uses closest SpO2 before extubation / FiO2 at extubation ---
    # --- other windows use lowest SpO2 / highest FiO2 in window ---

    # Window 0: closest SpO2 before extubation
    spo2_pre_extub = (
        spo2_vitals
        .join(
            cohort.select(["hospitalization_id", "extubation_time"]),
            on="hospitalization_id",
            how="inner",
        )
        .filter(pl.col("recorded_dttm") <= pl.col("extubation_time"))
        .sort(["hospitalization_id", "recorded_dttm"])
        .group_by("hospitalization_id")
        .last()
        .select(["hospitalization_id", pl.col("vital_value").alias("spo2_pre_extub")])
    )

    # FiO2 at extubation: closest resp fio2_set before extubation
    fio2_at_extub = (
        resp
        .join(
            cohort.select(["hospitalization_id", "extubation_time"]),
            on="hospitalization_id",
            how="inner",
        )
        .filter(
            (pl.col("recorded_dttm") <= pl.col("extubation_time"))
            & pl.col("fio2_set").is_not_null()
        )
        .sort(["hospitalization_id", "recorded_dttm"])
        .group_by("hospitalization_id")
        .last()
        .select(["hospitalization_id", pl.col("fio2_set").alias("fio2_at_extub")])
    )

    # sf_ratio for window 0
    sf_window0 = (
        spo2_pre_extub
        .join(fio2_at_extub, on="hospitalization_id", how="inner")
        .with_columns([
            (pl.col("spo2_pre_extub") / pl.col("fio2_at_extub")).alias("sf_ratio"),
            pl.col("spo2_pre_extub").alias("sf_spo2"),
            pl.col("fio2_at_extub").alias("sf_fio2"),
        ])
        .with_columns(pl.lit(0).alias("window_idx"))
        .select(["hospitalization_id", "window_idx", "sf_ratio", "sf_spo2", "sf_fio2"])
    )

    # sf_ratio for other windows: lowest SpO2 / highest FiO2 in window
    # Get highest FiO2 from resp (HFNC) per window
    hfnc_fio2_windowed = (
        resp
        .filter(pl.col("device_category") == "high flow nc")
        .join(
            cohort.select(["hospitalization_id", "extubation_time"]),
            on="hospitalization_id",
            how="inner",
        )
        .filter(pl.col("recorded_dttm") > pl.col("extubation_time"))
        .join(
            outcomes.select(["hospitalization_id", "window_idx", "window_start", "window_end"]),
            on="hospitalization_id",
            how="inner",
        )
        .filter(
            (pl.col("recorded_dttm") >= pl.col("window_start"))
            & (pl.col("recorded_dttm") < pl.col("window_end"))
        )
        .group_by(["hospitalization_id", "window_idx"])
        .agg(pl.col("fio2_set").max().alias("window_fio2_max"))
    )

    sf_other = (
        spo2_agg
        .join(hfnc_fio2_windowed, on=["hospitalization_id", "window_idx"], how="inner")
        .filter(pl.col("window_idx") > 0)
        .with_columns([
            (pl.col("rox_lowest_spo2") / pl.col("window_fio2_max")).alias("sf_ratio"),
            pl.col("rox_lowest_spo2").alias("sf_spo2"),
            pl.col("window_fio2_max").alias("sf_fio2"),
        ])
        .select(["hospitalization_id", "window_idx", "sf_ratio", "sf_spo2", "sf_fio2"])
    )

    # Combine sf_ratio
    sf_ratio_all = pl.concat([sf_window0, sf_other])

    # Combine vitals aggregates
    vitals_agg = (
        spo2_agg
        .join(rr_agg, on=["hospitalization_id", "window_idx"], how="outer_coalesce")
        .join(sf_ratio_all, on=["hospitalization_id", "window_idx"], how="outer_coalesce")
    )

    print(f"Vitals aggregates: {len(vitals_agg)} window-rows")
    print(f"  rox_lowest_spo2 non-null: {vitals_agg['rox_lowest_spo2'].is_not_null().sum()}")
    print(f"  rox_highest_rr non-null: {vitals_agg['rox_highest_rr'].is_not_null().sum()}")
    print(f"  sf_ratio non-null: {vitals_agg['sf_ratio'].is_not_null().sum()}")
    return (vitals_agg,)


@app.cell
def _(pl, resp_agg, vitals_agg):
    # Compute ROX index: (lowest_spo2 / fio2) / highest_rr
    rox = (
        vitals_agg.select(["hospitalization_id", "window_idx", "rox_lowest_spo2", "rox_highest_rr"])
        .join(
            resp_agg.select(["hospitalization_id", "window_idx", "hfno_fio2"]),
            on=["hospitalization_id", "window_idx"],
            how="inner",
        )
        .with_columns(
            (
                (pl.col("rox_lowest_spo2") / pl.col("hfno_fio2")) / pl.col("rox_highest_rr")
            ).alias("rox_index")
        )
        .select(["hospitalization_id", "window_idx", "rox_index"])
    )

    print(f"ROX index computed: {len(rox)} window-rows")
    print(f"  ROX non-null: {rox['rox_index'].is_not_null().sum()}")
    non_null_rox = rox.filter(pl.col("rox_index").is_not_null())
    if len(non_null_rox) > 0:
        print(f"  ROX min: {non_null_rox['rox_index'].min():.2f}, max: {non_null_rox['rox_index'].max():.2f}, median: {non_null_rox['rox_index'].median():.2f}")
    return (rox,)


@app.cell
def _(outcomes, pl, vaso_ext):
    # Aggregate vasopressor administrations per window
    vaso_windowed = (
        vaso_ext
        .filter(pl.col("admin_dttm") > pl.col("extubation_time"))
        .join(
            outcomes.select(["hospitalization_id", "window_idx", "window_start", "window_end"]),
            on="hospitalization_id",
            how="inner",
        )
        .filter(
            (pl.col("admin_dttm") >= pl.col("window_start"))
            & (pl.col("admin_dttm") < pl.col("window_end"))
        )
    )

    # Raw per-window flag
    vaso_raw = (
        vaso_windowed
        .select(["hospitalization_id", "window_idx"])
        .unique()
        .with_columns(pl.lit(True).alias("is_vaso"))
    )

    # Build full grid and fill single-window gaps (1,0,1 → 1,1,1)
    vaso_grid = (
        outcomes.select(["hospitalization_id", "window_idx"])
        .unique()
        .join(vaso_raw, on=["hospitalization_id", "window_idx"], how="left")
        .with_columns(pl.col("is_vaso").fill_null(False))
        .sort(["hospitalization_id", "window_idx"])
    )

    vaso_agg = (
        vaso_grid
        .with_columns(
            pl.when(
                ~pl.col("is_vaso")
                & pl.col("is_vaso").shift(1).over("hospitalization_id")
                & pl.col("is_vaso").shift(-1).over("hospitalization_id")
            )
            .then(True)
            .otherwise(pl.col("is_vaso"))
            .alias("is_vaso")
        )
        .filter(pl.col("is_vaso"))
        .select(["hospitalization_id", "window_idx", "is_vaso"])
    )

    _n_filled = len(vaso_agg) - len(vaso_raw)
    print(f"Vaso aggregates: {len(vaso_raw)} raw → {len(vaso_agg)} after gap-fill ({_n_filled} single gaps filled)")
    return (vaso_agg,)


@app.cell
def _(adt_ext, cci_df, cohort, outcomes, pl, resp_agg, rox, vaso_agg, vitals_agg):
    # Assemble final panel

    # Baseline demographics (repeated per window)
    baseline = (
        cohort.select([
            "hospitalization_id",
            "patient_id",
            pl.col("age_at_admission").alias("age"),
            pl.col("sex_category").alias("sex"),
            pl.col("race_category").alias("race"),
            "bmi",
            "sofa_extubation",
        ])
        .join(cci_df, on="hospitalization_id", how="left")
        .rename({"cci_score": "cci"})
        .join(adt_ext, on="hospitalization_id", how="left")
    )

    # Start with outcomes (already truncated)
    panel = outcomes.select([
        "hospitalization_id",
        "window_idx",
        "hour_window",
        "window_start",
        "window_end",
        "outcome_death_reintubate",
        "outcome_death_reintubate_nippv_cpap",
    ])

    # Join baseline
    panel = panel.join(baseline, on="hospitalization_id", how="left")

    # Join resp aggregates
    panel = panel.join(resp_agg, on=["hospitalization_id", "window_idx"], how="left")

    # Join vitals aggregates
    panel = panel.join(vitals_agg, on=["hospitalization_id", "window_idx"], how="left")

    # Join ROX index
    panel = panel.join(rox, on=["hospitalization_id", "window_idx"], how="left")

    # Join vasopressor flag
    panel = panel.join(vaso_agg, on=["hospitalization_id", "window_idx"], how="left")
    panel = panel.with_columns(pl.col("is_vaso").fill_null(False))

    # Select final column order
    panel = panel.select([
        "window_idx",
        "hour_window",
        "window_start",
        "window_end",
        "patient_id",
        "hospitalization_id",
        "age",
        "sex",
        "race",
        "cci",
        "bmi",
        "sofa_extubation",
        "extubation_icu_type",
        "hospital_id",
        "sf_ratio",
        "sf_spo2",
        "sf_fio2",
        "hfno_flow_rate_category",
        "hfno_flow_rate",
        "hfno_fio2_category",
        "hfno_fio2",
        "rox_index",
        "rox_highest_rr",
        "rox_lowest_spo2",
        "is_vaso",
        "outcome_death_reintubate",
        "outcome_death_reintubate_nippv_cpap",
    ])

    print(f"\n=== FINAL PANEL ===")
    print(f"Rows: {len(panel)}")
    print(f"Unique hospitalizations: {panel['hospitalization_id'].n_unique()}")
    print(f"Columns: {panel.columns}")
    print(f"\nNull counts:")
    for c in panel.columns:
        null_ct = panel[c].is_null().sum()
        print(f"  {c}: {null_ct} nulls ({null_ct / len(panel) * 100:.1f}%)")
    return (panel,)


@app.cell
def _(Path, panel):
    # Export to parquet
    output_path = Path(__file__).parent.parent / "output" / "hfno_trajectory.parquet"
    panel.write_parquet(output_path)

    print(f"Panel saved to: {output_path}")
    print(f"Shape: {panel.shape}")
    print(f"\nSample (first 5 rows):")
    print(panel.head(5))

    # Summary stats
    print(f"\n=== Summary Stats ===")
    print(f"Total rows: {len(panel)}")
    print(f"Unique hospitalizations: {panel['hospitalization_id'].n_unique()}")
    print(f"Windows per patient: min={panel.group_by('hospitalization_id').len()['len'].min()}, "
          f"max={panel.group_by('hospitalization_id').len()['len'].max()}, "
          f"median={panel.group_by('hospitalization_id').len()['len'].median()}")

    # Outcome value counts
    print(f"\noutcome_death_reintubate: {dict(zip(panel.group_by('outcome_death_reintubate').len()['outcome_death_reintubate'].to_list(), panel.group_by('outcome_death_reintubate').len()['len'].to_list()))}")
    print(f"outcome_death_reintubate_nippv_cpap: {dict(zip(panel.group_by('outcome_death_reintubate_nippv_cpap').len()['outcome_death_reintubate_nippv_cpap'].to_list(), panel.group_by('outcome_death_reintubate_nippv_cpap').len()['len'].to_list()))}")

    # HFNC ranges
    hfnc_rows = panel.filter(panel["hfno_flow_rate"].is_not_null())
    if len(hfnc_rows) > 0:
        print(f"\nHFNO flow rate range: {hfnc_rows['hfno_flow_rate'].min()} - {hfnc_rows['hfno_flow_rate'].max()}")
    fio2_rows = panel.filter(panel["hfno_fio2"].is_not_null())
    if len(fio2_rows) > 0:
        print(f"HFNO FiO2 range: {fio2_rows['hfno_fio2'].min()} - {fio2_rows['hfno_fio2'].max()}")
    rox_rows = panel.filter(panel["rox_index"].is_not_null())
    if len(rox_rows) > 0:
        print(f"ROX index range: {rox_rows['rox_index'].min():.2f} - {rox_rows['rox_index'].max():.2f}")
    return


@app.cell
def _(panel, pl, plt):
    _raw = panel.select(["window_idx", "hfno_flow_rate"]).drop_nulls()
    _agg = (
        _raw
        .group_by("window_idx")
        .agg(pl.col("hfno_flow_rate").mean())
        .sort("window_idx")
    )
    plt.figure(figsize=(12, 6))
    plt.scatter(
        (_raw["window_idx"] * 4 / 24).to_list(),
        _raw["hfno_flow_rate"].to_list(),
        alpha=0.15,
        s=8,
        label="Individual",
    )
    plt.plot(
        (_agg["window_idx"] * 4 / 24).to_list(),
        _agg["hfno_flow_rate"].to_list(),
        lw=2.5,
        label="Mean",
    )
    plt.title("HFNO Flow Over Time Since Extubation")
    plt.xlabel("Days Since Extubation")
    plt.ylabel("HFNO Flow (L/min)")
    plt.legend()
    return


@app.cell
def _(panel, pl, plt):
    _raw = panel.select(["window_idx", "hfno_fio2"]).drop_nulls()
    _agg = (
        _raw
        .group_by("window_idx")
        .agg(pl.col("hfno_fio2").mean())
        .sort("window_idx")
    )
    plt.figure(figsize=(12, 6))
    plt.scatter(
        (_raw["window_idx"] * 4).to_list(),
        (_raw["hfno_fio2"] * 100).to_list(),
        alpha=0.15,
        s=8,
        label="Individual",
    )
    plt.plot(
        (_agg["window_idx"] * 4).to_list(),
        (_agg["hfno_fio2"] * 100).to_list(),
        lw=2.5,
        label="Mean",
    )
    plt.title("FiO2 Levels Over Time (0 to 168 Hours)")
    plt.xlabel("Time (hours)")
    plt.ylabel("FiO2 (%)")
    plt.legend()
    return


@app.cell
def _(mo):
    mo.md("""
    # ROX Prediction Flat File

    A one-row-per-hospitalization dataset for ROX-based extubation outcome prediction.
    Combines demographics, comorbidities, severity, vasopressor flags, ROX index at
    3 early time points, and 4 binary outcomes.
    """)
    return


@app.cell
def _(
    DATA_DIR,
    FILETYPE,
    MedicationAdminContinuous,
    TIMEZONE,
    cohort,
    cohort_ids,
    pd,
    pl,
):
    # Load vasopressor (vasoactive) medication data
    vaso_table = MedicationAdminContinuous.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={"hospitalization_id": cohort_ids, "med_group": "vasoactives"},
    )
    vaso_pd = vaso_table.df[["hospitalization_id", "admin_dttm"]].copy()
    vaso_pd["admin_dttm"] = pd.to_datetime(vaso_pd["admin_dttm"], errors="coerce")
    if vaso_pd["admin_dttm"].dt.tz is not None:
        vaso_pd["admin_dttm"] = vaso_pd["admin_dttm"].dt.tz_localize(None)

    vaso = pl.from_pandas(vaso_pd)
    del vaso_pd

    # Join with extubation time
    vaso_ext = vaso.join(
        cohort.select(["hospitalization_id", "extubation_time"]),
        on="hospitalization_id",
        how="inner",
    )

    # Compute per-window vasopressor flags (0-4h, 4-8h, 8-12h post-extubation)
    _window_defs = [
        ("vasopressor_0_4h", 0, 4),
        ("vasopressor_4_8h", 4, 8),
        ("vasopressor_8_12h", 8, 12),
    ]

    _vaso_flags = cohort.select("hospitalization_id")
    for _col_name, _start_h, _end_h in _window_defs:
        _flag = (
            vaso_ext
            .filter(
                (pl.col("admin_dttm") >= (pl.col("extubation_time") + pl.duration(hours=_start_h)))
                & (pl.col("admin_dttm") < (pl.col("extubation_time") + pl.duration(hours=_end_h)))
            )
            .select("hospitalization_id")
            .unique()
            .with_columns(pl.lit(True).alias(_col_name))
        )
        _vaso_flags = _vaso_flags.join(_flag, on="hospitalization_id", how="left")

    vaso_windows = _vaso_flags.with_columns([
        pl.col("vasopressor_0_4h").fill_null(False),
        pl.col("vasopressor_4_8h").fill_null(False),
        pl.col("vasopressor_8_12h").fill_null(False),
    ])

    print(f"Vasopressor flags computed:")
    for _col_name, _, _ in _window_defs:
        _n_true = vaso_windows[_col_name].sum()
        print(f"  {_col_name}: {_n_true} ({_n_true / len(vaso_windows) * 100:.1f}%)")
    return vaso_ext, vaso_windows


@app.cell
def _(
    DATA_DIR,
    FILETYPE,
    HospitalDiagnosis,
    Hospitalization,
    TIMEZONE,
    cohort,
    cohort_patient_ids,
    pd,
    pl,
):
    # Chronic respiratory failure with hypoxia (J96.11) from PRIOR hospitalizations

    # Load ALL hospitalizations for cohort patients (not just current)
    all_hosp_table = Hospitalization.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={"patient_id": cohort_patient_ids},
    )
    all_hosp_pd = all_hosp_table.df[["hospitalization_id", "patient_id", "admission_dttm"]].copy()
    all_hosp_pd["admission_dttm"] = pd.to_datetime(all_hosp_pd["admission_dttm"], errors="coerce")
    if all_hosp_pd["admission_dttm"].dt.tz is not None:
        all_hosp_pd["admission_dttm"] = all_hosp_pd["admission_dttm"].dt.tz_localize(None)
    all_hosp = pl.from_pandas(all_hosp_pd)
    del all_hosp_pd

    # Load diagnoses for ALL hospitalizations of these patients
    all_hosp_ids = all_hosp["hospitalization_id"].to_list()
    all_diag_table = HospitalDiagnosis.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={"hospitalization_id": all_hosp_ids},
    )
    all_diag_pd = all_diag_table.df[["hospitalization_id", "diagnosis_code"]].copy()
    all_diag = pl.from_pandas(all_diag_pd)
    del all_diag_pd

    # Clean diagnosis codes: lowercase, remove dots
    all_diag = all_diag.with_columns(
        pl.col("diagnosis_code").str.to_lowercase().str.replace_all(r"\.", "").alias("dx_clean")
    )

    # Filter to J96.11 (chronic respiratory failure with hypoxia)
    j9611_diag = all_diag.filter(pl.col("dx_clean").str.starts_with("j9611"))

    # Join with hospitalization to get admission_dttm and patient_id
    j9611_hosp = j9611_diag.join(
        all_hosp.select(["hospitalization_id", "patient_id", "admission_dttm"]),
        on="hospitalization_id",
        how="inner",
    ).select(["patient_id", "admission_dttm"]).unique()

    # For each cohort hospitalization, check if patient has ANY prior hospitalization with J96.11
    cohort_admit = cohort.select(["hospitalization_id", "patient_id"]).join(
        all_hosp.select(["hospitalization_id", "admission_dttm"]),
        on="hospitalization_id",
        how="left",
    ).rename({"admission_dttm": "cohort_admit_dttm"})

    j9611_prior = (
        cohort_admit
        .join(j9611_hosp, on="patient_id", how="left")
        .filter(
            pl.col("admission_dttm").is_not_null()
            & (pl.col("admission_dttm") < pl.col("cohort_admit_dttm"))
        )
        .select("hospitalization_id")
        .unique()
        .with_columns(pl.lit(True).alias("chronic_resp_failure_hypoxia"))
    )

    # Join back to full cohort
    j9611_flag = (
        cohort.select("hospitalization_id")
        .join(j9611_prior, on="hospitalization_id", how="left")
        .with_columns(pl.col("chronic_resp_failure_hypoxia").fill_null(False))
    )

    print(f"J96.11 prior hospitalization flag:")
    _n_true = j9611_flag["chronic_resp_failure_hypoxia"].sum()
    print(f"  True: {_n_true} ({_n_true / len(j9611_flag) * 100:.1f}%)")
    return (j9611_flag,)


@app.cell
def _(pl, rox):
    # Pivot ROX index for the first 3 windows (0-4h, 4-8h, 8-12h)
    rox_early = rox.filter(pl.col("window_idx").is_in([0, 1, 2]))

    rox_wide = (
        rox_early
        .with_columns(
            pl.when(pl.col("window_idx") == 0).then(pl.lit("rox_0_4h"))
            .when(pl.col("window_idx") == 1).then(pl.lit("rox_4_8h"))
            .when(pl.col("window_idx") == 2).then(pl.lit("rox_8_12h"))
            .alias("rox_col")
        )
        .pivot(on="rox_col", index="hospitalization_id", values="rox_index")
    )

    # Ensure all 3 columns exist (some may be missing if no data)
    for _col_name in ["rox_0_4h", "rox_4_8h", "rox_8_12h"]:
        if _col_name not in rox_wide.columns:
            rox_wide = rox_wide.with_columns(pl.lit(None, dtype=pl.Float64).alias(_col_name))

    rox_wide = rox_wide.select(["hospitalization_id", "rox_0_4h", "rox_4_8h", "rox_8_12h"])

    print(f"ROX wide: {len(rox_wide)} hospitalizations with at least one early ROX value")
    for _c in ["rox_0_4h", "rox_4_8h", "rox_8_12h"]:
        _non_null = rox_wide[_c].is_not_null().sum()
        print(f"  {_c}: {_non_null} non-null")
    return (rox_wide,)


@app.cell
def _(cohort, events, pl):
    # Compute 4 binary extubation success outcomes

    outcome_base = (
        cohort.select(["hospitalization_id", "extubation_time"])
        .join(events.select(["hospitalization_id", "reintubation_time", "nippv_cpap_time", "death_time"]),
              on="hospitalization_id", how="left")
    )

    # Define time windows
    outcome_base = outcome_base.with_columns([
        (pl.col("extubation_time") + pl.duration(hours=72)).alias("cutoff_3d"),
        (pl.col("extubation_time") + pl.duration(hours=168)).alias("cutoff_7d"),
    ])

    # Event flags within each window
    outcome_base = outcome_base.with_columns([
        # 3-day flags
        (
            pl.col("reintubation_time").is_not_null()
            & (pl.col("reintubation_time") <= pl.col("cutoff_3d"))
        ).alias("reintubation_in_3d"),
        (
            pl.col("death_time").is_not_null()
            & (pl.col("death_time") > pl.col("extubation_time"))
            & (pl.col("death_time") <= pl.col("cutoff_3d"))
        ).alias("death_in_3d"),
        (
            pl.col("nippv_cpap_time").is_not_null()
            & (pl.col("nippv_cpap_time") <= pl.col("cutoff_3d"))
        ).alias("nippv_cpap_in_3d"),
        # 7-day flags
        (
            pl.col("reintubation_time").is_not_null()
            & (pl.col("reintubation_time") <= pl.col("cutoff_7d"))
        ).alias("reintubation_in_7d"),
        (
            pl.col("death_time").is_not_null()
            & (pl.col("death_time") > pl.col("extubation_time"))
            & (pl.col("death_time") <= pl.col("cutoff_7d"))
        ).alias("death_in_7d"),
        (
            pl.col("nippv_cpap_time").is_not_null()
            & (pl.col("nippv_cpap_time") <= pl.col("cutoff_7d"))
        ).alias("nippv_cpap_in_7d"),
    ])

    # Compute success outcomes
    flat_outcomes = outcome_base.select([
        "hospitalization_id",
        (~pl.col("death_in_7d") & ~pl.col("reintubation_in_7d")).alias("extubation_success_7d"),
        (~pl.col("death_in_7d") & ~pl.col("reintubation_in_7d") & ~pl.col("nippv_cpap_in_7d")).alias("extubation_success_strict_7d"),
        (~pl.col("death_in_3d") & ~pl.col("reintubation_in_3d")).alias("extubation_success_3d"),
        (~pl.col("death_in_3d") & ~pl.col("reintubation_in_3d") & ~pl.col("nippv_cpap_in_3d")).alias("extubation_success_strict_3d"),
    ])

    print(f"Extubation outcomes computed for {len(flat_outcomes)} hospitalizations:")
    for _c in ["extubation_success_7d", "extubation_success_strict_7d", "extubation_success_3d", "extubation_success_strict_3d"]:
        _n_success = flat_outcomes[_c].sum()
        print(f"  {_c}: {_n_success} ({_n_success / len(flat_outcomes) * 100:.1f}%) success")
    return (flat_outcomes,)


@app.cell
def _(
    Path,
    adt_ext,
    cci_df,
    cohort,
    flat_outcomes,
    j9611_flag,
    pl,
    rox_wide,
    vaso_windows,
):
    # Assemble the ROX prediction flat file

    _rox_flat = (
        cohort.select([
            "hospitalization_id",
            pl.col("age_at_admission").alias("age"),
            pl.col("sex_category").alias("sex"),
            "sofa_extubation",
        ])
        .join(cci_df.rename({"cci_score": "cci"}), on="hospitalization_id", how="left")
        .join(adt_ext, on="hospitalization_id", how="left")
        .join(j9611_flag, on="hospitalization_id", how="left")
        .join(vaso_windows, on="hospitalization_id", how="left")
        .join(rox_wide, on="hospitalization_id", how="left")
        .join(flat_outcomes, on="hospitalization_id", how="left")
    )

    # Fill null booleans with False
    _bool_cols = [
        "chronic_resp_failure_hypoxia",
        "vasopressor_0_4h", "vasopressor_4_8h", "vasopressor_8_12h",
    ]
    _rox_flat = _rox_flat.with_columns([
        pl.col(_c).fill_null(False) for _c in _bool_cols
    ])

    # Select final column order
    _rox_flat = _rox_flat.select([
        "hospitalization_id",
        "age",
        "sex",
        "cci",
        "extubation_icu_type",
        "hospital_id",
        "chronic_resp_failure_hypoxia",
        "sofa_extubation",
        "vasopressor_0_4h",
        "vasopressor_4_8h",
        "vasopressor_8_12h",
        "rox_0_4h",
        "rox_4_8h",
        "rox_8_12h",
        "extubation_success_7d",
        "extubation_success_strict_7d",
        "extubation_success_3d",
        "extubation_success_strict_3d",
    ])

    # Save
    _rox_output_path = Path(__file__).parent.parent / "output" / "rox_prediction.parquet"
    _rox_flat.write_parquet(_rox_output_path)

    print(f"\n=== ROX Prediction Flat File ===")
    print(f"Saved to: {_rox_output_path}")
    print(f"Shape: {_rox_flat.shape}")
    print(f"\nColumn summary:")
    for _c in _rox_flat.columns:
        _null_ct = _rox_flat[_c].is_null().sum()
        _dtype = _rox_flat[_c].dtype
        if _dtype == pl.Boolean:
            _n_true = _rox_flat[_c].sum()
            print(f"  {_c} ({_dtype}): {_n_true} True ({_n_true / len(_rox_flat) * 100:.1f}%), {_null_ct} nulls")
        elif _dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
            _non_null = _rox_flat.filter(pl.col(_c).is_not_null())
            if len(_non_null) > 0:
                print(f"  {_c} ({_dtype}): min={_non_null[_c].min()}, max={_non_null[_c].max()}, median={_non_null[_c].median()}, {_null_ct} nulls")
            else:
                print(f"  {_c} ({_dtype}): all null")
        else:
            print(f"  {_c} ({_dtype}): {_null_ct} nulls")
    return


if __name__ == "__main__":
    app.run()
