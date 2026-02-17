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
    from clifpy.tables import Patient, CrrtTherapy, MedicationAdminContinuous, Vitals
    return (
        CrrtTherapy,
        MedicationAdminContinuous,
        Path,
        Patient,
        Vitals,
        json,
        mo,
        pd,
        pl,
    )


@app.cell
def _(mo):
    mo.md("""
    # 03 Extubation Success: 7-Day Outcome Flags

    Adds binary extubation success flags to the cohort.

    - **`extubation_success_7d`** = alive + no reintubation (no IMV) within 7 days of extubation.
      All non-IMV respiratory devices (HFNC, NIPPV, CPAP, nasal cannula, etc.) are allowed.
    - **`extubation_success_strict_7d`** = alive + no reintubation + **no NIPPV/CPAP** within 7 days.
      Only HFNC, nasal cannula, and room air are allowed.
    - **`definitive_hfno_weaning`** = patient transitioned from HFNO to low-support device
      (room air, nasal cannula, face mask), sustained >48 hours with no relapse, no escalation
      to NIPPV/CPAP/IMV, and alive at Day 7.
    - **`time_to_hfno_weaning_hours`** = hours from first HFNO after extubation to first sustained
      transition to low-support device. Null if not definitively weaned.

    **Life support prior to extubation** (ICU admission → extubation):
    - **`crrt_prior`** = any CRRT during ICU stay before extubation
    - **`vasopressor_prior`** = any vasopressor during ICU stay before extubation
    - **`nippv_cpap_prior`** = any NIPPV/CPAP during ICU stay before extubation
    - **`hfno_prior`** = any HFNO during ICU stay before extubation
    - **`any_life_support_prior`** = OR of the four above

    **Life support at time of extubation** (24h before extubation):
    - **`crrt_at_extubation`** = CRRT within 24h before extubation
    - **`vasopressor_at_extubation`** = vasopressor within 24h before extubation
    - **`life_support_at_extubation`** = OR of the two above
    - **`worst_sf_ratio_at_extubation`** = worst (lowest) SpO2/FiO2 ratio in 24h before extubation
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
    # Load HFNO cohort
    cohort_hfno_pd = pd.read_parquet(
        Path(__file__).parent.parent / "output" / "cohort_inclusion.parquet"
    )
    for col in cohort_hfno_pd.select_dtypes(include=["datetimetz"]).columns:
        cohort_hfno_pd[col] = cohort_hfno_pd[col].dt.tz_localize(None)
    cohort_hfno = pl.from_pandas(cohort_hfno_pd).with_columns(pl.lit(True).alias("is_hfno_cohort"))
    del cohort_hfno_pd

    # Load low-flow cohort
    cohort_low_flow_pd = pd.read_parquet(
        Path(__file__).parent.parent / "output" / "cohort_low_flow.parquet"
    )
    for col in cohort_low_flow_pd.select_dtypes(include=["datetimetz"]).columns:
        cohort_low_flow_pd[col] = cohort_low_flow_pd[col].dt.tz_localize(None)
    cohort_low_flow = pl.from_pandas(cohort_low_flow_pd).with_columns(pl.lit(False).alias("is_hfno_cohort"))
    del cohort_low_flow_pd

    # Combine both cohorts
    cohort = pl.concat([cohort_hfno, cohort_low_flow])

    # Add 7-day window end
    cohort = cohort.with_columns(
        (pl.col("extubation_time") + pl.duration(hours=168)).alias("window_end_7d")
    )

    cohort_ids = cohort["hospitalization_id"].to_list()
    cohort_patient_ids = cohort["patient_id"].unique().to_list()

    print(f"HFNO cohort: {len(cohort_hfno)} hospitalizations")
    print(f"Low-flow cohort: {len(cohort_low_flow)} hospitalizations")
    print(f"Combined cohort: {len(cohort)} hospitalizations, {len(cohort_patient_ids)} patients")
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
def _(DATA_DIR, FILETYPE, TIMEZONE, Vitals, cohort_ids, pd, pl):
    # Load SpO2 vitals from CLIF Vitals table
    vitals_table = Vitals.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={
            "hospitalization_id": cohort_ids,
            "vital_category": ["spo2"],
        },
    )
    vitals_pd = vitals_table.df.copy()

    # Strip tz
    if vitals_pd["recorded_dttm"].dt.tz is not None:
        vitals_pd["recorded_dttm"] = vitals_pd["recorded_dttm"].dt.tz_localize(None)

    spo2_vitals = pl.from_pandas(vitals_pd)
    del vitals_pd

    print(f"SpO2 vitals loaded: {len(spo2_vitals)} rows for {spo2_vitals['hospitalization_id'].n_unique()} hospitalizations")
    return (spo2_vitals,)


@app.cell
def _(cohort, pl, resp, spo2_vitals):
    # Compute worst (lowest) S/F ratio in the 24h before extubation
    # S/F ratio = min(SpO2) / max(FiO2) in the window

    extub_window = cohort.select(["hospitalization_id", "extubation_time"])

    # Min SpO2 in [extubation_time - 24h, extubation_time]
    min_spo2 = (
        spo2_vitals
        .join(extub_window, on="hospitalization_id", how="inner")
        .filter(
            (pl.col("recorded_dttm") >= (pl.col("extubation_time") - pl.duration(hours=24)))
            & (pl.col("recorded_dttm") <= pl.col("extubation_time"))
        )
        .group_by("hospitalization_id")
        .agg(pl.col("vital_value").min().alias("min_spo2"))
    )

    # Max FiO2 in [extubation_time - 24h, extubation_time]
    max_fio2 = (
        resp
        .filter(pl.col("fio2_set").is_not_null())
        .join(extub_window, on="hospitalization_id", how="inner")
        .filter(
            (pl.col("recorded_dttm") >= (pl.col("extubation_time") - pl.duration(hours=24)))
            & (pl.col("recorded_dttm") <= pl.col("extubation_time"))
        )
        .group_by("hospitalization_id")
        .agg(pl.col("fio2_set").max().alias("max_fio2"))
    )

    # Join and compute ratio
    sf_at_extubation = (
        min_spo2
        .join(max_fio2, on="hospitalization_id", how="inner")
        .with_columns(
            (pl.col("min_spo2") / pl.col("max_fio2")).alias("worst_sf_ratio_at_extubation")
        )
        .select(["hospitalization_id", "worst_sf_ratio_at_extubation"])
    )

    n_sf = len(sf_at_extubation)
    median_sf = sf_at_extubation["worst_sf_ratio_at_extubation"].median()
    print(f"Worst S/F ratio at extubation: {n_sf} hospitalizations with values")
    print(f"  Median: {median_sf:.1f}")
    return (sf_at_extubation,)


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
def _(CrrtTherapy, DATA_DIR, FILETYPE, TIMEZONE, cohort, cohort_ids, pd, pl):
    # Load CRRT table
    crrt_table = CrrtTherapy.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={"hospitalization_id": cohort_ids},
    )
    crrt_pd = crrt_table.df[["hospitalization_id", "recorded_dttm"]].copy()
    crrt_pd["recorded_dttm"] = pd.to_datetime(crrt_pd["recorded_dttm"], errors="coerce")
    if crrt_pd["recorded_dttm"].dt.tz is not None:
        crrt_pd["recorded_dttm"] = crrt_pd["recorded_dttm"].dt.tz_localize(None)

    crrt = pl.from_pandas(crrt_pd)
    del crrt_pd

    crrt_with_cohort = crrt.join(
        cohort.select(["hospitalization_id", "icu_start", "extubation_time"]),
        on="hospitalization_id",
        how="inner",
    )

    # crrt_prior: any CRRT record in [icu_start, extubation_time)
    crrt_prior = (
        crrt_with_cohort
        .filter(
            (pl.col("recorded_dttm") >= pl.col("icu_start"))
            & (pl.col("recorded_dttm") < pl.col("extubation_time"))
        )
        .select("hospitalization_id")
        .unique()
        .with_columns(pl.lit(True).alias("crrt_prior"))
    )

    # crrt_at_extubation: any CRRT record in [extubation_time - 24h, extubation_time]
    crrt_at_extubation = (
        crrt_with_cohort
        .filter(
            (pl.col("recorded_dttm") >= (pl.col("extubation_time") - pl.duration(hours=24)))
            & (pl.col("recorded_dttm") <= pl.col("extubation_time"))
        )
        .select("hospitalization_id")
        .unique()
        .with_columns(pl.lit(True).alias("crrt_at_extubation"))
    )

    print(f"CRRT prior to extubation: {len(crrt_prior)} hospitalizations")
    print(f"CRRT at extubation (24h window): {len(crrt_at_extubation)} hospitalizations")
    return crrt_at_extubation, crrt_prior


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
    # Load MedicationAdminContinuous filtered to vasoactives
    med_table = MedicationAdminContinuous.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={"hospitalization_id": cohort_ids, "med_group": "vasoactives"},
    )
    med_pd = med_table.df[["hospitalization_id", "admin_dttm"]].copy()
    med_pd["admin_dttm"] = pd.to_datetime(med_pd["admin_dttm"], errors="coerce")
    if med_pd["admin_dttm"].dt.tz is not None:
        med_pd["admin_dttm"] = med_pd["admin_dttm"].dt.tz_localize(None)

    vaso = pl.from_pandas(med_pd)
    del med_pd

    vaso_with_cohort = vaso.join(
        cohort.select(["hospitalization_id", "icu_start", "extubation_time"]),
        on="hospitalization_id",
        how="inner",
    )

    # vasopressor_prior: any vasoactive record in [icu_start, extubation_time)
    vasopressor_prior = (
        vaso_with_cohort
        .filter(
            (pl.col("admin_dttm") >= pl.col("icu_start"))
            & (pl.col("admin_dttm") < pl.col("extubation_time"))
        )
        .select("hospitalization_id")
        .unique()
        .with_columns(pl.lit(True).alias("vasopressor_prior"))
    )

    # vasopressor_at_extubation: any vasoactive record in [extubation_time - 24h, extubation_time]
    vasopressor_at_extubation = (
        vaso_with_cohort
        .filter(
            (pl.col("admin_dttm") >= (pl.col("extubation_time") - pl.duration(hours=24)))
            & (pl.col("admin_dttm") <= pl.col("extubation_time"))
        )
        .select("hospitalization_id")
        .unique()
        .with_columns(pl.lit(True).alias("vasopressor_at_extubation"))
    )

    print(f"Vasopressor prior to extubation: {len(vasopressor_prior)} hospitalizations")
    print(f"Vasopressor at extubation (24h window): {len(vasopressor_at_extubation)} hospitalizations")
    return vasopressor_at_extubation, vasopressor_prior


@app.cell
def _(cohort, pl, resp):
    # NIPPV/CPAP and HFNO prior flags from resp waterfall
    # Window: [icu_start, extubation_time)
    resp_prior = resp.join(
        cohort.select(["hospitalization_id", "icu_start", "extubation_time"]),
        on="hospitalization_id",
        how="inner",
    ).filter(
        (pl.col("recorded_dttm") >= pl.col("icu_start"))
        & (pl.col("recorded_dttm") < pl.col("extubation_time"))
    )

    # nippv_cpap_prior: any NIPPV/CPAP in [icu_start, extubation_time)
    nippv_cpap_prior = (
        resp_prior
        .filter(pl.col("device_category").is_in(["nippv", "cpap"]))
        .select("hospitalization_id")
        .unique()
        .with_columns(pl.lit(True).alias("nippv_cpap_prior"))
    )

    # hfno_prior: any HFNO in [icu_start, extubation_time)
    hfno_prior = (
        resp_prior
        .filter(pl.col("device_category") == "high flow nc")
        .select("hospitalization_id")
        .unique()
        .with_columns(pl.lit(True).alias("hfno_prior"))
    )

    print(f"NIPPV/CPAP prior to extubation: {len(nippv_cpap_prior)} hospitalizations")
    print(f"HFNO prior to extubation: {len(hfno_prior)} hospitalizations")
    return hfno_prior, nippv_cpap_prior


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
def _(cohort, death_by_hosp, pl, resp):
    WEANED_DEVICES = ["room air", "nasal cannula", "face mask"]
    ESCALATION_DEVICES = ["nippv", "cpap", "imv"]

    # All post-extubation records within 7-day window
    post_ext = resp.join(
        cohort.select(["hospitalization_id", "extubation_time", "window_end_7d"]),
        on="hospitalization_id",
        how="inner",
    ).filter(
        (pl.col("recorded_dttm") > pl.col("extubation_time"))
        & (pl.col("recorded_dttm") <= pl.col("window_end_7d"))
    )

    # First and last HFNO times per patient
    hfno_times = (
        post_ext.filter(pl.col("device_category") == "high flow nc")
        .group_by("hospitalization_id")
        .agg([
            pl.col("recorded_dttm").min().alias("first_hfno_time"),
            pl.col("recorded_dttm").max().alias("last_hfno_time"),
        ])
    )

    # Any escalation (NIPPV/CPAP/IMV) post-extubation
    has_escalation = (
        post_ext.filter(pl.col("device_category").is_in(ESCALATION_DEVICES))
        .select("hospitalization_id").unique()
        .with_columns(pl.lit(True).alias("has_escalation"))
    )

    # First weaned-device record AFTER last HFNO
    weaning_start = (
        post_ext.filter(pl.col("device_category").is_in(WEANED_DEVICES))
        .join(hfno_times.select(["hospitalization_id", "last_hfno_time"]),
              on="hospitalization_id", how="inner")
        .filter(pl.col("recorded_dttm") > pl.col("last_hfno_time"))
        .group_by("hospitalization_id")
        .agg(pl.col("recorded_dttm").min().alias("weaning_start_time"))
    )

    # Check no non-weaned records in (weaning_start_time, weaning_start_time + 48h]
    non_weaned_after_wean = (
        post_ext.filter(~pl.col("device_category").is_in(WEANED_DEVICES))
        .join(weaning_start, on="hospitalization_id", how="inner")
        .filter(
            (pl.col("recorded_dttm") > pl.col("weaning_start_time"))
            & (pl.col("recorded_dttm") <= (pl.col("weaning_start_time") + pl.duration(hours=48)))
        )
        .select("hospitalization_id").unique()
        .with_columns(pl.lit(True).alias("relapse_in_48h"))
    )

    # Alive at Day 7
    alive_7d = (
        cohort.select(["hospitalization_id", "patient_id", "extubation_time", "window_end_7d"])
        .join(death_by_hosp, on="hospitalization_id", how="left")
        .with_columns(
            (pl.col("death_dttm").is_null()
             | (pl.col("death_dttm") > pl.col("window_end_7d"))
            ).alias("alive_at_7d")
        )
        .select(["hospitalization_id", "alive_at_7d"])
    )

    # Assemble per HFNO patient
    hfno_weaning = (
        hfno_times
        .join(weaning_start, on="hospitalization_id", how="left")
        .join(has_escalation, on="hospitalization_id", how="left")
        .join(non_weaned_after_wean, on="hospitalization_id", how="left")
        .join(alive_7d, on="hospitalization_id", how="left")
        .join(
            cohort.select(["hospitalization_id", "window_end_7d"]),
            on="hospitalization_id", how="left",
        )
        .with_columns([
            pl.col("has_escalation").fill_null(False),
            pl.col("relapse_in_48h").fill_null(False),
        ])
        .with_columns(
            (
                pl.col("alive_at_7d")
                & ~pl.col("has_escalation")
                & ~pl.col("relapse_in_48h")
                & pl.col("weaning_start_time").is_not_null()
                & ((pl.col("window_end_7d") - pl.col("weaning_start_time")).dt.total_hours() > 48)
            ).alias("definitive_hfno_weaning")
        )
        .with_columns(
            pl.when(pl.col("definitive_hfno_weaning"))
            .then((pl.col("weaning_start_time") - pl.col("first_hfno_time")).dt.total_hours())
            .otherwise(None)
            .alias("time_to_hfno_weaning_hours")
        )
        .select(["hospitalization_id", "definitive_hfno_weaning", "time_to_hfno_weaning_hours"])
    )

    n_hfno = len(hfno_weaning)
    n_weaned = hfno_weaning["definitive_hfno_weaning"].sum()
    print(f"HFNO patients: {n_hfno}")
    print(f"Definitive HFNO weaning: {n_weaned}/{n_hfno} ({n_weaned / n_hfno * 100:.1f}%)")
    return (hfno_weaning,)


@app.cell
def _(
    cohort,
    crrt_at_extubation,
    crrt_prior,
    death_flag,
    hfno_prior,
    hfno_weaning,
    nippv_cpap_flag,
    nippv_cpap_icu_flag,
    nippv_cpap_prior,
    pl,
    reintubation_flag,
    sf_at_extubation,
    vasopressor_at_extubation,
    vasopressor_prior,
):
    # Build extubation success flags and merge to cohort
    result = (
        cohort
        .join(reintubation_flag, on="hospitalization_id", how="left")
        .join(death_flag, on="hospitalization_id", how="left")
        .join(nippv_cpap_flag, on="hospitalization_id", how="left")
        .join(nippv_cpap_icu_flag, on="hospitalization_id", how="left")
        .join(hfno_weaning, on="hospitalization_id", how="left")
        .join(crrt_prior, on="hospitalization_id", how="left")
        .join(crrt_at_extubation, on="hospitalization_id", how="left")
        .join(vasopressor_prior, on="hospitalization_id", how="left")
        .join(vasopressor_at_extubation, on="hospitalization_id", how="left")
        .join(nippv_cpap_prior, on="hospitalization_id", how="left")
        .join(hfno_prior, on="hospitalization_id", how="left")
        .join(sf_at_extubation, on="hospitalization_id", how="left")
        .with_columns([
            pl.col("reintubation_in_7d").fill_null(False),
            pl.col("death_in_7d").fill_null(False),
            pl.col("nippv_cpap_in_7d").fill_null(False),
            pl.col("nippv_cpap_in_7d_plus_in_ICU").fill_null(False),
            pl.col("definitive_hfno_weaning").fill_null(False),
            pl.col("crrt_prior").fill_null(False),
            pl.col("crrt_at_extubation").fill_null(False),
            pl.col("vasopressor_prior").fill_null(False),
            pl.col("vasopressor_at_extubation").fill_null(False),
            pl.col("nippv_cpap_prior").fill_null(False),
            pl.col("hfno_prior").fill_null(False),
        ])
        .with_columns([
            (~pl.col("death_in_7d") & ~pl.col("reintubation_in_7d")).alias("extubation_success_7d"),
            (~pl.col("death_in_7d") & ~pl.col("reintubation_in_7d") & ~pl.col("nippv_cpap_in_7d")).alias("extubation_success_strict_7d"),
            (pl.col("crrt_prior") | pl.col("vasopressor_prior") | pl.col("nippv_cpap_prior") | pl.col("hfno_prior")).alias("any_life_support_prior"),
            (pl.col("crrt_at_extubation") | pl.col("vasopressor_at_extubation")).alias("life_support_at_extubation"),
        ])
    )

    N = len(result)
    n_reintub = result["reintubation_in_7d"].sum()
    n_death = result["death_in_7d"].sum()
    n_nippv_cpap = result["nippv_cpap_in_7d"].sum()
    n_nippv_cpap_icu = result["nippv_cpap_in_7d_plus_in_ICU"].sum()
    n_success = result["extubation_success_7d"].sum()
    n_success_strict = result["extubation_success_strict_7d"].sum()
    n_hfno_weaned = result["definitive_hfno_weaning"].sum()
    n_hfno_weaned_with_time = result["time_to_hfno_weaning_hours"].is_not_null().sum()
    n_crrt_prior = result["crrt_prior"].sum()
    n_crrt_at_ext = result["crrt_at_extubation"].sum()
    n_vaso_prior = result["vasopressor_prior"].sum()
    n_vaso_at_ext = result["vasopressor_at_extubation"].sum()
    n_nippv_prior = result["nippv_cpap_prior"].sum()
    n_hfno_prior = result["hfno_prior"].sum()
    n_any_ls_prior = result["any_life_support_prior"].sum()
    n_ls_at_ext = result["life_support_at_extubation"].sum()
    sf_col = result["worst_sf_ratio_at_extubation"].drop_nulls()
    n_sf_nonnull = len(sf_col)
    sf_median = sf_col.median() if n_sf_nonnull > 0 else None
    sf_q25 = sf_col.quantile(0.25) if n_sf_nonnull > 0 else None
    sf_q75 = sf_col.quantile(0.75) if n_sf_nonnull > 0 else None

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
    print(f"  definitive_hfno_weaning: Sustained weaning from HFNO >48h, no escalation, alive at 7d")
    print(f"    -> {n_hfno_weaned}/{N} ({n_hfno_weaned / N * 100:.1f}%)")
    print(f"  time_to_hfno_weaning_hours: Hours from first HFNO to sustained low-support transition")
    print(f"    -> {n_hfno_weaned_with_time} patients with non-null values")
    print()
    print("=== Life Support Flags ===")
    print(f"  crrt_prior: CRRT during ICU stay before extubation")
    print(f"    -> {n_crrt_prior}/{N} ({n_crrt_prior / N * 100:.1f}%)")
    print(f"  vasopressor_prior: Vasopressor during ICU stay before extubation")
    print(f"    -> {n_vaso_prior}/{N} ({n_vaso_prior / N * 100:.1f}%)")
    print(f"  nippv_cpap_prior: NIPPV/CPAP during ICU stay before extubation")
    print(f"    -> {n_nippv_prior}/{N} ({n_nippv_prior / N * 100:.1f}%)")
    print(f"  hfno_prior: HFNO during ICU stay before extubation")
    print(f"    -> {n_hfno_prior}/{N} ({n_hfno_prior / N * 100:.1f}%)")
    print(f"  any_life_support_prior: Any of CRRT/vasopressor/NIPPV-CPAP/HFNO prior")
    print(f"    -> {n_any_ls_prior}/{N} ({n_any_ls_prior / N * 100:.1f}%)")
    print(f"  crrt_at_extubation: CRRT within 24h before extubation")
    print(f"    -> {n_crrt_at_ext}/{N} ({n_crrt_at_ext / N * 100:.1f}%)")
    print(f"  vasopressor_at_extubation: Vasopressor within 24h before extubation")
    print(f"    -> {n_vaso_at_ext}/{N} ({n_vaso_at_ext / N * 100:.1f}%)")
    print(f"  life_support_at_extubation: CRRT or vasopressor within 24h before extubation")
    print(f"    -> {n_ls_at_ext}/{N} ({n_ls_at_ext / N * 100:.1f}%)")
    print(f"  worst_sf_ratio_at_extubation: Worst (lowest) SpO2/FiO2 ratio in 24h before extubation")
    print(f"    -> {n_sf_nonnull}/{N} non-null ({n_sf_nonnull / N * 100:.1f}%)")
    if sf_median is not None:
        print(f"    -> Median: {sf_median:.1f}, IQR: [{sf_q25:.1f}, {sf_q75:.1f}]")
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
