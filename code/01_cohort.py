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
    from tqdm import tqdm
    from clifpy.tables import Hospitalization, RespiratorySupport, Adt, CodeStatus, Labs, Patient, Vitals, HospitalDiagnosis
    from clifpy import compute_sofa_polars
    return (
        Adt,
        CodeStatus,
        HospitalDiagnosis,
        Hospitalization,
        Labs,
        Path,
        Patient,
        RespiratorySupport,
        Vitals,
        compute_sofa_polars,
        json,
        mo,
        pd,
        pl,
        tqdm,
    )


@app.cell
def _(mo):
    mo.md("""
    # 01 Cohort Identification: Inclusion Criteria

    HFNO Post-Extubation Study - Step 1: Apply inclusion criteria only.
    """)
    return


@app.cell
def _(Path, json):
    # Load config from clif_config.json
    config_path = Path(__file__).parent.parent / "clif_config.json"
    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    SITE = config["site"]
    DATA_DIR = config["data_directory"]
    FILETYPE = config["filetype"]
    TIMEZONE = config["timezone"]

    print(f"Site: {SITE}")
    print(f"Data directory: {DATA_DIR}")
    return DATA_DIR, FILETYPE, SITE, TIMEZONE


@app.cell
def _(DATA_DIR, FILETYPE, Hospitalization, SITE, TIMEZONE, pd):
    # Load hospitalization table
    hosp = Hospitalization.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
    )
    hosp_df_raw = hosp.df.copy()
    n_total_hosp = len(hosp_df_raw)
    print(f"Total hospitalizations in database: {n_total_hosp}")

    # Inclusion 1: Age >= 18
    hosp_df_adult = hosp_df_raw[hosp_df_raw["age_at_admission"] >= 18].copy()
    n_after_age = len(hosp_df_adult)
    n_excluded_age = n_total_hosp - n_after_age
    print(f"After age >= 18: {n_after_age} (excluded {n_excluded_age})")

    # Inclusion 7: Admission date range (2018-01-01 to 2024-12-01)
    # Skip date filter for MIMIC (different date range)
    hosp_df_adult["admission_dttm"] = pd.to_datetime(hosp_df_adult["admission_dttm"])

    if SITE == "mimic":
        # MIMIC has different date range, skip date filter
        hosp_df = hosp_df_adult.copy()
        n_after_date = len(hosp_df)
        n_excluded_date = 0
        print(f"Date filter SKIPPED for MIMIC site: {n_after_date} (excluded {n_excluded_date})")
    else:
        # Apply date filter for other sites
        hosp_df = hosp_df_adult[
            (hosp_df_adult["admission_dttm"] >= "2018-01-01")
            & (hosp_df_adult["admission_dttm"] < "2024-12-01")
        ].copy()
        n_after_date = len(hosp_df)
        n_excluded_date = n_after_age - n_after_date
        print(f"After date filter (2018-2024): {n_after_date} (excluded {n_excluded_date})")

    hosp_df.head()
    return hosp_df, n_after_date, n_excluded_age, n_excluded_date, n_total_hosp


@app.cell
def _(
    CodeStatus,
    DATA_DIR,
    FILETYPE,
    HospitalDiagnosis,
    TIMEZONE,
    hosp_df,
    pl,
):
    # === Diagnosis-based patient-level exclusion ===
    # Load hospital diagnosis for all current hospitalization IDs
    dx = HospitalDiagnosis.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={"hospitalization_id": hosp_df["hospitalization_id"].unique().tolist()},
    )
    dx_pl = pl.from_pandas(dx.df)

    # Clean codes: lowercase, remove dots
    dx_pl = dx_pl.with_columns(
        pl.col("diagnosis_code").str.to_lowercase().str.replace_all(r"\.", "").alias("dx_clean")
    )

    EXCLUDED_DX_PREFIXES = [
       # "j9610",  # Chronic respiratory failure, unspecified
       # "j9611",  # Chronic respiratory failure with hypoxia
        "j9622",  # Acute and chronic respiratory failure with hypercapnia
        "g4733",  # Obstructive sleep apnea
        "g4730",  # Sleep apnea, unspecified
        "i501",   # Left ventricular failure
        "i5021",  # Acute systolic HF
        "z9989",  # CPAP/BiPAP dependence
        "z9911",  # Respirator/ventilator dependence
    ]

    # Find hospitalization_ids with ANY matching diagnosis code
    _dx_matched = dx_pl.filter(
        pl.any_horizontal([pl.col("dx_clean").str.starts_with(p) for p in EXCLUDED_DX_PREFIXES])
    )
    excluded_hosp_ids = _dx_matched["hospitalization_id"].unique()

    # --- ICD-10 exclusion breakdown ---
    _prefix_descriptions = {
       # "j9610": "Chronic respiratory failure, unspecified",
       # "j9611": "Chronic respiratory failure with hypoxia",
        "j9622": "Acute and chronic respiratory failure with hypercapnia",
        "g4733": "Obstructive sleep apnea",
        "g4730": "Sleep apnea, unspecified",
        "i501": "Left ventricular failure",
        "i5021": "Acute systolic HF",
        "z9989": "CPAP/BiPAP dependence",
        "z9911": "Respirator/ventilator dependence",
    }
    _breakdown_rows = []
    for prefix in EXCLUDED_DX_PREFIXES:
        _matched = _dx_matched.filter(pl.col("dx_clean").str.starts_with(prefix))
        _hosp_ids = _matched["hospitalization_id"].unique()
        _patient_ids = (
            pl.from_pandas(hosp_df[hosp_df["hospitalization_id"].isin(_hosp_ids.to_list())][["patient_id"]])
            ["patient_id"].unique()
        )
        _breakdown_rows.append({
            "icd10_prefix": prefix,
            "description": _prefix_descriptions[prefix],
            "n_hospitalizations_excluded": len(_hosp_ids),
            "n_patients_excluded": len(_patient_ids),
        })
    dx_exclusion_breakdown = pl.DataFrame(_breakdown_rows)
    print("\nICD-10 diagnosis exclusion breakdown:")
    for _row in dx_exclusion_breakdown.iter_rows(named=True):
        print(f"  {_row['icd10_prefix']} ({_row['description']}): {_row['n_hospitalizations_excluded']} hospitalizations, {_row['n_patients_excluded']} patients")

    # Map to patient_ids via hosp_df
    excluded_patient_ids = hosp_df[
        hosp_df["hospitalization_id"].isin(excluded_hosp_ids.to_list())
    ]["patient_id"].unique()

    # Remove ALL hospitalizations for those patients
    n_before_dx = len(hosp_df)
    hosp_df_after_dx = hosp_df[~hosp_df["patient_id"].isin(excluded_patient_ids)].copy()
    n_excluded_dx = n_before_dx - len(hosp_df_after_dx)
    n_after_dx = len(hosp_df_after_dx)
    print(f"After diagnosis exclusion: {n_after_dx} (excluded {n_excluded_dx} patients with acute-chronic resp failure/OSA/HF/CPAP-BiPAP)")

    # Get patient IDs from hospitalizations
    patient_ids = hosp_df_after_dx["patient_id"].unique().tolist()

    # Load code status table (keyed by patient_id, not hospitalization_id)
    code_status = CodeStatus.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={"patient_id": patient_ids},
    )
    code_status_pd = code_status.df.copy()
    code_status_pd["start_dttm"] = code_status_pd["start_dttm"].dt.tz_localize(None)
    code_status_pl = pl.from_pandas(code_status_pd)
    del code_status_pd

    # Normalize category to lowercase
    code_status_pl = code_status_pl.with_columns(
        pl.col("code_status_category").str.to_lowercase().alias("code_status_lower")
    )

    print(f"Code status records loaded: {len(code_status_pl)}")

    # Pass through all hospitalizations (code status filter applied later, after extubation detection)
    hosp_df_filtered = hosp_df_after_dx.copy()
    return (
        code_status_pl,
        dx_exclusion_breakdown,
        hosp_df_filtered,
        n_after_dx,
        n_excluded_dx,
    )


@app.cell
def _(Adt, DATA_DIR, FILETYPE, TIMEZONE, hosp_df_filtered, pl, tqdm):
    hosp_ids = hosp_df_filtered["hospitalization_id"].unique().tolist()

    adt = Adt.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={"hospitalization_id": hosp_ids},
    )

    # Strip timezone in pandas before converting to Polars
    adt_pd = adt.df
    adt_pd["in_dttm"] = adt_pd["in_dttm"].dt.tz_localize(None)
    adt_pd["out_dttm"] = adt_pd["out_dttm"].dt.tz_localize(None)

    # Convert to Polars
    adt_df = pl.from_pandas(adt_pd)
    del adt, adt_pd  # Free pandas memory

    # Filter to only hospitalizations with at least 1 ICU stay before merging
    hosp_with_icu = set(
        adt_df.filter(pl.col("location_category") == "icu")["hospitalization_id"]
        .unique()
        .to_list()
    )
    n_with_icu = len(hosp_with_icu)
    n_excluded_no_icu = len(hosp_ids) - n_with_icu
    print(f"Hospitalizations with ICU stay: {n_with_icu} (excluded {n_excluded_no_icu} without ICU)")

    # Filter to only ICU hospitalizations for merge
    adt_df = adt_df.filter(pl.col("hospitalization_id").is_in(list(hosp_with_icu)))

    # Sort by hospitalization_id and in_dttm
    adt_df = adt_df.sort(["hospitalization_id", "in_dttm"])

    # Exclude ICU stays with null out_dttm
    n_null_out = adt_df.filter(
        (pl.col("location_category") == "icu") & pl.col("out_dttm").is_null()
    ).height
    adt_df = adt_df.filter(
        ~((pl.col("location_category") == "icu") & pl.col("out_dttm").is_null())
    )
    print(f"Excluded {n_null_out} ICU ADT rows with null out_dttm")

    # Recompute hosp_with_icu after removing null out_dttm ICU rows
    hosp_with_icu = set(
        adt_df.filter(pl.col("location_category") == "icu")["hospitalization_id"]
        .unique().to_list()
    )
    n_excluded_null_out = n_with_icu - len(hosp_with_icu)
    n_with_icu = len(hosp_with_icu)
    print(f"Hospitalizations excluded (all ICU stays had null out_dttm): {n_excluded_null_out}")

    def merge_icu_stays(df: pl.DataFrame) -> pl.DataFrame:
        """Merge consecutive ICU stays (direct ICU→ICU or ICU→Procedural→ICU)."""
        merged_rows = []
        hosp_ids_unique = df["hospitalization_id"].unique().to_list()

        for hosp_id in tqdm(hosp_ids_unique, desc="Merging consecutive ICU stays"):
            group = df.filter(pl.col("hospitalization_id") == hosp_id).sort("in_dttm")
            rows = group.to_dicts()
            i = 0
            while i < len(rows):
                row = rows[i].copy()

                if row["location_category"] == "icu":
                    j = i + 1
                    while j < len(rows):
                        next_row = rows[j]

                        # Direct ICU → ICU transition
                        if next_row["location_category"] == "icu":
                            row["out_dttm"] = max(row["out_dttm"], next_row["out_dttm"])
                            j += 1
                        # ICU → Procedural → ICU transition
                        elif (next_row["location_category"] == "procedural"
                              and j + 1 < len(rows)
                              and rows[j + 1]["location_category"] == "icu"):
                            row["out_dttm"] = max(row["out_dttm"], rows[j + 1]["out_dttm"])
                            j += 2
                        else:
                            break
                    i = j
                else:
                    i += 1

                merged_rows.append(row)

        return pl.DataFrame(merged_rows)

    adt_df = merge_icu_stays(adt_df)

    icu_adt_df = adt_df.filter(pl.col("location_category") == "icu")

    print(f"Total ADT records (after merging consecutive ICU stays): {len(adt_df)}")
    print(f"ICU ADT records: {len(icu_adt_df)}")
    return (
        adt_df,
        hosp_with_icu,
        icu_adt_df,
        n_excluded_no_icu,
        n_excluded_null_out,
        n_with_icu,
    )


@app.cell
def _(
    DATA_DIR,
    FILETYPE,
    Path,
    RespiratorySupport,
    TIMEZONE,
    hosp_with_icu,
    n_with_icu,
    pd,
    pl,
):
    # Load respiratory support ONLY for hospitalizations with ICU stays (more efficient)
    resp = RespiratorySupport.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={"hospitalization_id": list(hosp_with_icu)},
    )
    resp_df_raw = resp.df.copy()
    resp_df_raw = resp_df_raw.sort_values(["hospitalization_id", "recorded_dttm"])
    print(f"Respiratory support records (for {len(hosp_with_icu)} ICU hospitalizations): {len(resp_df_raw)}")

    # Identify hospitalizations that ever had tracheostomy
    hosp_ever_trach = set(resp_df_raw[resp_df_raw["tracheostomy"] == 1]["hospitalization_id"].unique())

    # Identify hospitalizations that ever had IMV
    hosp_ever_imv = set(resp_df_raw[resp_df_raw["device_category"] == "IMV"]["hospitalization_id"].unique())

    # Keep only hospitalizations with IMV but never tracheostomy
    hosp_imv_no_trach = hosp_ever_imv - hosp_ever_trach

    # Track exclusion counts
    n_excluded_trach = len(hosp_ever_imv & hosp_ever_trach)
    n_excluded_no_imv = n_with_icu - len(hosp_ever_imv)
    n_with_imv_no_trach = len(hosp_imv_no_trach)

    print(f"Hospitalizations with any IMV: {len(hosp_ever_imv)}")
    print(f"Hospitalizations with tracheostomy (excluded): {n_excluded_trach}")
    print(f"Hospitalizations without any IMV (excluded): {n_excluded_no_imv}")
    print(f"Hospitalizations with IMV and no tracheostomy: {n_with_imv_no_trach}")

    # Define output path for waterfall cache
    _waterfall_output_dir = Path(__file__).parent.parent / "output"
    waterfall_path = _waterfall_output_dir / "resp_support_waterfall.parquet"

    # Check cache and load or run waterfall
    if waterfall_path.exists():
        print(f"Loading cached waterfall data from: {waterfall_path}")
        resp_df = pd.read_parquet(waterfall_path)
        # Filter to current cohort (hosp_imv_no_trach)
        resp_df = resp_df[resp_df["hospitalization_id"].isin(hosp_imv_no_trach)].copy()
    else:
        # Filter resp to qualifying hospitalizations first
        resp.df = resp.df[resp.df["hospitalization_id"].isin(hosp_imv_no_trach)].copy()

        # Run waterfall on the resp object
        resp_filled = resp.waterfall(bfill=False, verbose=True)
        resp_df = resp_filled.df.copy()

        # Save to cache
        _waterfall_output_dir.mkdir(parents=True, exist_ok=True)
        resp_df.to_parquet(waterfall_path, index=False)
        print(f"Saved waterfall data to: {waterfall_path}")

    # Delete the raw copy to free memory
    del resp_df_raw

    # Strip timezone in pandas before converting to Polars
    resp_df["recorded_dttm"] = resp_df["recorded_dttm"].dt.tz_localize(None)

    # Convert to Polars
    resp_df = pl.from_pandas(resp_df)

    # HFNO cleanup: reclassify and cap flow rates
    n_before_hfno_clean = resp_df.height
    n_reclassified = resp_df.filter(
        (pl.col("device_category") == "high flow nc") & (pl.col("lpm_set") < 30)
    ).height
    n_capped = resp_df.filter(
        (pl.col("device_category") == "high flow nc") & (pl.col("lpm_set") > 60)
    ).height

    resp_df = resp_df.with_columns(
        pl.when(
            (pl.col("device_category") == "high flow nc") & (pl.col("lpm_set") < 30)
        )
        .then(pl.lit("low flow nc"))
        .otherwise(pl.col("device_category"))
        .alias("device_category")
    ).with_columns(
        pl.when(
            (pl.col("device_category") == "high flow nc") & (pl.col("lpm_set") > 60)
        )
        .then(pl.lit(60.0))
        .otherwise(pl.col("lpm_set"))
        .alias("lpm_set")
    )

    print(f"HFNO cleanup: {n_reclassified} rows reclassified to low flow nc (lpm < 30), {n_capped} rows capped at 60 lpm")

    print(f"Respiratory support records after waterfall processing: {len(resp_df)}")
    return n_excluded_no_imv, n_excluded_trach, n_with_imv_no_trach, resp_df


@app.cell
def _(icu_adt_df, pl, resp_df):
    # Step A: First ICU stay per hospitalization
    # Filter to only hospitalizations in the IMV-no-trach set (resp_df is already filtered)
    imv_no_trach_ids = resp_df["hospitalization_id"].unique()
    icu_filtered = icu_adt_df.filter(pl.col("hospitalization_id").is_in(imv_no_trach_ids))

    # Rank ICU stays by in_dttm within each hospitalization, keep the earliest
    first_icu = (
        icu_filtered
        .sort(["hospitalization_id", "in_dttm"])
        .group_by("hospitalization_id")
        .first()
    )

    # Rename ICU timing columns for clarity
    first_icu = first_icu.rename({"in_dttm": "icu_start", "out_dttm": "icu_end", "location_type": "icu_type"})

    # Compute ICU LOS in hours
    first_icu = first_icu.with_columns(
        ((pl.col("icu_end") - pl.col("icu_start")).dt.total_seconds() / 3600)
        .alias("icu_los_hours")
    )

    # Find ICU readmissions: earliest ICU stay starting after the first ICU ended
    readmissions = (
        icu_filtered
        .rename({"in_dttm": "readmission_icu_start", "out_dttm": "readmission_icu_end"})
        .join(
            first_icu.select(["hospitalization_id", "icu_end"]),
            on="hospitalization_id",
            how="inner",
        )
        .filter(pl.col("readmission_icu_start") > pl.col("icu_end"))
        .sort(["hospitalization_id", "readmission_icu_start"])
        .group_by("hospitalization_id")
        .first()
        .select(["hospitalization_id", "readmission_icu_start"])
    )

    first_icu = first_icu.join(readmissions, on="hospitalization_id", how="left")
    first_icu = first_icu.with_columns([
        pl.col("readmission_icu_start").is_not_null().alias("readmission_to_icu"),
        ((pl.col("readmission_icu_start") - pl.col("icu_end")).dt.total_seconds() / 3600)
        .alias("hours_to_icu_readmission"),
    ])

    n_readmit = first_icu.filter(pl.col("readmission_to_icu")).height
    print(f"First ICU stays (from IMV no-trach set): {len(first_icu)}")
    print(f"ICU readmissions detected: {n_readmit} / {len(first_icu)}")
    return (first_icu,)


@app.cell
def _(first_icu, pl, resp_df):
    # Step B: Clip respiratory support to before ICU end (allow pre-ICU records for intubation location)
    resp_clipped = (
        resp_df
        .join(
            first_icu.select(["hospitalization_id", "icu_start", "icu_end"]),
            on="hospitalization_id",
            how="inner",
        )
        .filter(pl.col("recorded_dttm") < pl.col("icu_end"))
        .sort(["hospitalization_id", "recorded_dttm"])
    )

    print(f"Respiratory records clipped to before ICU end: {len(resp_clipped)}")
    print(f"Hospitalizations in clipped set: {resp_clipped['hospitalization_id'].n_unique()}")

    # IDs with no respiratory data in ICU window
    all_icu_ids = first_icu.select("hospitalization_id")
    clipped_ids = resp_clipped.select("hospitalization_id").unique()
    ids_no_resp_data = all_icu_ids.filter(
        ~pl.col("hospitalization_id").is_in(clipped_ids["hospitalization_id"])
    )
    n_excluded_no_resp = len(ids_no_resp_data)
    n_after_resp = resp_clipped["hospitalization_id"].n_unique()
    print(f"Excluded {n_excluded_no_resp} hospitalizations with no resp data before ICU end")

    # IDs with no IMV before ICU end (have resp data, but no IMV in window)
    ids_with_imv_before_icu = resp_clipped.filter(
        pl.col("device_category") == "imv"
    ).select("hospitalization_id").unique()

    ids_no_imv_before_icu = clipped_ids.filter(
        ~pl.col("hospitalization_id").is_in(ids_with_imv_before_icu["hospitalization_id"])
    )
    n_excluded_no_imv_before_icu = len(ids_no_imv_before_icu)

    # Filter resp_clipped to only hospitalizations with IMV before ICU end
    resp_clipped = resp_clipped.filter(
        pl.col("hospitalization_id").is_in(ids_with_imv_before_icu["hospitalization_id"])
    )
    n_after_imv_before_icu = resp_clipped["hospitalization_id"].n_unique()
    print(f"Excluded {n_excluded_no_imv_before_icu} hospitalizations with no IMV before ICU end")
    print(f"Hospitalizations with IMV before ICU end: {n_after_imv_before_icu}")
    return (
        n_after_imv_before_icu,
        n_after_resp,
        n_excluded_no_imv_before_icu,
        n_excluded_no_resp,
        resp_clipped,
    )


@app.cell
def _(pl, resp_clipped):
    # Step C: Detect intubation and extubation with 2-row confirmation
    # Waterfall lowercases device_category, so use "imv" not "IMV"

    # Step 1 — Sort, forward-fill, compute is_imv, then materialize
    resp_sorted = (
        resp_clipped
        .sort(["hospitalization_id", "recorded_dttm"])
        .with_columns(
            pl.col("device_category")
            .forward_fill()
            .over("hospitalization_id")
            .alias("device_category")
        )
        .with_columns(
            (pl.col("device_category") == "imv").cast(pl.Int8).alias("is_imv")
        )
    )

    # Step 2 — Compute lag/lead on the materialized, already-sorted DF
    resp_flagged = resp_sorted.with_columns([
        pl.col("is_imv").shift(1).over("hospitalization_id").alias("lag1"),
        pl.col("is_imv").shift(2).over("hospitalization_id").alias("lag2"),
        pl.col("is_imv").shift(-1).over("hospitalization_id").alias("lead1"),
        pl.col("is_imv").shift(-2).over("hospitalization_id").alias("lead2"),
    ])

    # Add lead1_device: the device category of the next row (for post-extubation device)
    resp_flagged = resp_flagged.with_columns(
        pl.col("device_category").shift(-1).over("hospitalization_id").alias("lead1_device")
    )

    # Intubation: is_imv==1, (lag1==0 or null), (lag2==0 or null)
    intubations = resp_flagged.filter(
        (pl.col("is_imv") == 1)
        & ((pl.col("lag1") == 0) | pl.col("lag1").is_null())
        & ((pl.col("lag2") == 0) | pl.col("lag2").is_null())
    )

    # First intubation per hospitalization
    first_intubation = (
        intubations
        .sort(["hospitalization_id", "recorded_dttm"])
        .group_by("hospitalization_id")
        .first()
        .select([
            "hospitalization_id",
            pl.col("recorded_dttm").alias("intubation_time"),
        ])
    )

    n_with_intubation = len(first_intubation)
    all_ids = resp_clipped.select("hospitalization_id").unique()
    ids_no_intubation = all_ids.filter(
        ~pl.col("hospitalization_id").is_in(first_intubation["hospitalization_id"])
    )
    n_no_intubation = len(ids_no_intubation)
    print(f"Hospitalizations with confirmed intubation: {n_with_intubation}")
    print(f"Hospitalizations without confirmed intubation: {n_no_intubation}")

    # Extubation: current is IMV, next is not IMV, and the one after is also not IMV
    extubations = resp_flagged.filter(
        (pl.col("is_imv") == 1)
        & (pl.col("lead1") == 0)
        & ((pl.col("lead2") == 0) | pl.col("lead2").is_null())
    )

    # Join extubations with intubation times, keep only extubations after intubation
    extub_with_intub = (
        extubations
        .join(first_intubation, on="hospitalization_id", how="inner")
        .filter(pl.col("recorded_dttm") > pl.col("intubation_time"))
        .sort(["hospitalization_id", "recorded_dttm"])
        .group_by("hospitalization_id")
        .first()
        .select([
            "hospitalization_id",
            "intubation_time",
            pl.col("recorded_dttm").alias("extubation_time"),
            pl.col("lead1_device").alias("device_after_extubation"),
        ])
    )

    # Compute IMV duration
    intub_extub = extub_with_intub.with_columns(
        ((pl.col("extubation_time") - pl.col("intubation_time")).dt.total_seconds() / 3600)
        .alias("imv_duration_hours")
    )

    n_with_extubation = len(intub_extub)
    n_no_extubation = n_with_intubation - n_with_extubation
    print(f"Hospitalizations with confirmed extubation: {n_with_extubation}")
    print(f"Hospitalizations without confirmed extubation: {n_no_extubation}")
    return (
        first_intubation,
        intub_extub,
        n_no_extubation,
        n_no_intubation,
        n_with_extubation,
        n_with_intubation,
    )


@app.cell
def _(DATA_DIR, FILETYPE, Labs, TIMEZONE, intub_extub, pl):
    # Step C2: Load PaCO2 labs and find the latest value before extubation
    hosp_ids_for_labs = intub_extub["hospitalization_id"].to_list()
    paco2_labs = Labs.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={
            "hospitalization_id": hosp_ids_for_labs,
            "lab_category": ["pco2_arterial"],
        },
    )
    paco2_df = paco2_labs.df.copy()
    paco2_df["lab_result_dttm"] = paco2_df["lab_result_dttm"].dt.tz_localize(None)
    paco2_pl = pl.from_pandas(paco2_df)
    del paco2_df

    # Join with intub_extub to get extubation_time, filter to before extubation
    paco2_pre = (
        paco2_pl
        .join(intub_extub.select(["hospitalization_id", "extubation_time"]),
              on="hospitalization_id", how="inner")
        .filter(pl.col("lab_result_dttm") < pl.col("extubation_time"))
        .sort(["hospitalization_id", "lab_result_dttm"])
        .group_by("hospitalization_id").last()
        .with_columns(
            ((pl.col("extubation_time") - pl.col("lab_result_dttm")).dt.total_seconds() / 3600)
            .alias("paco2_to_extubation_hours")
        )
        .select([
            "hospitalization_id",
            pl.col("lab_value_numeric").alias("paco2_pre_extubation"),
            pl.col("lab_result_dttm").alias("paco2_pre_extubation_dttm"),
            "paco2_to_extubation_hours",
        ])
    )
    print(f"PaCO2 pre-extubation values found: {len(paco2_pre)}")
    return (paco2_pre,)


@app.cell
def _(DATA_DIR, FILETYPE, TIMEZONE, Vitals, intub_extub, pl):
    # Step C2b: Load Vitals for height and weight before extubation
    hosp_ids_for_vitals = intub_extub["hospitalization_id"].to_list()
    vitals_table = Vitals.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={
            "hospitalization_id": hosp_ids_for_vitals,
            "vital_category": ["height_cm", "weight_kg"],
        },
    )
    vitals_df = vitals_table.df.copy()
    vitals_df["recorded_dttm"] = vitals_df["recorded_dttm"].dt.tz_localize(None)
    vitals_pl = pl.from_pandas(vitals_df)
    del vitals_df

    # Join with intub_extub to get extubation_time, filter to before extubation
    vitals_pre_extub = (
        vitals_pl
        .join(
            intub_extub.select(["hospitalization_id", "extubation_time"]),
            on="hospitalization_id",
            how="inner",
        )
        .filter(pl.col("recorded_dttm") < pl.col("extubation_time"))
        .sort(["hospitalization_id", "recorded_dttm"])
        .group_by(["hospitalization_id", "vital_category"])
        .last()
        .select(["hospitalization_id", "vital_category", "vital_value"])
        .pivot(
            on="vital_category",
            index="hospitalization_id",
            values="vital_value",
        )
    )

    # Ensure both columns exist even if no data for one category
    if "height_cm" not in vitals_pre_extub.columns:
        vitals_pre_extub = vitals_pre_extub.with_columns(pl.lit(None).cast(pl.Float64).alias("height_cm"))
    if "weight_kg" not in vitals_pre_extub.columns:
        vitals_pre_extub = vitals_pre_extub.with_columns(pl.lit(None).cast(pl.Float64).alias("weight_kg"))

    vitals_pre_extub = vitals_pre_extub.select(["hospitalization_id", "height_cm", "weight_kg"])

    print(f"Vitals pre-extubation: {len(vitals_pre_extub)} hospitalizations")
    print(f"  height_cm non-null: {vitals_pre_extub['height_cm'].is_not_null().sum()}")
    print(f"  weight_kg non-null: {vitals_pre_extub['weight_kg'].is_not_null().sum()}")
    return (vitals_pre_extub,)


@app.cell
def _(
    DATA_DIR,
    FILETYPE,
    TIMEZONE,
    compute_sofa_polars,
    first_icu,
    intub_extub,
    pl,
):
    # Build cohort_df for SOFA at ICU admission (first 24h)
    sofa_admission_cohort = (
        intub_extub.select("hospitalization_id")
        .join(first_icu.select(["hospitalization_id", "icu_start"]), on="hospitalization_id", how="inner")
        .with_columns([
            pl.col("icu_start").alias("start_dttm"),
            (pl.col("icu_start") + pl.duration(hours=24)).alias("end_dttm"),
        ])
        .select(["hospitalization_id", "start_dttm", "end_dttm"])
    )

    sofa_admission = compute_sofa_polars(
        data_directory=DATA_DIR,
        cohort_df=sofa_admission_cohort,
        filetype=FILETYPE,
        timezone=TIMEZONE,
    ).select([
        "hospitalization_id",
        pl.col("sofa_total").alias("sofa_icu_admission"),
    ])

    # Build cohort_df for SOFA at extubation (24h before extubation)
    sofa_extub_cohort = (
        intub_extub.select(["hospitalization_id", "extubation_time"])
        .with_columns([
            (pl.col("extubation_time") - pl.duration(hours=24)).alias("start_dttm"),
            pl.col("extubation_time").alias("end_dttm"),
        ])
        .select(["hospitalization_id", "start_dttm", "end_dttm"])
    )

    sofa_extubation = compute_sofa_polars(
        data_directory=DATA_DIR,
        cohort_df=sofa_extub_cohort,
        filetype=FILETYPE,
        timezone=TIMEZONE,
    ).select([
        "hospitalization_id",
        pl.col("sofa_total").alias("sofa_extubation"),
    ])

    # Combine into one DF
    sofa_scores = (
        sofa_admission
        .join(sofa_extubation, on="hospitalization_id", how="outer_coalesce")
    )

    print(f"SOFA scores computed: {len(sofa_scores)} hospitalizations")
    print(f"  sofa_icu_admission non-null: {sofa_scores['sofa_icu_admission'].is_not_null().sum()}")
    print(f"  sofa_extubation non-null: {sofa_scores['sofa_extubation'].is_not_null().sum()}")
    return (sofa_scores,)


@app.cell
def _(adt_df, first_icu, intub_extub, pl):
    # Step C3: Intubation/extubation location and pre-ICU trajectory

    # --- Intubation location ---
    intub_location = (
        intub_extub.select(["hospitalization_id", "intubation_time"])
        .join(adt_df, on="hospitalization_id", how="inner")
        .filter(
            (pl.col("in_dttm") <= pl.col("intubation_time"))
            & (pl.col("intubation_time") < pl.col("out_dttm"))
        )
        .sort(["hospitalization_id", "in_dttm"], descending=[False, True])
        .group_by("hospitalization_id").first()
        .select([
            "hospitalization_id",
            pl.col("location_category").alias("intubation_location_category"),
        ])
    )

    # --- Extubation location ---
    extub_location = (
        intub_extub.select(["hospitalization_id", "extubation_time"])
        .join(adt_df, on="hospitalization_id", how="inner")
        .filter(
            (pl.col("in_dttm") <= pl.col("extubation_time"))
            & (pl.col("extubation_time") < pl.col("out_dttm"))
        )
        .sort(["hospitalization_id", "in_dttm"], descending=[False, True])
        .group_by("hospitalization_id").first()
        .select([
            "hospitalization_id",
            pl.col("location_category").alias("extubation_location_category"),
        ])
    )

    # --- Pre-ICU trajectory ---
    pre_icu_records = (
        adt_df
        .join(
            first_icu.select(["hospitalization_id", "icu_start"]),
            on="hospitalization_id", how="inner",
        )
        .filter(pl.col("out_dttm") <= pl.col("icu_start"))
        .filter(pl.col("location_category") != "icu")
        .sort(["hospitalization_id", "in_dttm"])
    )

    last_pre_icu = (
        pre_icu_records
        .group_by("hospitalization_id").last()
        .select([
            "hospitalization_id",
            pl.col("location_category").alias("pre_icu_location_category"),
        ])
    )

    pre_icu_trajectory = (
        pre_icu_records
        .group_by("hospitalization_id")
        .agg(pl.col("location_category"))
        .with_columns(
            pl.col("location_category").list.join(",").alias("pre_icu_trajectory")
        )
        .select(["hospitalization_id", "pre_icu_trajectory"])
    )

    # Combine
    location_info = (
        intub_extub.select("hospitalization_id")
        .join(intub_location, on="hospitalization_id", how="left")
        .join(extub_location, on="hospitalization_id", how="left")
        .join(last_pre_icu, on="hospitalization_id", how="left")
        .join(pre_icu_trajectory, on="hospitalization_id", how="left")
    )

    print(f"Intubation location found: {intub_location.height}")
    print(f"Extubation location found: {extub_location.height}")
    print(f"Pre-ICU trajectory found: {last_pre_icu.height}")
    return (location_info,)


@app.cell
def _(
    DATA_DIR,
    FILETYPE,
    Patient,
    TIMEZONE,
    code_status_pl,
    first_icu,
    hosp_df_filtered,
    intub_extub,
    location_info,
    paco2_pre,
    pd,
    pl,
    resp_df,
    sofa_scores,
    vitals_pre_extub,
):
    # Step D: Build wide one-row-per-hospitalization dataset
    # Filter to rows with non-null extubation_time
    extub_valid = intub_extub.filter(pl.col("extubation_time").is_not_null())

    # Join with first_icu for ICU timing columns
    wide = extub_valid.join(
        first_icu.select([
            "hospitalization_id", "icu_start", "icu_end",
            "readmission_to_icu", "readmission_icu_start", "hours_to_icu_readmission",
            "icu_type", "icu_los_hours",
        ]),
        on="hospitalization_id",
        how="left",
    )

    # Left join PaCO2 pre-extubation values
    wide = wide.join(
        paco2_pre,
        on="hospitalization_id",
        how="left",
    )

    # Left join location info (intubation/extubation location and pre-ICU trajectory)
    wide = wide.join(
        location_info,
        on="hospitalization_id",
        how="left",
    )

    # Left join vitals (height/weight) pre-extubation
    wide = wide.join(
        vitals_pre_extub,
        on="hospitalization_id",
        how="left",
    )

    # Left join SOFA scores
    wide = wide.join(sofa_scores, on="hospitalization_id", how="left")

    # Compute ICU LOS before extubation
    wide = wide.with_columns(
        ((pl.col("extubation_time") - pl.col("icu_start")).dt.total_seconds() / 3600)
        .alias("icu_los_before_extubation_hours")
    )

    # Exclude hospitalizations where extubation did not occur in ICU
    n_before_extub_loc = len(wide)
    wide = wide.filter(pl.col("extubation_location_category") == "icu")
    n_excluded_extub_not_icu = n_before_extub_loc - len(wide)
    n_after_extub_loc = len(wide)
    print(f"After extubation-in-ICU filter: {n_after_extub_loc} (excluded {n_excluded_extub_not_icu} extubated outside ICU)")

    # Full code status at extubation: last code status before extubation must be Full/Presume Full
    hosp_patient_map = pl.from_pandas(
        hosp_df_filtered[["hospitalization_id", "patient_id"]]
    )

    last_code_before_extub = (
        code_status_pl
        .join(hosp_patient_map, on="patient_id", how="inner")
        .join(
            wide.select(["hospitalization_id", "extubation_time"]),
            on="hospitalization_id",
            how="inner",
        )
        .filter(pl.col("start_dttm") <= pl.col("extubation_time"))
        .sort(["hospitalization_id", "start_dttm"])
        .group_by("hospitalization_id")
        .last()
        .select(["hospitalization_id", "code_status_lower"])
    )

    EXCLUDED_CODE_STATUSES = [
        "dnr", "dnar", "udnr", "dnr/dni", "dnar/dni", "dni_only", "and",
    ]
    non_full_code_ids = last_code_before_extub.filter(
        pl.col("code_status_lower").is_in(EXCLUDED_CODE_STATUSES)
    )["hospitalization_id"]

    n_before_code = len(wide)
    wide = wide.filter(~pl.col("hospitalization_id").is_in(non_full_code_ids))
    n_excluded_not_full_code = n_before_code - len(wide)
    n_after_full_code = len(wide)
    print(f"After code status exclusion: {n_after_full_code} (excluded {n_excluded_not_full_code} with DNR/DNI/DNAR/AND code status)")

    # Exclude hospitalizations with PaCO2 > 50 mmHg pre-extubation
    n_before_paco2 = len(wide)
    wide = wide.filter(
        (pl.col("paco2_pre_extubation").is_null()) | (pl.col("paco2_pre_extubation") <= 45)
    )
    n_excluded_paco2 = n_before_paco2 - len(wide)
    n_after_paco2 = len(wide)
    print(f"After PaCO2 <= 45 filter: {n_after_paco2} (excluded {n_excluded_paco2} with PaCO2 > 45)")

    # Exclude hospitalizations with IMV duration < 12 hours
    n_before_imv_dur = len(wide)
    wide = wide.filter(pl.col("imv_duration_hours") >= 12)
    n_excluded_imv_dur = n_before_imv_dur - len(wide)
    n_after_imv_dur = len(wide)
    print(f"After IMV duration >= 12h filter: {n_after_imv_dur} (excluded {n_excluded_imv_dur} with IMV < 12h)")

    # === Snapshot after Step 15: all clinically eligible extubated patients ===
    wide_eligible = wide.clone()

    # Step 16 (HFNO path): Exclude patients extubated to NIPPV or CPAP
    n_before_nippv_cpap = len(wide)
    wide = wide.filter(~pl.col("device_after_extubation").is_in(["nippv", "cpap"]))
    n_excluded_nippv_cpap_extub = n_before_nippv_cpap - len(wide)
    n_after_nippv_cpap = len(wide)
    print(f"After excluding NIPPV/CPAP extubations: {n_after_nippv_cpap} (excluded {n_excluded_nippv_cpap_extub} extubated to NIPPV/CPAP)")

    # 2-hour post-extubation window: >= 1h cumulative HFNC with lpm_set >= 30
    window_resp = (
        resp_df
        .join(
            wide.select(["hospitalization_id", "extubation_time"]),
            on="hospitalization_id",
            how="inner",
        )
        .filter(
            (pl.col("recorded_dttm") > pl.col("extubation_time"))
            & (pl.col("recorded_dttm") <= pl.col("extubation_time") + pl.duration(hours=2))
        )
    )

    # Only keep patients whose first device in the window is HFNC
    window_resp = (
        window_resp
        .sort("hospitalization_id", "recorded_dttm")
        .with_columns(
            pl.col("device_category")
            .first()
            .over("hospitalization_id")
            .alias("first_device"),
        )
        .filter(pl.col("first_device") == "high flow nc")
    )

    # Forward-fill lpm_set within each hospitalization so rows where
    # flow rate was not re-charted still count toward cumulative duration
    window_resp = (
        window_resp
        .with_columns(
            pl.col("lpm_set")
            .forward_fill()
            .over("hospitalization_id")
            .alias("lpm_set"),
        )
    )

    # Compute duration each row is "active" from consecutive timestamps
    window_resp = (
        window_resp
        .with_columns(
            pl.col("recorded_dttm")
            .shift(-1)
            .over("hospitalization_id")
            .alias("next_recorded_dttm"),
        )
        .with_columns(
            # Clip next timestamp to window end; use window end if no next record
            pl.min_horizontal(
                pl.col("next_recorded_dttm"),
                pl.col("extubation_time") + pl.duration(hours=2),
            )
            .fill_null(pl.col("extubation_time") + pl.duration(hours=2))
            .alias("effective_end"),
        )
        .with_columns(
            ((pl.col("effective_end") - pl.col("recorded_dttm")).dt.total_seconds() / 60)
            .alias("duration_minutes"),
        )
        .with_columns(
            ((pl.col("device_category") == "high flow nc") & pl.col("lpm_set").is_not_null() & (pl.col("lpm_set") >= 30))
            .alias("is_qualifying"),
        )
    )

    window_summary = (
        window_resp
        .group_by("hospitalization_id")
        .agg(
            pl.col("duration_minutes")
            .filter(pl.col("is_qualifying"))
            .sum()
            .fill_null(0)
            .alias("hfnc_gte30_minutes"),
        )
    )

    # Step 17: >= 1h cumulative HFNC with lpm_set >= 30 in 2h post-extubation
    qualifying_ids = window_summary.filter(pl.col("hfnc_gte30_minutes") >= 60)["hospitalization_id"]
    n_before_hfnc_lpm = len(wide)
    wide_before_hfnc = wide.clone()
    wide = wide.filter(pl.col("hospitalization_id").is_in(qualifying_ids))
    n_excluded_hfnc_lpm = n_before_hfnc_lpm - len(wide)
    n_after_hfnc_lpm = len(wide)
    print(f"After >= 1h HFNC lpm>=30 in 2h window: {n_after_hfnc_lpm} (excluded {n_excluded_hfnc_lpm})")

    # Print device breakdown for excluded patients
    _excluded_at_hfnc = wide_before_hfnc.filter(~pl.col("hospitalization_id").is_in(qualifying_ids))
    _device_breakdown = _excluded_at_hfnc.group_by("device_after_extubation").len().sort("len", descending=True)
    print("  Device breakdown of excluded patients:")
    for row in _device_breakdown.iter_rows(named=True):
        print(f"    {row['device_after_extubation']}: {row['len']}")

    # === Derive low-flow group ===
    # Includes: (a) patients whose first post-extubation device was low-flow,
    #           (b) patients who started on HFNC but failed the >=1h threshold
    LOW_FLOW_DEVICES = ["face mask", "nasal cannula", "room air", "other", "low flow nc"]
    hfno_ids = set(wide["hospitalization_id"].to_list())
    failed_hfnc_ids = set(
        window_summary.filter(pl.col("hfnc_gte30_minutes") < 60)
        ["hospitalization_id"].to_list()
    )
    low_flow_wide = wide_eligible.filter(
        ~pl.col("hospitalization_id").is_in(hfno_ids)
        & (
            pl.col("device_after_extubation").is_in(LOW_FLOW_DEVICES)
            | pl.col("hospitalization_id").is_in(failed_hfnc_ids)
        )
    )
    n_failed_hfnc_in_lf = len(low_flow_wide.filter(pl.col("hospitalization_id").is_in(failed_hfnc_ids)))
    n_native_lf = len(low_flow_wide) - n_failed_hfnc_in_lf
    n_excluded_nippv_cpap = len(wide_eligible.filter(~pl.col("hospitalization_id").is_in(hfno_ids))) - len(low_flow_wide)
    print(f"\nLow-flow group: {len(low_flow_wide)} total")
    print(f"  Started on low-flow device: {n_native_lf}")
    print(f"  Failed HFNC reclassified to low-flow: {n_failed_hfnc_in_lf}")
    print(f"  Excluded (NIPPV/CPAP/other): {n_excluded_nippv_cpap}")

    # Same column selection for both groups
    _keep_cols = [
        "hospitalization_id",
        "icu_start",
        "icu_end",
        "intubation_time",
        "extubation_time",
        "imv_duration_hours",
        "device_after_extubation",
        "paco2_pre_extubation",
        "paco2_pre_extubation_dttm",
        "paco2_to_extubation_hours",
        "intubation_location_category",
        "extubation_location_category",
        "pre_icu_location_category",
        "pre_icu_trajectory",
        "readmission_to_icu",
        "readmission_icu_start",
        "hours_to_icu_readmission",
        "icu_type",
        "icu_los_hours",
        "icu_los_before_extubation_hours",
        "height_cm",
        "weight_kg",
        "sofa_icu_admission",
        "sofa_extubation",
    ]

    wide = wide.select(_keep_cols)
    low_flow_wide = low_flow_wide.select(_keep_cols)

    # Helper function: merge demographics onto a wide DataFrame
    def _merge_demographics(wide_df):
        wide_pd_ = wide_df.to_pandas()
        merged = pd.merge(
            hosp_df_filtered[['patient_id', 'hospitalization_id', 'admission_dttm', 'discharge_dttm','age_at_admission','discharge_category']],
            wide_pd_,
            on="hospitalization_id",
            how="inner",
        )
        return merged

    # Load Patient table for demographics + death_dttm (need all patient_ids from both groups)
    all_wide_ids = set(wide.to_pandas()["hospitalization_id"].tolist()) | set(low_flow_wide.to_pandas()["hospitalization_id"].tolist())
    all_patient_ids = hosp_df_filtered[hosp_df_filtered["hospitalization_id"].isin(all_wide_ids)]["patient_id"].unique().tolist()

    patient_table = Patient.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={"patient_id": all_patient_ids},
    )
    patient_demo = patient_table.df[["patient_id", "sex_category", "race_category", "ethnicity_category", "death_dttm"]].copy()
    patient_demo["death_dttm"] = pd.to_datetime(patient_demo["death_dttm"], errors="coerce")
    if patient_demo["death_dttm"].dt.tz is not None:
        patient_demo["death_dttm"] = patient_demo["death_dttm"].dt.tz_localize(None)

    def _add_derived_columns(df):
        df = df.merge(patient_demo, on="patient_id", how="left")
        df["icu_mortality"] = (
            df["death_dttm"].notna()
            & (df["death_dttm"] >= df["icu_start"])
            & (df["death_dttm"] <= df["icu_end"])
        )
        df["hospital_los_hours"] = (
            (df["discharge_dttm"] - df["admission_dttm"]).dt.total_seconds() / 3600
        )
        df["hospital_mortality"] = df["discharge_category"] == "Expired"
        df = df.drop(columns=["death_dttm"])
        return df

    # Process HFNO cohort
    cohort = _add_derived_columns(_merge_demographics(wide))

    # Process low-flow cohort
    cohort_low_flow = _add_derived_columns(_merge_demographics(low_flow_wide))

    print("\n=== FINAL HFNO COHORT ===")
    print(f"Total hospitalizations: {len(cohort)}")
    print(f"Unique patients: {cohort['patient_id'].nunique()}")
    print(f"Columns: {cohort.columns.tolist()}")

    print(f"\n=== LOW-FLOW COHORT ===")
    print(f"Total hospitalizations: {len(cohort_low_flow)}")
    print(f"Unique patients: {cohort_low_flow['patient_id'].nunique()}")
    return (
        cohort,
        cohort_low_flow,
        n_after_extub_loc,
        n_after_full_code,
        n_after_hfnc_lpm,
        n_after_imv_dur,
        n_after_nippv_cpap,
        n_after_paco2,
        n_excluded_extub_not_icu,
        n_excluded_hfnc_lpm,
        n_excluded_imv_dur,
        n_excluded_nippv_cpap_extub,
        n_excluded_not_full_code,
        n_excluded_paco2,
        wide_before_hfnc,
        window_summary,
    )


@app.cell
def _(Path, pl, resp_df, wide_before_hfnc, window_summary):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    from scipy.stats import gaussian_kde

    _out = Path("output_to_share")
    _out.mkdir(parents=True, exist_ok=True)

    # --- Categorize patients entering Step 17 ---
    _step17_input = (
        wide_before_hfnc
        .select(["hospitalization_id", "device_after_extubation"])
        .join(window_summary, on="hospitalization_id", how="left")
        .with_columns(pl.col("hfnc_gte30_minutes").fill_null(0))
    )
    _step17_input = _step17_input.with_columns(
        pl.when(pl.col("hfnc_gte30_minutes") >= 60)
        .then(pl.lit("HFNC >= 1h (qualifying)"))
        .when(pl.col("hfnc_gte30_minutes") > 0)
        .then(pl.lit("HFNC < 1h (excluded)"))
        .when(pl.col("device_after_extubation") == "high flow nc")
        .then(pl.lit("HFNC, no qualifying LPM"))
        .otherwise(pl.lit("Non-HFNC device"))
        .alias("group")
    )

    # --- Table 1: Device breakdown by group ---
    _device_table = (
        _step17_input
        .group_by(["group", "device_after_extubation"])
        .len()
        .sort(["group", "len"], descending=[False, True])
    )
    _device_table.write_csv(_out / "step17_device_by_group.csv")

    # --- Table 2: Summary stats per group ---
    _group_stats = (
        _step17_input
        .group_by("group")
        .agg([
            pl.len().alias("n_patients"),
            pl.col("hfnc_gte30_minutes").mean().alias("mean_minutes"),
            pl.col("hfnc_gte30_minutes").median().alias("median_minutes"),
            pl.col("hfnc_gte30_minutes").quantile(0.25).alias("q25_minutes"),
            pl.col("hfnc_gte30_minutes").quantile(0.75).alias("q75_minutes"),
            pl.col("hfnc_gte30_minutes").max().alias("max_minutes"),
        ])
        .sort("n_patients", descending=True)
    )
    _group_stats.write_csv(_out / "step17_group_stats.csv")

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Panel 1: histogram of cumulative HFNC minutes
    _qualifying = _step17_input.filter(pl.col("hfnc_gte30_minutes") >= 60)["hfnc_gte30_minutes"].to_list()
    _not_qualifying = _step17_input.filter(pl.col("hfnc_gte30_minutes") < 60)["hfnc_gte30_minutes"].to_list()
    _bins = np.concatenate([[-0.5], np.arange(0.5, 125, 4)])
    axes[0].hist(
        [_not_qualifying, _qualifying],
        bins=_bins, stacked=True,
        label=[f"< 1h ({len(_not_qualifying):,})", f">= 1h ({len(_qualifying):,})"],
        color=["#e74c3c", "#2ecc71"], edgecolor="white", log=True,
    )
    axes[0].axvline(x=60, color="black", linestyle="--", linewidth=1, label="60 min threshold")
    axes[0].set_xlabel("Cumulative HFNC minutes (lpm >= 30)")
    axes[0].set_ylabel("Number of patients (log scale)")
    axes[0].set_title("Cumulative HFNC time (lpm >= 30)\n(all patients entering Step 17)")
    axes[0].legend()

    # Panel 2: bar chart of device_after_extubation
    _device_counts = (
        _step17_input.group_by("device_after_extubation")
        .agg([pl.len().alias("total"), (pl.col("hfnc_gte30_minutes") >= 60).sum().alias("qualifying")])
        .sort("total", descending=True)
    )
    _devices = _device_counts["device_after_extubation"].to_list()
    _totals = _device_counts["total"].to_list()
    _quals = _device_counts["qualifying"].to_list()
    _exc = [t - q for t, q in zip(_totals, _quals)]
    axes[1].bar(_devices, _exc, label="Excluded", color="#e74c3c")
    axes[1].bar(_devices, _quals, bottom=_exc, label="Qualifying", color="#2ecc71")
    axes[1].set_xlabel("Device after extubation")
    axes[1].set_ylabel("Number of patients")
    axes[1].set_title("Device after extubation\n(patients entering Step 17)")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=45)

    # Panel 3: KDE of lpm_set by device category
    _resp = pl.from_pandas(resp_df) if not isinstance(resp_df, pl.DataFrame) else resp_df
    _window_all = (
        _resp.join(
            wide_before_hfnc.select(["hospitalization_id", "extubation_time"]),
            on="hospitalization_id", how="inner",
        )
        .filter(
            (pl.col("recorded_dttm") > pl.col("extubation_time"))
            & (pl.col("recorded_dttm") <= pl.col("extubation_time") + pl.duration(hours=2))
        )
        .filter(pl.col("lpm_set").is_not_null() & (pl.col("lpm_set") > 0))
    )
    _colors = {"high flow nc": "#3498db", "nasal cannula": "#e67e22", "face mask": "#9b59b6",
               "low flow nc": "#1abc9c", "room air": "#95a5a6", "other": "#7f8c8d"}
    _top_devices = (
        _window_all.group_by("device_category").len().sort("len", descending=True)
        .head(5)["device_category"].to_list()
    )
    _x_grid = np.linspace(0, _window_all["lpm_set"].max() + 5, 300)
    for _dev in _top_devices:
        _vals = _window_all.filter(pl.col("device_category") == _dev)["lpm_set"].to_numpy()
        if len(_vals) >= 2:
            _kde = gaussian_kde(_vals, bw_method=0.3)
            axes[2].plot(_x_grid, _kde(_x_grid), label=f"{_dev} (n={len(_vals):,})",
                         color=_colors.get(_dev, None), linewidth=2)
    axes[2].axvline(x=30, color="black", linestyle="--", linewidth=1, label="lpm = 30 threshold")
    axes[2].set_xlabel("LPM set")
    axes[2].set_ylabel("Density")
    axes[2].set_title("LPM distribution by device\n(2h post-extubation window)")
    axes[2].legend()
    for ax in axes:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    fig.savefig(_out / "step17_diagnostic.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {_out / 'step17_diagnostic.png'}")

    # --- Sub-analysis: HFNC-first patients who failed the 1h threshold ---
    _hfnc_first_failed_ids = (
        window_summary.filter(pl.col("hfnc_gte30_minutes") < 60)
        ["hospitalization_id"].to_list()
    )
    _n_hfnc_first_failed = len(_hfnc_first_failed_ids)

    if _n_hfnc_first_failed > 0:
        _failed_window = (
            _resp.join(
                wide_before_hfnc
                .filter(pl.col("hospitalization_id").is_in(_hfnc_first_failed_ids))
                .select(["hospitalization_id", "extubation_time"]),
                on="hospitalization_id", how="inner",
            )
            .filter(
                (pl.col("recorded_dttm") > pl.col("extubation_time"))
                & (pl.col("recorded_dttm") <= pl.col("extubation_time") + pl.duration(hours=2))
            )
            .sort("hospitalization_id", "recorded_dttm")
        )
        _failed_summary = (
            _failed_window.group_by("hospitalization_id")
            .agg([
                pl.len().alias("n_readings"),
                (pl.col("device_category") != "high flow nc").any().alias("had_device_switch"),
                pl.col("device_category").filter(pl.col("device_category") != "high flow nc").len().alias("n_non_hfnc_readings"),
                pl.col("lpm_set").filter(
                    (pl.col("device_category") == "high flow nc")
                    & ((pl.col("lpm_set").is_null()) | (pl.col("lpm_set") < 30))
                ).len().alias("n_hfnc_low_lpm"),
                pl.col("lpm_set").filter(
                    (pl.col("device_category") == "high flow nc")
                    & (pl.col("lpm_set").is_not_null()) & (pl.col("lpm_set") >= 30)
                ).len().alias("n_hfnc_qualifying"),
                pl.col("device_category").filter(pl.col("device_category") == "high flow nc").len().alias("n_hfnc_readings"),
                pl.col("lpm_set").filter(pl.col("device_category") == "high flow nc").mean().alias("mean_lpm_hfnc"),
                pl.col("lpm_set").filter(pl.col("device_category") == "high flow nc").median().alias("median_lpm_hfnc"),
            ])
            .join(window_summary.select(["hospitalization_id", "hfnc_gte30_minutes"]), on="hospitalization_id", how="left")
        )
        _failed_summary = _failed_summary.with_columns(
            pl.when(pl.col("had_device_switch") & (pl.col("n_hfnc_qualifying") == 0))
            .then(pl.lit("Switched device, no qualifying HFNC"))
            .when(pl.col("had_device_switch") & (pl.col("n_hfnc_qualifying") > 0))
            .then(pl.lit("Switched device, some qualifying HFNC"))
            .when(pl.col("n_hfnc_low_lpm") > 0)
            .then(pl.lit("HFNC only, but lpm < 30 or null"))
            .when(pl.col("n_readings") <= 2)
            .then(pl.lit("Too few readings in window"))
            .otherwise(pl.lit("Other"))
            .alias("failure_reason")
        )
        _reason_table = (
            _failed_summary.group_by("failure_reason")
            .agg([
                pl.len().alias("n_patients"),
                pl.col("hfnc_gte30_minutes").mean().alias("mean_qualifying_min"),
                pl.col("hfnc_gte30_minutes").median().alias("median_qualifying_min"),
                pl.col("mean_lpm_hfnc").mean().alias("avg_mean_lpm"),
            ])
            .sort("n_patients", descending=True)
        )
        _reason_table.write_csv(_out / "step17_hfnc_failure_reasons.csv")

        _switched_ids = _failed_summary.filter(pl.col("had_device_switch"))["hospitalization_id"].to_list()
        if _switched_ids:
            _switch_devices = (
                _failed_window
                .filter(
                    pl.col("hospitalization_id").is_in(_switched_ids)
                    & (pl.col("device_category") != "high flow nc")
                )
                .group_by("device_category")
                .agg(pl.col("hospitalization_id").n_unique().alias("n_patients"))
                .sort("n_patients", descending=True)
            )
            _switch_devices.write_csv(_out / "step17_hfnc_switch_devices.csv")
            print(f"Saved: {_out / 'step17_hfnc_switch_devices.csv'}")

        print(f"\nHFNC-first patients who failed >= 1h threshold: {_n_hfnc_first_failed}")
        print(f"Saved: {_out / 'step17_hfnc_failure_reasons.csv'}")
        print("\nFailure reasons:")
        for _row in _reason_table.iter_rows(named=True):
            print(f"  {_row['failure_reason']}: {_row['n_patients']} patients")
    else:
        print("All HFNC-first patients met the >= 1h threshold.")

    print(f"\nSaved: {_out / 'step17_group_stats.csv'}")
    print(f"Saved: {_out / 'step17_device_by_group.csv'}")
    return


@app.cell
def _(
    SITE,
    cohort,
    mo,
    n_after_date,
    n_after_dx,
    n_after_extub_loc,
    n_after_full_code,
    n_after_hfnc_lpm,
    n_after_imv_before_icu,
    n_after_imv_dur,
    n_after_nippv_cpap,
    n_after_paco2,
    n_after_resp,
    n_excluded_age,
    n_excluded_date,
    n_excluded_dx,
    n_excluded_extub_not_icu,
    n_excluded_hfnc_lpm,
    n_excluded_imv_dur,
    n_excluded_nippv_cpap_extub,
    n_excluded_no_icu,
    n_excluded_no_imv,
    n_excluded_no_imv_before_icu,
    n_excluded_no_resp,
    n_excluded_not_full_code,
    n_excluded_null_out,
    n_excluded_paco2,
    n_excluded_trach,
    n_no_extubation,
    n_no_intubation,
    n_total_hosp,
    n_with_extubation,
    n_with_icu,
    n_with_imv_no_trach,
    n_with_intubation,
):
    # Compute intermediate count: hospitalizations with any IMV (before trach exclusion)
    n_with_any_imv = n_with_imv_no_trach + n_excluded_trach

    # Build CONSORT-style flow diagram data
    consort_flow = {
        "site": SITE,
        "steps": [
            {
                "step": 0,
                "description": "Total hospitalizations in database",
                "n_remaining": n_total_hosp,
                "n_excluded": 0,
                "exclusion_reason": None,
            },
            {
                "step": 1,
                "description": "Adults (age >= 18)",
                "n_remaining": n_total_hosp - n_excluded_age,
                "n_excluded": n_excluded_age,
                "exclusion_reason": "Age < 18",
            },
            {
                "step": 2,
                "description": "Admission in study period (2018-01-01 to 2024-12-01)",
                "n_remaining": n_after_date,
                "n_excluded": n_excluded_date,
                "exclusion_reason": "Outside study date range",
            },
            {
                "step": 3,
                "description": "No acute-chronic resp failure/OSA/HF/CPAP-BiPAP dx",
                "n_remaining": n_after_dx,
                "n_excluded": n_excluded_dx,
                "exclusion_reason": "Chronic resp failure, OSA, HF, or CPAP/BiPAP diagnosis",
            },
            {
                "step": 4,
                "description": "With ICU stay",
                "n_remaining": n_with_icu + n_excluded_null_out,
                "n_excluded": n_excluded_no_icu,
                "exclusion_reason": "No ICU stay",
            },
            {
                "step": 5,
                "description": "ICU stays with valid discharge time",
                "n_remaining": n_with_icu,
                "n_excluded": n_excluded_null_out,
                "exclusion_reason": "All ICU stays had null out_dttm",
            },
            {
                "step": 6,
                "description": "With invasive mechanical ventilation",
                "n_remaining": n_with_any_imv,
                "n_excluded": n_excluded_no_imv,
                "exclusion_reason": "No IMV",
            },
            {
                "step": 7,
                "description": "No tracheostomy during hospitalization",
                "n_remaining": n_with_imv_no_trach,
                "n_excluded": n_excluded_trach,
                "exclusion_reason": "Tracheostomy",
            },
            {
                "step": 8,
                "description": "Respiratory data in ICU window",
                "n_remaining": n_after_resp,
                "n_excluded": n_excluded_no_resp,
                "exclusion_reason": "No respiratory data before ICU end",
            },
            {
                "step": 9,
                "description": "IMV before ICU end",
                "n_remaining": n_after_imv_before_icu,
                "n_excluded": n_excluded_no_imv_before_icu,
                "exclusion_reason": "No IMV before ICU end",
            },
            {
                "step": 10,
                "description": "Confirmed intubation detected",
                "n_remaining": n_with_intubation,
                "n_excluded": n_no_intubation,
                "exclusion_reason": "No confirmed intubation",
            },
            {
                "step": 11,
                "description": "Confirmed extubation detected",
                "n_remaining": n_with_extubation,
                "n_excluded": n_no_extubation,
                "exclusion_reason": "No confirmed extubation",
            },
            {
                "step": 12,
                "description": "Extubation occurred in ICU",
                "n_remaining": n_after_extub_loc,
                "n_excluded": n_excluded_extub_not_icu,
                "exclusion_reason": "Extubation outside ICU",
            },
            {
                "step": 13,
                "description": "No DNR/DNI/DNAR/AND code status at extubation",
                "n_remaining": n_after_full_code,
                "n_excluded": n_excluded_not_full_code,
                "exclusion_reason": "DNR/DNI/DNAR/UDNR/AND code status at extubation",
            },
            {
                "step": 14,
                "description": "PaCO2 <= 45 mmHg pre-extubation",
                "n_remaining": n_after_paco2,
                "n_excluded": n_excluded_paco2,
                "exclusion_reason": "PaCO2 > 45 mmHg",
            },
            {
                "step": 15,
                "description": "IMV duration >= 12 hours",
                "n_remaining": n_after_imv_dur,
                "n_excluded": n_excluded_imv_dur,
                "exclusion_reason": "IMV duration < 12 hours",
            },
            {
                "step": 16,
                "description": "Not extubated to NIPPV/CPAP",
                "n_remaining": n_after_nippv_cpap,
                "n_excluded": n_excluded_nippv_cpap_extub,
                "exclusion_reason": "Extubated to NIPPV or CPAP",
            },
            {
                "step": 17,
                "description": ">= 1h cumulative HFNC (lpm >= 30) in 2h post-extubation",
                "n_remaining": n_after_hfnc_lpm,
                "n_excluded": n_excluded_hfnc_lpm,
                "exclusion_reason": "< 1h cumulative HFNC with lpm >= 30 in 2h post-extubation",
            },
        ],
        "final_cohort": {
            "n_hospitalizations": len(cohort),
            "n_patients": cohort["patient_id"].nunique(),
        },
    }

    # Display CONSORT table
    mo.md(
        f"""
        ## Cohort Flow Summary (CONSORT Style)

        | Step | Criterion | N Remaining | N Excluded | Exclusion Reason |
        |------|-----------|-------------|------------|------------------|
        | 0 | Total hospitalizations | {n_total_hosp:,} | - | - |
        | 1 | Adults (age >= 18) | {n_total_hosp - n_excluded_age:,} | {n_excluded_age:,} | Age < 18 |
        | 2 | Study period (2018-2024) | {n_after_date:,} | {n_excluded_date:,} | Outside date range |
        | 3 | No excluded diagnoses | {n_after_dx:,} | {n_excluded_dx:,} | Chronic resp failure, OSA, HF, or CPAP/BiPAP dx |
        | 4 | With ICU stay | {n_with_icu + n_excluded_null_out:,} | {n_excluded_no_icu:,} | No ICU stay |
        | 5 | Valid ICU discharge time | {n_with_icu:,} | {n_excluded_null_out:,} | All ICU stays had null out_dttm |
        | 6 | With IMV | {n_with_any_imv:,} | {n_excluded_no_imv:,} | No IMV |
        | 7 | No tracheostomy | {n_with_imv_no_trach:,} | {n_excluded_trach:,} | Tracheostomy |
        | 8 | Resp data in ICU window | {n_after_resp:,} | {n_excluded_no_resp:,} | No resp data before ICU end |
        | 9 | IMV before ICU end | {n_after_imv_before_icu:,} | {n_excluded_no_imv_before_icu:,} | No IMV before ICU end |
        | 10 | Confirmed intubation | {n_with_intubation:,} | {n_no_intubation:,} | No confirmed intubation |
        | 11 | Confirmed extubation | {n_with_extubation:,} | {n_no_extubation:,} | No confirmed extubation |
        | 12 | Extubation in ICU | {n_after_extub_loc:,} | {n_excluded_extub_not_icu:,} | Extubation outside ICU |
        | 13 | No DNR/DNI/DNAR/AND at extubation | {n_after_full_code:,} | {n_excluded_not_full_code:,} | DNR/DNI/DNAR/UDNR/AND code status at extubation |
        | 14 | PaCO2 <= 45 mmHg | {n_after_paco2:,} | {n_excluded_paco2:,} | PaCO2 > 45 mmHg |
        | 15 | IMV >= 12 hours | {n_after_imv_dur:,} | {n_excluded_imv_dur:,} | IMV < 12 hours |
        | 16 | Not extubated to NIPPV/CPAP | {n_after_nippv_cpap:,} | {n_excluded_nippv_cpap_extub:,} | Extubated to NIPPV/CPAP |
        | 17 | >= 1h HFNC lpm>=30 (2h window) | {n_after_hfnc_lpm:,} | {n_excluded_hfnc_lpm:,} | < 1h HFNC with lpm >= 30 in 2h window |

        **Final cohort: {len(cohort):,} hospitalizations, {cohort['patient_id'].nunique():,} unique patients**
        """
    )
    return (consort_flow,)


@app.cell
def _(
    Path,
    cohort,
    cohort_low_flow,
    consort_flow,
    dx_exclusion_breakdown,
    json,
):
    # Create output directories
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_to_share_dir = Path(__file__).parent.parent / "output_to_share"
    output_to_share_dir.mkdir(parents=True, exist_ok=True)

    # Save HFNO cohort parquet to output/
    cohort_path = output_dir / "cohort_inclusion.parquet"
    cohort.to_parquet(cohort_path, index=False)
    print(f"HFNO cohort saved to: {cohort_path}")

    # Save low-flow cohort parquet to output/
    low_flow_path = output_dir / "cohort_low_flow.parquet"
    cohort_low_flow.to_parquet(low_flow_path, index=False)
    print(f"Low-flow cohort saved to: {low_flow_path}")

    # Save CONSORT flow JSON to output_to_share/
    consort_path = output_to_share_dir / "consort_inclusion.json"
    with open(consort_path, "w") as consort_file:
        json.dump(consort_flow, consort_file, indent=2)
    print(f"CONSORT flow saved to: {consort_path}")

    # Save ICD-10 exclusion breakdown CSV
    dx_breakdown_path = output_to_share_dir / "dx_exclusion_breakdown.csv"
    dx_exclusion_breakdown.write_csv(dx_breakdown_path)
    print(f"ICD-10 exclusion breakdown saved to: {dx_breakdown_path}")

    print(f"\nHFNO cohort columns: {cohort.columns.tolist()}")
    print(f"Low-flow cohort columns: {cohort_low_flow.columns.tolist()}")
    return


@app.cell
def _(Path, intub_extub, pl):
    # Sub-analysis 1: Device distribution after extubation
    _device_counts = (
        intub_extub
        .group_by("device_after_extubation")
        .len()
        .rename({"len": "n"})
        .sort("n", descending=True)
    )
    _total = _device_counts["n"].sum()
    _device_counts = _device_counts.with_columns(
        (pl.col("n") / _total * 100).round(1).alias("percent")
    )

    _device_counts_pd = _device_counts.to_pandas()
    _output_dir = Path(__file__).parent.parent / "output_to_share"
    _device_counts_pd.to_csv(
        _output_dir / "subanalysis_device_after_extubation.csv", index=False
    )
    print("Sub-analysis 1: Device after extubation")
    print(_device_counts_pd.to_string(index=False))
    return


@app.cell
def _(
    DATA_DIR,
    FILETYPE,
    Path,
    Patient,
    TIMEZONE,
    adt_df,
    first_icu,
    first_intubation,
    hosp_df_filtered,
    intub_extub,
    pd,
    pl,
):
    # Sub-analysis 2: Why extubation was not found
    _no_extub_ids = (
        first_intubation
        .filter(~pl.col("hospitalization_id").is_in(intub_extub["hospitalization_id"]))
        ["hospitalization_id"]
        .to_list()
    )
    print(f"Hospitalizations with intubation but no extubation: {len(_no_extub_ids)}")

    # Get patient_ids for these hospitalizations
    _no_extub_hosp = hosp_df_filtered[
        hosp_df_filtered["hospitalization_id"].isin(_no_extub_ids)
    ][["hospitalization_id", "patient_id", "discharge_dttm", "discharge_category"]].copy()
    _patient_ids = _no_extub_hosp["patient_id"].unique().tolist()

    # Load Patient table for death_dttm
    _patient_table = Patient.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={"patient_id": _patient_ids},
    )
    _patient_pd = _patient_table.df[["patient_id", "death_dttm"]].copy()
    _patient_pd["death_dttm"] = pd.to_datetime(_patient_pd["death_dttm"], errors="coerce")
    _patient_pd["death_dttm"] = _patient_pd["death_dttm"].dt.tz_localize(None)

    # Merge death info
    _no_extub_hosp = _no_extub_hosp.merge(_patient_pd, on="patient_id", how="left")

    # Get ICU end times
    _icu_info = first_icu.select(["hospitalization_id", "icu_end"]).to_pandas()
    _no_extub_hosp = _no_extub_hosp.merge(_icu_info, on="hospitalization_id", how="left")

    # Get next location after ICU (from adt_df)
    _no_extub_adt = (
        adt_df
        .filter(pl.col("hospitalization_id").is_in(_no_extub_ids))
        .join(
            first_icu.select(["hospitalization_id", "icu_end"]),
            on="hospitalization_id",
            how="inner",
        )
        .filter(pl.col("in_dttm") >= pl.col("icu_end"))
        .filter(pl.col("location_category") != "icu")
        .sort(["hospitalization_id", "in_dttm"])
        .group_by("hospitalization_id")
        .first()
        .select(["hospitalization_id", pl.col("location_category").alias("next_location_after_icu")])
        .to_pandas()
    )
    _no_extub_hosp = _no_extub_hosp.merge(_no_extub_adt, on="hospitalization_id", how="left")

    # Classify reason
    def _classify_reason(row):
        # Check death in ICU (death_dttm <= icu_end)
        if pd.notna(row.get("death_dttm")) and pd.notna(row.get("icu_end")):
            if row["death_dttm"] <= row["icu_end"]:
                return "Died in ICU"
        if row.get("discharge_category") == "Expired" and pd.isna(row.get("next_location_after_icu")):
            return "Died in ICU"
        # Check transferred from ICU to another location
        if pd.notna(row.get("next_location_after_icu")):
            return f"Transferred from ICU (to {row['next_location_after_icu']})"
        # Check discharged from hospital
        if pd.notna(row.get("discharge_dttm")):
            return f"Discharged from hospital ({row.get('discharge_category', 'unknown')})"
        return "Unknown/Other"

    _no_extub_hosp["reason"] = _no_extub_hosp.apply(_classify_reason, axis=1)

    # Aggregate
    _reason_counts = (
        _no_extub_hosp.groupby("reason")
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )
    _reason_counts["percent"] = (_reason_counts["n"] / _reason_counts["n"].sum() * 100).round(1)

    _output_dir = Path(__file__).parent.parent / "output_to_share"
    _reason_counts.to_csv(
        _output_dir / "subanalysis_no_extubation_reasons.csv", index=False
    )
    print("\nSub-analysis 2: Reasons extubation not found")
    print(_reason_counts.to_string(index=False))
    return


@app.cell
def _(cohort):
    cohort
    return


if __name__ == "__main__":
    app.run()
