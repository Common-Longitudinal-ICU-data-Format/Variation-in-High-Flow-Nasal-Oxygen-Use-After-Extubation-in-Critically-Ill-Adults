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
    from clifpy.tables import Hospitalization, RespiratorySupport, Adt, CodeStatus, Labs, Patient
    return (
        Adt,
        CodeStatus,
        Hospitalization,
        Labs,
        Path,
        Patient,
        RespiratorySupport,
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
def _(CodeStatus, DATA_DIR, FILETYPE, TIMEZONE, hosp_df, pl):
    # Get patient IDs from hospitalizations
    patient_ids = hosp_df["patient_id"].unique().tolist()

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
    hosp_df_filtered = hosp_df.copy()
    return code_status_pl, hosp_df_filtered


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

    def merge_icu_procedural_icu(df: pl.DataFrame) -> pl.DataFrame:
        """Merge ICU stays separated only by procedural visits."""
        merged_rows = []
        hosp_ids_unique = df["hospitalization_id"].unique().to_list()

        for hosp_id in tqdm(hosp_ids_unique, desc="Merging ICU-procedural-ICU"):
            group = df.filter(pl.col("hospitalization_id") == hosp_id).sort("in_dttm")
            rows = group.to_dicts()
            i = 0
            while i < len(rows):
                row = rows[i].copy()

                if row["location_category"] == "icu":
                    j = i + 1
                    while j + 1 < len(rows):
                        next_row = rows[j]
                        next_next_row = rows[j + 1]

                        if (next_row["location_category"] == "procedural" and
                            next_next_row["location_category"] == "icu"):
                            row["out_dttm"] = next_next_row["out_dttm"]
                            j += 2
                        else:
                            break
                    i = j
                else:
                    i += 1

                merged_rows.append(row)

        return pl.DataFrame(merged_rows)

    adt_df = merge_icu_procedural_icu(adt_df)

    icu_adt_df = adt_df.filter(pl.col("location_category") == "icu")

    print(f"Total ADT records (after merging ICU-procedural-ICU): {len(adt_df)}")
    print(f"ICU ADT records: {len(icu_adt_df)}")
    return adt_df, hosp_with_icu, icu_adt_df, n_excluded_no_icu, n_with_icu


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
    first_icu = first_icu.rename({"in_dttm": "icu_start", "out_dttm": "icu_end"})

    # Compute ICU LOS in hours
    first_icu = first_icu.with_columns(
        ((pl.col("icu_end") - pl.col("icu_start")).dt.total_seconds() / 3600)
        .alias("icu_los_hours")
    )

    print(f"First ICU stays (from IMV no-trach set): {len(first_icu)}")
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
    code_status_pl,
    first_icu,
    hosp_df_filtered,
    intub_extub,
    location_info,
    paco2_pre,
    pd,
    pl,
    resp_df,
):
    # Step D: Build wide one-row-per-hospitalization dataset
    # Filter to rows with non-null extubation_time
    extub_valid = intub_extub.filter(pl.col("extubation_time").is_not_null())

    # Join with first_icu for ICU timing columns
    wide = extub_valid.join(
        first_icu.select([
            "hospitalization_id", "icu_start", "icu_end"
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
        (pl.col("paco2_pre_extubation").is_null()) | (pl.col("paco2_pre_extubation") <= 50)
    )
    n_excluded_paco2 = n_before_paco2 - len(wide)
    n_after_paco2 = len(wide)
    print(f"After PaCO2 <= 50 filter: {n_after_paco2} (excluded {n_excluded_paco2} with PaCO2 > 50)")

    # Exclude hospitalizations with IMV duration < 12 hours
    n_before_imv_dur = len(wide)
    wide = wide.filter(pl.col("imv_duration_hours") >= 12)
    n_excluded_imv_dur = n_before_imv_dur - len(wide)
    n_after_imv_dur = len(wide)
    print(f"After IMV duration >= 12h filter: {n_after_imv_dur} (excluded {n_excluded_imv_dur} with IMV < 12h)")

    # 1-hour post-extubation window: only HFNC device + at least one lpm_set >= 30
    window_resp = (
        resp_df
        .join(
            wide.select(["hospitalization_id", "extubation_time"]),
            on="hospitalization_id",
            how="inner",
        )
        .filter(
            (pl.col("recorded_dttm") > pl.col("extubation_time"))
            & (pl.col("recorded_dttm") <= pl.col("extubation_time") + pl.duration(hours=1))
        )
    )

    window_summary = (
        window_resp
        .group_by("hospitalization_id")
        .agg([
            pl.col("device_category").n_unique().alias("n_distinct_devices"),
            (pl.col("device_category") == "high flow nc").all().alias("all_hfnc"),
            ((pl.col("lpm_set").is_not_null()) & (pl.col("lpm_set") >= 30)).any().alias("has_lpm_gte_30"),
        ])
    )

    # Step 14: Exclusive HFNC in first hour post-extubation
    hfnc_only_ids = window_summary.filter(pl.col("all_hfnc"))["hospitalization_id"]
    n_before_hfnc = len(wide)
    wide = wide.filter(pl.col("hospitalization_id").is_in(hfnc_only_ids))
    n_excluded_not_hfnc = n_before_hfnc - len(wide)
    n_after_hfnc = len(wide)
    print(f"After exclusive HFNC in 1h window: {n_after_hfnc} (excluded {n_excluded_not_hfnc} without exclusive HFNC)")

    # Step 15: LPM >= 30 in first hour post-extubation
    lpm_valid_ids = window_summary.filter(pl.col("all_hfnc") & pl.col("has_lpm_gte_30"))["hospitalization_id"]
    n_before_lpm = len(wide)
    wide = wide.filter(pl.col("hospitalization_id").is_in(lpm_valid_ids))
    n_excluded_no_lpm = n_before_lpm - len(wide)
    n_after_lpm = len(wide)
    print(f"After LPM >= 30 in 1h window: {n_after_lpm} (excluded {n_excluded_no_lpm} without lpm_set >= 30)")

    wide = wide.select([
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
    ])

    # Convert to pandas and merge with demographics from hosp_df_filtered
    wide_pd = wide.to_pandas()
    cohort = pd.merge(
        hosp_df_filtered[['patient_id', 'hospitalization_id', 'admission_dttm', 'discharge_dttm','age_at_admission','discharge_category']],
        wide_pd,
        on="hospitalization_id",
        how="inner",
    )

    print("\n=== FINAL COHORT ===")
    print(f"Total hospitalizations: {len(cohort)}")
    print(f"Unique patients: {cohort['patient_id'].nunique()}")
    print(f"Columns: {cohort.columns.tolist()}")
    return (
        cohort,
        n_after_extub_loc,
        n_after_full_code,
        n_after_hfnc,
        n_after_imv_dur,
        n_after_lpm,
        n_after_paco2,
        n_excluded_extub_not_icu,
        n_excluded_imv_dur,
        n_excluded_no_lpm,
        n_excluded_not_full_code,
        n_excluded_not_hfnc,
        n_excluded_paco2,
    )


@app.cell
def _(
    SITE,
    cohort,
    mo,
    n_after_date,
    n_after_extub_loc,
    n_after_full_code,
    n_after_hfnc,
    n_after_imv_before_icu,
    n_after_imv_dur,
    n_after_lpm,
    n_after_paco2,
    n_after_resp,
    n_excluded_age,
    n_excluded_date,
    n_excluded_extub_not_icu,
    n_excluded_imv_dur,
    n_excluded_no_icu,
    n_excluded_no_imv,
    n_excluded_no_imv_before_icu,
    n_excluded_no_lpm,
    n_excluded_no_resp,
    n_excluded_not_full_code,
    n_excluded_not_hfnc,
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
                "description": "With ICU stay",
                "n_remaining": n_with_icu,
                "n_excluded": n_excluded_no_icu,
                "exclusion_reason": "No ICU stay",
            },
            {
                "step": 4,
                "description": "With invasive mechanical ventilation",
                "n_remaining": n_with_any_imv,
                "n_excluded": n_excluded_no_imv,
                "exclusion_reason": "No IMV",
            },
            {
                "step": 5,
                "description": "No tracheostomy during hospitalization",
                "n_remaining": n_with_imv_no_trach,
                "n_excluded": n_excluded_trach,
                "exclusion_reason": "Tracheostomy",
            },
            {
                "step": 6,
                "description": "Respiratory data in ICU window",
                "n_remaining": n_after_resp,
                "n_excluded": n_excluded_no_resp,
                "exclusion_reason": "No respiratory data before ICU end",
            },
            {
                "step": 7,
                "description": "IMV before ICU end",
                "n_remaining": n_after_imv_before_icu,
                "n_excluded": n_excluded_no_imv_before_icu,
                "exclusion_reason": "No IMV before ICU end",
            },
            {
                "step": 8,
                "description": "Confirmed intubation detected",
                "n_remaining": n_with_intubation,
                "n_excluded": n_no_intubation,
                "exclusion_reason": "No confirmed intubation",
            },
            {
                "step": 9,
                "description": "Confirmed extubation detected",
                "n_remaining": n_with_extubation,
                "n_excluded": n_no_extubation,
                "exclusion_reason": "No confirmed extubation",
            },
            {
                "step": 10,
                "description": "Extubation occurred in ICU",
                "n_remaining": n_after_extub_loc,
                "n_excluded": n_excluded_extub_not_icu,
                "exclusion_reason": "Extubation outside ICU",
            },
            {
                "step": 11,
                "description": "No DNR/DNI/DNAR/AND code status at extubation",
                "n_remaining": n_after_full_code,
                "n_excluded": n_excluded_not_full_code,
                "exclusion_reason": "DNR/DNI/DNAR/UDNR/AND code status at extubation",
            },
            {
                "step": 12,
                "description": "PaCO2 <= 50 mmHg pre-extubation",
                "n_remaining": n_after_paco2,
                "n_excluded": n_excluded_paco2,
                "exclusion_reason": "PaCO2 > 50 mmHg",
            },
            {
                "step": 13,
                "description": "IMV duration >= 12 hours",
                "n_remaining": n_after_imv_dur,
                "n_excluded": n_excluded_imv_dur,
                "exclusion_reason": "IMV duration < 12 hours",
            },
            {
                "step": 14,
                "description": "Exclusive HFNC in first hour post-extubation",
                "n_remaining": n_after_hfnc,
                "n_excluded": n_excluded_not_hfnc,
                "exclusion_reason": "Non-HFNC device in first hour post-extubation",
            },
            {
                "step": 15,
                "description": "LPM >= 30 in first hour post-extubation",
                "n_remaining": n_after_lpm,
                "n_excluded": n_excluded_no_lpm,
                "exclusion_reason": "No lpm_set >= 30 in first hour post-extubation",
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
        | 3 | With ICU stay | {n_with_icu:,} | {n_excluded_no_icu:,} | No ICU stay |
        | 4 | With IMV | {n_with_any_imv:,} | {n_excluded_no_imv:,} | No IMV |
        | 5 | No tracheostomy | {n_with_imv_no_trach:,} | {n_excluded_trach:,} | Tracheostomy |
        | 6 | Resp data in ICU window | {n_after_resp:,} | {n_excluded_no_resp:,} | No resp data before ICU end |
        | 7 | IMV before ICU end | {n_after_imv_before_icu:,} | {n_excluded_no_imv_before_icu:,} | No IMV before ICU end |
        | 8 | Confirmed intubation | {n_with_intubation:,} | {n_no_intubation:,} | No confirmed intubation |
        | 9 | Confirmed extubation | {n_with_extubation:,} | {n_no_extubation:,} | No confirmed extubation |
        | 10 | Extubation in ICU | {n_after_extub_loc:,} | {n_excluded_extub_not_icu:,} | Extubation outside ICU |
        | 11 | No DNR/DNI/DNAR/AND at extubation | {n_after_full_code:,} | {n_excluded_not_full_code:,} | DNR/DNI/DNAR/UDNR/AND code status at extubation |
        | 12 | PaCO2 <= 50 mmHg | {n_after_paco2:,} | {n_excluded_paco2:,} | PaCO2 > 50 mmHg |
        | 13 | IMV >= 12 hours | {n_after_imv_dur:,} | {n_excluded_imv_dur:,} | IMV < 12 hours |
        | 14 | Exclusive HFNC (1h window) | {n_after_hfnc:,} | {n_excluded_not_hfnc:,} | Non-HFNC device in 1h window |
        | 15 | LPM >= 30 (1h window) | {n_after_lpm:,} | {n_excluded_no_lpm:,} | No lpm_set >= 30 in 1h window |

        **Final cohort: {len(cohort):,} hospitalizations, {cohort['patient_id'].nunique():,} unique patients**
        """
    )
    return (consort_flow,)


@app.cell
def _(Path, cohort, consort_flow, json):
    # Create output directories
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_to_share_dir = Path(__file__).parent.parent / "output_to_share"
    output_to_share_dir.mkdir(parents=True, exist_ok=True)

    # Save cohort parquet to output/
    cohort_path = output_dir / "cohort_inclusion.parquet"
    cohort.to_parquet(cohort_path, index=False)
    print(f"Cohort saved to: {cohort_path}")

    # Save CONSORT flow JSON to output_to_share/
    consort_path = output_to_share_dir / "consort_inclusion.json"
    with open(consort_path, "w") as consort_file:
        json.dump(consort_flow, consort_file, indent=2)
    print(f"CONSORT flow saved to: {consort_path}")

    print(f"\nCohort columns: {cohort.columns.tolist()}")
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
