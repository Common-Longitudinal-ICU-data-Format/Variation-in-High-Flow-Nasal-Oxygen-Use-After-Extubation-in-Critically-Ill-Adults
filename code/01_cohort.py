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
    from clifpy.tables import Hospitalization, RespiratorySupport, Adt, CodeStatus
    return (
        Adt,
        CodeStatus,
        Hospitalization,
        Path,
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
def _(CodeStatus, DATA_DIR, FILETYPE, TIMEZONE, hosp_df, pd, pl):
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

    # Build hospitalizations Polars frame with timezone-naive datetimes
    hosp_pl = pl.from_pandas(
        hosp_df[["hospitalization_id", "patient_id", "admission_dttm", "discharge_dttm"]].assign(
            admission_dttm=pd.to_datetime(hosp_df["admission_dttm"]).dt.tz_localize(None),
            discharge_dttm=pd.to_datetime(hosp_df["discharge_dttm"]).dt.tz_localize(None),
        )
    )

    # Join code status to hospitalizations on patient_id using join_asof
    # First sort both by patient_id + time for asof join isn't ideal here;
    # instead use a cross-style join on patient_id then filter by time window
    joined = code_status_pl.join(hosp_pl, on="patient_id", how="inner").filter(
        (pl.col("start_dttm") >= pl.col("admission_dttm"))
        & (pl.col("start_dttm") <= pl.col("discharge_dttm"))
    )

    # Categories to exclude (full code throughout)
    exclude_categories = ["full", "presume full", "presume_full"]

    # For each hospitalization: check if ALL records are full/presume full
    hosp_code_summary = joined.group_by("hospitalization_id").agg(
        pl.col("code_status_lower").is_in(exclude_categories).all().alias("all_full"),
        pl.len().alias("n_records"),
    )

    hosp_to_exclude = set(
        hosp_code_summary.filter(pl.col("all_full"))["hospitalization_id"].to_list()
    )

    # Filter hospitalizations
    hosp_ids_after_code = set(hosp_df["hospitalization_id"].unique()) - hosp_to_exclude

    n_excluded_full_code = len(hosp_to_exclude)
    n_after_code_status = len(hosp_ids_after_code)

    print(f"Hospitalizations with only Full/Presume Full code status (excluded): {n_excluded_full_code}")
    print(f"After code status filter: {n_after_code_status}")

    # Filter hosp_df to remaining hospitalizations
    hosp_df_filtered = hosp_df[hosp_df["hospitalization_id"].isin(hosp_ids_after_code)].copy()
    return hosp_df_filtered, n_after_code_status, n_excluded_full_code


@app.cell
def _(
    Adt,
    DATA_DIR,
    FILETYPE,
    TIMEZONE,
    hosp_df_filtered,
    n_after_code_status,
    pl,
    tqdm,
):
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
    n_excluded_no_icu = n_after_code_status - n_with_icu
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
    return hosp_with_icu, icu_adt_df, n_excluded_no_icu, n_with_icu


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
def _(icu_adt_df, n_with_imv_no_trach, pl, resp_df):
    # Helper: check if timestamp is in ICU (works with Polars)
    def is_in_icu(hosp_id, ts, adt: pl.DataFrame):
        icu_stays = adt.filter(pl.col("hospitalization_id") == hosp_id)
        for row in icu_stays.iter_rows(named=True):
            if row["in_dttm"] <= ts <= row["out_dttm"]:
                return True
        return False

    # Get IMV records only
    imv_df = resp_df.filter(pl.col("device_category") == "IMV")

    n_with_imv = n_with_imv_no_trach
    print(f"Hospitalizations with IMV (no tracheostomy): {n_with_imv}")

    # First IMV episode per hospitalization
    first_imv = imv_df.group_by("hospitalization_id").agg(
        pl.col("recorded_dttm").min().alias("first_imv_time")
    )

    # Last IMV time (extubation time)
    last_imv = imv_df.group_by("hospitalization_id").agg(
        pl.col("recorded_dttm").max().alias("extubation_time")
    )

    # Merge
    imv_episodes_all = first_imv.join(last_imv, on="hospitalization_id")

    # IMV duration >= 12 hours
    imv_episodes_all = imv_episodes_all.with_columns([
        ((pl.col("extubation_time") - pl.col("first_imv_time")).dt.total_seconds() / 3600)
        .alias("imv_duration_hours")
    ])

    imv_12h = imv_episodes_all.filter(pl.col("imv_duration_hours") >= 12)
    n_imv_12h = len(imv_12h)
    n_excluded_imv_short = n_with_imv - n_imv_12h
    print(f"IMV >= 12 hours: {n_imv_12h} (excluded {n_excluded_imv_short} with IMV < 12h)")

    # Extubation in ICU - need to check row by row
    extub_in_icu = []
    for row in imv_12h.iter_rows(named=True):
        extub_in_icu.append(is_in_icu(row["hospitalization_id"], row["extubation_time"], icu_adt_df))

    imv_12h = imv_12h.with_columns(pl.Series("extubation_in_icu", extub_in_icu))
    imv_episodes = imv_12h.filter(pl.col("extubation_in_icu"))
    n_extub_icu = len(imv_episodes)
    n_excluded_extub_not_icu = n_imv_12h - n_extub_icu
    print(f"Extubation in ICU: {n_extub_icu} (excluded {n_excluded_extub_not_icu} extubated outside ICU)")
    return (
        imv_episodes,
        n_excluded_extub_not_icu,
        n_excluded_imv_short,
        n_extub_icu,
        n_imv_12h,
    )


@app.cell
def _(imv_episodes, n_extub_icu, pl, resp_df):
    # Get HFNO records
    hfno_df = resp_df.filter(pl.col("device_category") == "High Flow NC")

    # Merge with extubation times
    hfno_df = hfno_df.join(
        imv_episodes.select(["hospitalization_id", "extubation_time"]),
        on="hospitalization_id",
        how="inner",
    )

    # HFNO within 1 hour of extubation with flow >= 30 L/min
    hfno_df = hfno_df.with_columns([
        ((pl.col("recorded_dttm") - pl.col("extubation_time")).dt.total_seconds() / 3600)
        .alias("time_after_extubation")
    ])

    hfno_post_extub = hfno_df.filter(
        (pl.col("time_after_extubation") >= 0) &
        (pl.col("time_after_extubation") <= 1) &
        (pl.col("flow_rate_set") >= 30)
    )

    hfno_eligible = hfno_post_extub["hospitalization_id"].unique().to_list()
    n_hfno_1h = len(hfno_eligible)
    n_excluded_no_hfno_1h = n_extub_icu - n_hfno_1h
    print(f"HFNO within 1h of extubation (flow>=30): {n_hfno_1h} (excluded {n_excluded_no_hfno_1h})")
    return hfno_df, hfno_eligible, n_excluded_no_hfno_1h, n_hfno_1h


@app.cell
def _(hfno_df, hfno_eligible, n_hfno_1h, pl):
    # HFNO duration > 4 hours post-extubation
    hfno_post = hfno_df.filter(
        pl.col("hospitalization_id").is_in(hfno_eligible) &
        (pl.col("time_after_extubation") >= 0) &
        (pl.col("flow_rate_set") >= 30)
    )

    # Calculate HFNO duration per hospitalization
    hfno_duration = hfno_post.group_by("hospitalization_id").agg([
        pl.col("recorded_dttm").min().alias("hfno_start"),
        pl.col("recorded_dttm").max().alias("hfno_end"),
    ])

    hfno_duration = hfno_duration.with_columns([
        ((pl.col("hfno_end") - pl.col("hfno_start")).dt.total_seconds() / 3600)
        .alias("hfno_duration_hours")
    ])

    # Filter > 4 hours
    hfno_duration = hfno_duration.filter(pl.col("hfno_duration_hours") > 4)

    final_hosp_ids = hfno_duration["hospitalization_id"].unique().to_list()
    n_hfno_4h = len(final_hosp_ids)
    n_excluded_hfno_short = n_hfno_1h - n_hfno_4h
    print(f"HFNO duration > 4h: {n_hfno_4h} (excluded {n_excluded_hfno_short} with HFNO <= 4h)")
    return final_hosp_ids, hfno_duration, n_excluded_hfno_short, n_hfno_4h


@app.cell
def _(final_hosp_ids, hosp_df_filtered, imv_episodes, pd):
    # Build final cohort - hosp_df_filtered is still pandas
    cohort = hosp_df_filtered[hosp_df_filtered["hospitalization_id"].isin(final_hosp_ids)].copy()

    # Convert imv_episodes to pandas for merge
    imv_episodes_pd = imv_episodes.select([
        "hospitalization_id", "first_imv_time", "extubation_time", "imv_duration_hours"
    ]).to_pandas()

    cohort = pd.merge(
        cohort,
        imv_episodes_pd,
        on="hospitalization_id",
        how="left",
    )

    print("\n=== FINAL COHORT (Inclusion Criteria Only) ===")
    print(f"Total hospitalizations: {len(cohort)}")
    print(f"Unique patients: {cohort['patient_id'].nunique()}")
    return (cohort,)


@app.cell
def _(
    SITE,
    cohort,
    mo,
    n_after_code_status,
    n_after_date,
    n_excluded_age,
    n_excluded_date,
    n_excluded_extub_not_icu,
    n_excluded_full_code,
    n_excluded_hfno_short,
    n_excluded_imv_short,
    n_excluded_no_hfno_1h,
    n_excluded_no_icu,
    n_excluded_no_imv,
    n_excluded_trach,
    n_extub_icu,
    n_hfno_1h,
    n_hfno_4h,
    n_imv_12h,
    n_total_hosp,
    n_with_icu,
    n_with_imv_no_trach,
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
                "description": "Non-Full code status during hospitalization",
                "n_remaining": n_after_code_status,
                "n_excluded": n_excluded_full_code,
                "exclusion_reason": "Full/Presume Full code status throughout",
            },
            {
                "step": 4,
                "description": "With ICU stay",
                "n_remaining": n_with_icu,
                "n_excluded": n_excluded_no_icu,
                "exclusion_reason": "No ICU stay",
            },
            {
                "step": 5,
                "description": "With invasive mechanical ventilation",
                "n_remaining": n_with_any_imv,
                "n_excluded": n_excluded_no_imv,
                "exclusion_reason": "No IMV",
            },
            {
                "step": 6,
                "description": "No tracheostomy during hospitalization",
                "n_remaining": n_with_imv_no_trach,
                "n_excluded": n_excluded_trach,
                "exclusion_reason": "Tracheostomy",
            },
            {
                "step": 7,
                "description": "IMV duration >= 12 hours",
                "n_remaining": n_imv_12h,
                "n_excluded": n_excluded_imv_short,
                "exclusion_reason": "IMV < 12 hours",
            },
            {
                "step": 8,
                "description": "Extubation in ICU",
                "n_remaining": n_extub_icu,
                "n_excluded": n_excluded_extub_not_icu,
                "exclusion_reason": "Extubated outside ICU",
            },
            {
                "step": 9,
                "description": "HFNO within 1h of extubation (flow >= 30 L/min)",
                "n_remaining": n_hfno_1h,
                "n_excluded": n_excluded_no_hfno_1h,
                "exclusion_reason": "No HFNO within 1h or flow < 30 L/min",
            },
            {
                "step": 10,
                "description": "HFNO duration > 4 hours",
                "n_remaining": n_hfno_4h,
                "n_excluded": n_excluded_hfno_short,
                "exclusion_reason": "HFNO duration <= 4 hours",
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
        | 3 | Non-Full code status | {n_after_code_status:,} | {n_excluded_full_code:,} | Full/Presume Full code throughout |
        | 4 | With ICU stay | {n_with_icu:,} | {n_excluded_no_icu:,} | No ICU stay |
        | 5 | With IMV | {n_with_any_imv:,} | {n_excluded_no_imv:,} | No IMV |
        | 6 | No tracheostomy | {n_with_imv_no_trach:,} | {n_excluded_trach:,} | Tracheostomy |
        | 7 | IMV >= 12 hours | {n_imv_12h:,} | {n_excluded_imv_short:,} | IMV < 12h |
        | 8 | Extubation in ICU | {n_extub_icu:,} | {n_excluded_extub_not_icu:,} | Extubated outside ICU |
        | 9 | HFNO within 1h (flow >= 30) | {n_hfno_1h:,} | {n_excluded_no_hfno_1h:,} | No qualifying HFNO |
        | 10 | HFNO duration > 4h | {n_hfno_4h:,} | {n_excluded_hfno_short:,} | HFNO <= 4h |

        **Final cohort: {len(cohort):,} hospitalizations, {cohort['patient_id'].nunique():,} unique patients**
        """
    )
    return (consort_flow,)


@app.cell
def _(Path, cohort, consort_flow, hfno_duration, json, pd):
    # Create output directories
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_to_share_dir = Path(__file__).parent.parent / "output_to_share"
    output_to_share_dir.mkdir(parents=True, exist_ok=True)

    # Convert hfno_duration to pandas for merge
    hfno_duration_pd = hfno_duration.select([
        "hospitalization_id", "hfno_start", "hfno_end", "hfno_duration_hours"
    ]).to_pandas()

    # Merge HFNO timing into cohort
    cohort_final = pd.merge(
        cohort,
        hfno_duration_pd,
        on="hospitalization_id",
        how="left",
    )

    # Save cohort parquet to output/
    cohort_path = output_dir / "cohort_inclusion.parquet"
    cohort_final.to_parquet(cohort_path, index=False)
    print(f"Cohort saved to: {cohort_path}")

    # Save CONSORT flow JSON to output_to_share/
    consort_path = output_to_share_dir / "consort_inclusion.json"
    with open(consort_path, "w") as consort_file:
        json.dump(consort_flow, consort_file, indent=2)
    print(f"CONSORT flow saved to: {consort_path}")

    print(f"\nCohort columns: {cohort_final.columns.tolist()}")
    cohort_final
    return


if __name__ == "__main__":
    app.run()
