# Variation in High-Flow Nasal Oxygen Use After Extubation in Critically Ill Adults

## Introduction

High-flow nasal oxygen (HFNO) can provide respiratory support between the level delivered by traditional low-flow nasal cannula and invasive mechanical ventilation. Although the immediate use of HFNO after liberation from invasive mechanical ventilation likely decreases the risk of extubation failure, there is high variation in HFNO use between medical centers across the US, and the optimal settings for HFNO (fraction of inspired oxygen and flow rate) that best mitigate the risk of extubation failure remain unknown.

**Aim:** To evaluate the between-hospital variation in the use of protocolized post-extubation HFNO across the United States and its association with extubation failure.

This is a federated multi-site study using the [Common Longitudinal ICU Format (CLIF)](https://clif-consortium.github.io/website/). Each participating site runs the pipeline locally on their CLIF-formatted data and returns only aggregate outputs (in `output_to_share/`) to the coordinating center.

## CLIF Tables Used

| CLIF Table | Category Column | Values / Usage |
|---|---|---|
| `hospitalization` | — | Age >= 18, admission date 2018–2024 |
| `respiratory_support` | `device_category` | `IMV`, `tracheostomy`, `high flow nc`, `nippv`, `cpap`, low-flow devices |
| `adt` | `location_category` | `icu` (ICU stays, merging consecutive stays) |
| `code_status` | `code_status_category` | Exclude DNR, DNI, DNAR, AND |
| `hospital_diagnosis` | — | ICD-10 exclusions: J9622, G4733, G4730, I501, I5021, Z9989, Z9911 |
| `labs` | `lab_category` | `pco2_arterial` (PaCO2 before extubation) |
| `vitals` | `vital_category` | `spo2`, `respiratory_rate`, `height_cm`, `weight_kg` |
| `patient` | — | Death dates, demographics (sex, race, ethnicity) |
| `medication_admin_continuous` | `med_group` | `vasoactives` (vasopressor use) |
| `crrt_therapy` | — | CRRT records (life support flags) |

## Pipeline Overview

| Step | File | Description |
|---|---|---|
| 1 | `code/01_cohort.py` | Applies inclusion/exclusion criteria, detects intubation/extubation events, computes SOFA scores, and outputs the HFNO and low-flow cohorts |
| 2 | `code/02_hfno_trajectory.py` | Builds a longitudinal panel dataset (one row per hospitalization per 4-hour window, up to 7 days post-extubation) with HFNO settings, vitals, ROX index, and outcomes |
| 3 | `code/03_extubation_success.py` | Adds 7-day extubation success flags, HFNO weaning outcomes, and life-support-prior-to-extubation indicators to both cohorts |
| 4 | `code/04_table1.py` | Generates Table 1 (baseline characteristics) comparing HFNO vs low-flow groups |
| 5 | `code/05_hfno_site_analysis.qmd` | Runs the federated marginal structural model analysis with IPTW, sensitivity analyses, and generates all figures/tables for the coordinating site |

## Run Instructions

### 1. Configure your site

Copy the config template and fill in your site details:

```bash
cp clif_config_template.json clif_config.json
```

Edit `clif_config.json`:

```json
{
  "site": "your_site_name",
  "data_directory": "./data",
  "filetype": "parquet",
  "timezone": "US/Central"
}
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Run the Python pipeline (in order)

```bash
uv run python code/01_cohort.py
uv run python code/02_hfno_trajectory.py
uv run python code/03_extubation_success.py
uv run python code/04_table1.py
```

### 4. Run the R analysis

Open `code/05_hfno_site_analysis.qmd` in RStudio and render it, or from the command line:

```bash
quarto render code/05_hfno_site_analysis.qmd
```

### 5. Return results

Share the `output_to_share/` directory with the coordinating site.
