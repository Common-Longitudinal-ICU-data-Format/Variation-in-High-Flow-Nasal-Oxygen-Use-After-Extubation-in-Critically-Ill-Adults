#!/usr/bin/env bash
set -euo pipefail

uv sync

uv run python code/01_cohort.py
uv run python code/02_hfno_trajectory.py
uv run python code/03_extubation_success.py
uv run python code/04_table1.py

# 5. Run R/Quarto analyses (with fallback message)
quarto render code/05_hfno_site_analysis.qmd || { echo "Quarto render failed. Please use RStudio to render/run the .qmd files."; exit 1; }
quarto render code/06_rox_prediction_site_analysis.qmd || { echo "Quarto render failed. Please use RStudio to render/run the .qmd files."; exit 1; }
