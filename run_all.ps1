# Stop on first error for Python steps
$ErrorActionPreference = "Stop"

# 1. Install dependencies
uv sync

# 2. Run Python pipeline
uv run python code/01_cohort.py
uv run python code/02_hfno_trajectory.py
uv run python code/03_extubation_success.py
uv run python code/04_table1.py

# 3. R/Quarto notice
Write-Host ""
Write-Host "Python steps complete. To finish the analysis, open and render these files in RStudio:" -ForegroundColor Cyan
Write-Host "  - code/05_hfno_site_analysis.qmd" -ForegroundColor Cyan
Write-Host "  - code/06_rox_prediction_site_analysis.qmd" -ForegroundColor Cyan
