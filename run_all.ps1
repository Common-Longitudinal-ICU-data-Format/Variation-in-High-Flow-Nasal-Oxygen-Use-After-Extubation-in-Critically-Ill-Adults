# Stop on first error for Python steps
$ErrorActionPreference = "Stop"

# 1. Install dependencies
uv sync

# 2. Run Python pipeline
uv run python code/01_cohort.py
uv run python code/02_hfno_trajectory.py
uv run python code/03_extubation_success.py
uv run python code/04_table1.py

# 3. Run R/Quarto analyses (with fallback message)
try {
    quarto render code/05_hfno_site_analysis.qmd
    quarto render code/06_rox_prediction_site_analysis.qmd
} catch {
    Write-Host ""
    Write-Host "Quarto render failed. Please use RStudio to render/run the .qmd files:" -ForegroundColor Yellow
    Write-Host "  - code/05_hfno_site_analysis.qmd" -ForegroundColor Yellow
    Write-Host "  - code/06_rox_prediction_site_analysis.qmd" -ForegroundColor Yellow
    exit 1
}
