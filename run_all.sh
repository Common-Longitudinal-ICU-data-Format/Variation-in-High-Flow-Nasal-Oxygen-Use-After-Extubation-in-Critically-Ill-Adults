#!/usr/bin/env bash
set -euo pipefail

# ── Logging setup ───────────────────────────────────────────────────
LOGFILE="output/run_all_$(date '+%Y-%m-%d_%H%M%S').log"
exec > >(tee "$LOGFILE") 2>&1

SCRIPT_START=$SECONDS

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_step() {
    local name="$1"; shift
    log "START  $name"
    local step_start=$SECONDS
    "$@"
    local elapsed=$(( SECONDS - step_start ))
    log "DONE   $name  (${elapsed}s)"
}

# ── Pipeline ────────────────────────────────────────────────────────
log "Pipeline started"

run_step "uv sync"                        uv sync
run_step "01_cohort.py"                   uv run python code/01_cohort.py
run_step "02_hfno_trajectory.py"          uv run python code/02_hfno_trajectory.py
run_step "03_extubation_success.py"       uv run python code/03_extubation_success.py
run_step "04_table1.py"                   uv run python code/04_table1.py
run_step "05_hfno_site_analysis.qmd"      quarto render code/05_hfno_site_analysis.qmd
run_step "06_rox_prediction_site_analysis.qmd" quarto render code/06_rox_prediction_site_analysis.qmd

# ── Summary ─────────────────────────────────────────────────────────
TOTAL=$(( SECONDS - SCRIPT_START ))
log "Pipeline finished  (total ${TOTAL}s)"
log "Log saved to $LOGFILE"
