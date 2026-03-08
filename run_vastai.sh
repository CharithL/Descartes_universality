#!/bin/bash
# =============================================================
# DESCARTES Human Universality -- Vast.ai Full Pipeline
# =============================================================
# Run this on a fresh Vast.ai instance with a GPU.
# Recommended: RTX 3090/4090, >=24GB RAM, >=50GB disk
#
# Usage:
#   chmod +x run_vastai.sh && ./run_vastai.sh
#
# Or step by step:
#   ./run_vastai.sh setup     # install deps only
#   ./run_vastai.sh download  # download NWB data only
#   ./run_vastai.sh run       # run pipeline only
# =============================================================

set -e  # exit on error

REPO_URL="https://github.com/CharithL/Descartes_universality.git"
WORK_DIR="/workspace/descartes"
HIDDEN=128  # hidden size for surrogates

# ---- Colors for output ----
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${GREEN}[DESCARTES]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
err() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================
# Step 1: Setup
# =============================================================
do_setup() {
    log "=== STEP 1: Setup ==="

    # Clone repo if not already present
    if [ ! -d "$WORK_DIR" ]; then
        log "Cloning repository..."
        git clone "$REPO_URL" "$WORK_DIR"
    else
        log "Repository exists, pulling latest..."
        cd "$WORK_DIR" && git pull origin master
    fi

    cd "$WORK_DIR"

    # Install dependencies
    log "Installing Python dependencies..."
    pip install -q -r requirements.txt

    # Verify GPU
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"

    log "Setup complete."
}

# =============================================================
# Step 2: Download NWB data
# =============================================================
do_download() {
    log "=== STEP 2: Download NWB data from DANDI 000469 ==="
    cd "$WORK_DIR"

    # Use API mode for reliable download (no dandi CLI needed)
    python scripts/11_human_download.py --mode api

    # Count downloaded files
    NWB_COUNT=$(find data/raw -name "*.nwb" 2>/dev/null | wc -l)
    log "Downloaded $NWB_COUNT NWB files."
}

# =============================================================
# Step 3: Run pipeline
# =============================================================
do_run() {
    log "=== STEP 3: Run DESCARTES Pipeline ==="
    cd "$WORK_DIR"

    # 3a. Generate NWB schema (explore first file)
    log "--- 3a: Generating NWB schema ---"
    python scripts/10_human_explore_nwb.py

    # 3b. Build patient inventory
    log "--- 3b: Building patient inventory ---"
    python scripts/12_human_inventory.py

    # 3c. Single patient proof-of-concept
    log "--- 3c: Single patient proof-of-concept ---"
    python scripts/13_human_single_patient.py --hidden $HIDDEN

    # 3d. Cross-seed consistency (10 seeds x best patient)
    log "--- 3d: Cross-seed test (10 seeds) ---"
    python scripts/14_human_cross_seed.py --hidden $HIDDEN --n-seeds 10

    # 3e. Cross-patient universality (1 seed x all patients)
    log "--- 3e: Cross-patient test (all patients) ---"
    python scripts/15_human_cross_patient.py --hidden $HIDDEN

    # 3f. Cross-architecture test (4 archs x best patient)
    log "--- 3f: Cross-architecture test ---"
    python scripts/16_human_cross_architecture.py --hidden $HIDDEN

    # 3g. Generate final universality report
    log "--- 3g: Generating universality report ---"
    python scripts/17_human_universality_report.py

    log "=== PIPELINE COMPLETE ==="
    log "Results saved to: $WORK_DIR/data/results/"
    log "Final report: $WORK_DIR/data/results/universality_report.txt"
    echo ""
    cat data/results/universality_report.txt
}

# =============================================================
# Main
# =============================================================
case "${1:-all}" in
    setup)    do_setup ;;
    download) do_download ;;
    run)      do_run ;;
    all)
        do_setup
        do_download
        do_run
        ;;
    *)
        echo "Usage: $0 {setup|download|run|all}"
        exit 1
        ;;
esac
