#!/usr/bin/env bash
# =============================================================================
# Train cells C, D, E in PARALLEL — one cell per Blackhole chip.
#
# Quietbox 2 has 4 Blackhole chips (device IDs 0-3).
# This runs Cell E on chip 1, C on chip 2, D on chip 3.
# Chip 0 is left free for diagnostics/eval.
#
# Cells A and B have been removed. A (pure Mamba2) plateaued at ~35% and
# served only as a floor. B (Mamba2+attention, lr=6e-4) plateaued at 60%/0%
# Task 1 — identical to E (same architecture, lr=2e-4) at 62%/5% Task 1,
# confirming the Task 1 plateau is architectural, not LR-related.
#
# Usage:
#   ./run_all_cells_parallel.sh              # fresh start, all 3 cells
#   ./run_all_cells_parallel.sh --resume     # resume from checkpoints
#   ./run_all_cells_parallel.sh --steps 500  # override step count
#
# Logs: logs/cell_<X>_parallel_<timestamp>.log
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-/home/rfenwick/Documents/jasper/.tt-venv/bin/python}"
LOG_DIR="$SCRIPT_DIR/logs"
CKPT_DIR="$SCRIPT_DIR/checkpoints"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

# Defaults
RESUME_FLAG=""
STEPS_FLAG=""

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume) RESUME_FLAG="yes"; shift ;;
        --steps)  STEPS_FLAG="--steps $2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--resume] [--steps N]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Cell → device ID mapping (device 0 left free for eval/diagnostics)
declare -A CELL_DEVICE
CELL_DEVICE[E]=1
CELL_DEVICE[C]=2
CELL_DEVICE[D]=3

# Cell → config
declare -A CELL_CONFIG
CELL_CONFIG[E]="configs/cell_e_tt.yaml"
CELL_CONFIG[C]="configs/cell_c_tt.yaml"
CELL_CONFIG[D]="configs/cell_d_tt.yaml"

echo "============================================"
echo "  Parallel Training — 3 cells on 3 chips"
echo "  Start: $(date)"
echo "============================================"

PIPELINE_START=$(date +%s)
PIDS=()

CELLS=(E C D)
for CELL in "${CELLS[@]}"; do
    CONFIG="${CELL_CONFIG[$CELL]}"
    DEV_ID="${CELL_DEVICE[$CELL]}"
    LOG_FILE="$LOG_DIR/cell_${CELL}_parallel_${TIMESTAMP}.log"
    FINAL_CKPT="$CKPT_DIR/cell_${CELL}_final.pt"

    # Build resume argument
    RESUME_ARG=""
    if [[ "$RESUME_FLAG" == "yes" && -f "$FINAL_CKPT" ]]; then
        echo "  Cell $CELL → chip $DEV_ID (resuming from $FINAL_CKPT)"
        RESUME_ARG="--resume $FINAL_CKPT"
    else
        echo "  Cell $CELL → chip $DEV_ID (fresh start)"
    fi

    # Launch in background
    $PYTHON -u train_ttnn.py \
        --config "$CONFIG" \
        --device "$DEV_ID" \
        --checkpoint_dir "$CKPT_DIR" \
        $STEPS_FLAG \
        $RESUME_ARG \
        > "$LOG_FILE" 2>&1 &

    PID=$!
    PIDS+=($PID)
    echo "    PID: $PID, Log: $LOG_FILE"
done

echo ""
echo "  All 3 cells launched. Waiting for completion..."
echo "  Monitor with: tail -f $LOG_DIR/cell_*_parallel_${TIMESTAMP}.log"
echo ""

# Wait for all to finish
FAILED=0
for i in "${!PIDS[@]}"; do
    CELL="${CELLS[$i]}"
    PID="${PIDS[$i]}"
    if wait "$PID"; then
        echo "  Cell $CELL (PID $PID) completed successfully"
    else
        echo "  Cell $CELL (PID $PID) FAILED with exit code $?"
        FAILED=1
    fi
done

PIPELINE_END=$(date +%s)
DURATION=$((PIPELINE_END - PIPELINE_START))
HOURS=$((DURATION / 3600))
MINS=$(((DURATION % 3600) / 60))

echo ""
echo "============================================"
echo "  Parallel training complete!"
echo "  Total wall time: ${HOURS}h ${MINS}m"
echo "  End: $(date)"
echo "  Logs: $LOG_DIR/cell_*_parallel_${TIMESTAMP}.log"
echo "  Checkpoints: $CKPT_DIR/"
if [[ $FAILED -eq 1 ]]; then
    echo "  WARNING: Some cells failed — check logs"
fi
echo "============================================"
