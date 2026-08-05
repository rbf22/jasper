#!/usr/bin/env bash
# =============================================================================
# Train cells A, B, C sequentially on Tenstorrent hardware.
#
# Usage:
#   ./run_all_cells.sh              # train all cells, fresh start
#   ./run_all_cells.sh --resume     # resume from last checkpoint if available
#   ./run_all_cells.sh --cell C     # train only cell C
#   ./run_all_cells.sh --steps 500  # override step count for all cells
#
# Logs are written to logs/cell_<X>_<timestamp>.log
# Checkpoints are saved to checkpoints/cell_<X>_final.pt
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-/home/ttuser/Documents/jasper/.tt-venv/bin/python}"
LOG_DIR="$SCRIPT_DIR/logs"
CKPT_DIR="$SCRIPT_DIR/checkpoints"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

# Defaults
RESUME_FLAG=""
CELLS="A B C"
STEPS_FLAG=""

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME_FLAG="yes"
            shift
            ;;
        --cell)
            CELLS="$2"
            shift 2
            ;;
        --steps)
            STEPS_FLAG="--steps $2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--resume] [--cell 'A B C D'] [--steps N]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "============================================"
echo "  TT-nn Training Pipeline"
echo "  Cells: $CELLS"
echo "  Resume: ${RESUME_FLAG:-no}"
echo "  Start:  $(date)"
echo "  Python: $PYTHON"
echo "============================================"
echo ""

# Track overall timing
PIPELINE_START=$(date +%s)

for CELL in $CELLS; do
    CONFIG="configs/cell_${CELL,,}_tt.yaml"
    if [[ ! -f "$CONFIG" ]]; then
        echo "ERROR: Config not found: $CONFIG"
        continue
    fi

    LOG_FILE="$LOG_DIR/cell_${CELL}_${TIMESTAMP}.log"
    FINAL_CKPT="$CKPT_DIR/cell_${CELL}_final.pt"

    echo "──────────────────────────────────────────"
    echo "  Cell $CELL → $LOG_FILE"
    echo "──────────────────────────────────────────"

    CELL_START=$(date +%s)

    # Build resume argument
    RESUME_ARG=""
    if [[ "$RESUME_FLAG" == "yes" && -f "$FINAL_CKPT" ]]; then
        echo "  Resuming from $FINAL_CKPT"
        RESUME_ARG="--resume $FINAL_CKPT"
    elif [[ "$RESUME_FLAG" == "yes" ]]; then
        echo "  No checkpoint found, starting fresh"
    fi

    # Run training
    $PYTHON -u train_ttnn.py \
        --config "$CONFIG" \
        --checkpoint_dir "$CKPT_DIR" \
        $STEPS_FLAG \
        $RESUME_ARG \
        2>&1 | tee "$LOG_FILE"

    CELL_END=$(date +%s)
    CELL_DURATION=$((CELL_END - CELL_START))
    CELL_HOURS=$((CELL_DURATION / 3600))
    CELL_MINS=$(((CELL_DURATION % 3600) / 60))

    echo ""
    echo "  Cell $CELL completed in ${CELL_HOURS}h ${CELL_MINS}m"
    echo ""

    # Sync device state between runs
    sleep 5
done

PIPELINE_END=$(date +%s)
PIPELINE_DURATION=$((PIPELINE_END - PIPELINE_START))
PIPELINE_HOURS=$((PIPELINE_DURATION / 3600))
PIPELINE_MINS=$(((PIPELINE_DURATION % 3600) / 60))

echo "============================================"
echo "  All cells completed!"
echo "  Total time: ${PIPELINE_HOURS}h ${PIPELINE_MINS}m"
echo "  End: $(date)"
echo "  Logs: $LOG_DIR/"
echo "  Checkpoints: $CKPT_DIR/"
echo "============================================"
