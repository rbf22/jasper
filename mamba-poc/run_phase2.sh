#!/bin/bash
# Phase 2: Run Cell A (resume) + Cell E (control) alongside Cell D (still running)
#
# Prerequisites:
#   - Cells B and C have been killed (their devices are free)
#   - Cell D is still running on device 3
#   - Cell A checkpoint exists at run_A/checkpoints/cell_A_step300.pt
#
# Usage:
#   bash run_phase2.sh
#
# This launches:
#   Cell A on device 0 (resumed from step 300)
#   Cell E on device 1 (fresh start, B architecture with C/D learning rate)
#   Cell D continues on device 3 (already running)

set -e

cd /home/ttuser/Documents/jasper/mamba-poc
PY=/home/ttuser/Documents/jasper/.tt-venv/bin/python
TS=$(date +%s)

# Verify Cell D is still running
if ! pgrep -f "cell_d_tt.yaml" > /dev/null; then
    echo "WARNING: Cell D does not appear to be running!"
    echo "Proceeding anyway, but Phase 2 may not make sense without D."
fi

# Verify Cell A checkpoint exists
A_CKPT="run_A/checkpoints/cell_A_step300.pt"
if [ ! -f "$A_CKPT" ]; then
    echo "ERROR: Cell A checkpoint not found at $A_CKPT"
    exit 1
fi

# Verify devices 0 and 1 are free (no training processes)
if pgrep -f "device.?0\|--device 0" > /dev/null 2>&1; then
    echo "WARNING: A process may still be using device 0"
fi
if pgrep -f "device.?1\|--device 1" > /dev/null 2>&1; then
    echo "WARNING: A process may still be using device 1"
fi

echo "=== Phase 2 Launch ==="
echo "  Cell A: device 0, resuming from $A_CKPT"
echo "  Cell E: device 1, fresh start (B arch, lr=2e-4)"
echo "  Cell D: device 3, already running"
echo ""

# Cell A on device 0 (resume from step 300)
cd /home/ttuser/Documents/jasper/mamba-poc/run_A
TT_METAL_CACHE="$PWD/tt_cache" \
  nohup $PY -u train_ttnn.py \
    --config configs/cell_a_tt.yaml \
    --device 0 \
    --checkpoint_dir checkpoints \
    --steps 30000 \
    --resume checkpoints/cell_A_step300.pt \
    > logs/A_${TS}.log 2>&1 &
A_PID=$!
echo "Cell A: PID $A_PID, device 0, resuming from step 300"

# Cell E on device 1 (fresh start)
cd /home/ttuser/Documents/jasper/mamba-poc/run_E
TT_METAL_CACHE="$PWD/tt_cache" \
  nohup $PY -u train_ttnn.py \
    --config configs/cell_e_tt.yaml \
    --device 1 \
    --checkpoint_dir checkpoints \
    --steps 30000 \
    > logs/E_${TS}.log 2>&1 &
E_PID=$!
echo "Cell E: PID $E_PID, device 1, fresh start (lr=2e-4)"

cd /home/ttuser/Documents/jasper/mamba-poc
echo ""
echo "Phase 2 launched at $(date)"
echo "  Cell A: PID $A_PID"
echo "  Cell E: PID $E_PID"
echo ""
echo "Monitor with:"
echo "  for CELL in A D E; do LOG=\$(ls -t run_\${CELL}/logs/\${CELL}_*.log | head -1); echo \"=== Cell \$CELL ===\"; grep -E '^\s+[0-9]+' \$LOG | tail -3; done"
