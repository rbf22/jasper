#!/bin/bash
# Monitor training jobs — reports every 10 minutes
# Usage: nohup ./monitor_training.sh > logs/monitor.log 2>&1 &
#
# Runs are matched by their --config argument in the process table (pgrep),
# so no PID files are needed. Update the RUNS array below when starting
# new runs.

cd /home/rfenwick/Documents/jasper/workspace-poc

# Runs to track, one per line: "name|config|logfile"
#   name   — display label
#   config — the --config value used by train_ttnn.py (for pgrep matching)
#   log    — path to the run's stdout log
RUNS=(
    "Cell C (AR)|configs/cell_c_attn_residual.yaml|logs/cell_c_ar_chain_fix_20260804.log"
    "Cell C (AR+K3)|configs/cell_c_attn_residual_k3.yaml|logs/cell_c_k3_chainfix_20260804.log"
)

# Find the python PID for a given --config value.
# pgrep -f matches both the bash wrapper and the python process; we filter
# to the python process itself.
find_pid() {
    local config="$1"
    for pid in $(pgrep -f "train_ttnn.py --config ${config} "); do
        comm=$(ps -p "$pid" -o comm= 2>/dev/null)
        [[ "$comm" == "python" || "$comm" == "python3" ]] && { echo "$pid"; return; }
    done
}

# Print elapsed time for a PID (e.g. "36:23" or "1-02:03:04").
elapsed_of() {
    ps -p "$1" -o etime= 2>/dev/null | tr -d ' '
}

while true; do
    echo "========================================================"
    echo "Training monitor: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================================"

    for entry in "${RUNS[@]}"; do
        IFS='|' read -r name config log <<< "$entry"

        echo ""
        echo "--- $name ---"
        echo "  Config: $config"
        echo "  Log:    $log"

        # Process status
        pid=$(find_pid "$config")
        if [ -n "$pid" ]; then
            elapsed=$(elapsed_of "$pid")
            cpu=$(ps -p "$pid" -o pcpu= 2>/dev/null | tr -d ' ')
            mem=$(ps -p "$pid" -o pmem= 2>/dev/null | tr -d ' ')
            echo "  Status: RUNNING (PID $pid, elapsed ${elapsed}, ${cpu}% CPU, ${mem}% MEM)"
        else
            echo "  Status: STOPPED (no matching process)"
        fi

        # Latest training step lines (start with whitespace + number + number)
        latest=$(grep -E '^\s+[0-9]+\s+[0-9]+\.' "$log" 2>/dev/null | tail -5)
        if [ -n "$latest" ]; then
            echo "  Recent steps:"
            echo "$latest" | while IFS= read -r line; do
                echo "    $line"
            done
            # Summary: current step + loss from the last step line
            last_line=$(echo "$latest" | tail -1)
            cur_step=$(echo "$last_line" | awk '{print $1}')
            cur_loss=$(echo "$last_line" | awk '{print $2}')
            cur_tps=$(echo "$last_line" | awk '{print $5}')
            cur_grad=$(echo "$last_line" | awk '{print $6}')
            echo "  Current: step ${cur_step}, loss ${cur_loss}, ${cur_tps} tok/s, grad_norm ${cur_grad}"
        else
            echo "  Recent steps: (none yet)"
        fi

        # Latest checkpoint — extracted from the log itself so each run's
        # checkpoint is reported correctly even when multiple runs share the
        # same checkpoint filename prefix.
        ckpt_line=$(grep "Checkpoint saved to" "$log" 2>/dev/null | tail -1)
        if [ -n "$ckpt_line" ]; then
            echo "  Latest checkpoint: $(echo "$ckpt_line" | sed 's/.*saved to /saved /')"
        else
            echo "  Latest checkpoint: (none yet)"
        fi

        # Check for errors (filter known TT-nn noise)
        errors=$(grep -iE 'error|traceback|exception|fatal|nan|inf' "$log" 2>/dev/null \
                 | grep -v 'sfpi\|deprecated\|TT_METAL_LOGGER\|vConst\|NoDebias\|compile\|warning\|nanobind\|leaked' \
                 | tail -3)
        if [ -n "$errors" ]; then
            echo "  *** ERRORS DETECTED ***"
            echo "$errors" | while IFS= read -r line; do
                echo "    $line"
            done
        fi

        # Early stopping status
        early_stop=$(grep "Early stopping" "$log" 2>/dev/null | tail -1)
        if [ -n "$early_stop" ]; then
            echo "  Early stop: $early_stop"
        fi
    done

    echo ""
    echo "========================================================"
    echo ""

    sleep 600  # 10 minutes
done
