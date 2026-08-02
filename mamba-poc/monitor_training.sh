#!/bin/bash
# Monitor training jobs — reports every 10 minutes
# Usage: nohup ./monitor_training.sh > logs/monitor.log 2>&1 &

cd /home/rfenwick/Documents/jasper/mamba-poc

while true; do
    echo "========================================================"
    echo "Training monitor: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================================"

    for cell in A B C; do
        log="logs/cell_$(echo $cell | tr A-Z a-z).log"
        pid_file="run_${cell}/cell_$(echo $cell | tr A-Z a-z).pid"

        echo ""
        echo "--- Cell $cell ---"

        # Check if process is alive
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Status: RUNNING (PID $pid)"
            else
                echo "  Status: STOPPED (PID $pid no longer exists)"
            fi
        else
            echo "  Status: NO PID FILE"
        fi

        # Latest training lines (step lines start with whitespace + number)
        latest=$(grep -E '^\s+[0-9]+\s+[0-9]+\.' "$log" 2>/dev/null | tail -5)
        if [ -n "$latest" ]; then
            echo "  Recent steps:"
            echo "$latest" | while IFS= read -r line; do
                echo "    $line"
            done
        else
            echo "  Recent steps: (none yet)"
        fi

        # Latest checkpoint
        ckpt=$(ls -t "checkpoints/cell_${cell}_step"*.pt 2>/dev/null | head -1)
        if [ -n "$ckpt" ]; then
            echo "  Latest checkpoint: $(basename $ckpt)"
        else
            echo "  Latest checkpoint: (none yet)"
        fi

        # Check for errors
        errors=$(grep -iE 'error|traceback|exception|fatal|nan|inf' "$log" 2>/dev/null | grep -v 'sfpi\|deprecated\|TT_METAL_LOGGER\|vConst\|NoDebias\|compile\|warning' | tail -3)
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
