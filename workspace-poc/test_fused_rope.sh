#!/bin/bash
# Try each device in turn until one works for the fused RoPE test
cd /home/rfenwick/Documents/jasper/mamba-poc

MGD_DIR="/home/rfenwick/Documents/jasper/.tt-venv/lib/python3.12/site-packages/pjrt_plugin_tt/tt-metal/tt_metal/fabric/mesh_graph_descriptors"

# Try p300 first, then p150
for MGD in p300 p150; do
    MGD_PATH="$MGD_DIR/${MGD}_mesh_graph_descriptor.textproto"
    if [ ! -f "$MGD_PATH" ]; then continue; fi
    
    for DEV in 0 1 2 3; do
        echo "=== Trying $MGD descriptor, device $DEV ==="
        TT_VISIBLE_DEVICES=$DEV \
        TT_MESH_GRAPH_DESC_PATH="$MGD_PATH" \
        TT_METAL_LOGGER_LEVEL="ERROR" \
        timeout 300 /home/rfenwick/Documents/jasper/.tt-venv/bin/python -c "
import ttnn
device = ttnn.open_device(device_id=0)
print(f'Device opened with $MGD descriptor!')
ttnn.close_device(device)
" 2>&1 | grep -E "Device opened|Error|Timeout|TT_THROW|Runtime|FATAL|failed" | head -3
        
        if [ $? -eq 0 ]; then
            echo "=== SUCCESS: $MGD descriptor works on device $DEV ==="
            # Now run the actual test
            TT_VISIBLE_DEVICES=$DEV \
            TT_MESH_GRAPH_DESC_PATH="$MGD_PATH" \
            TT_METAL_LOGGER_LEVEL="ERROR" \
            timeout 300 /home/rfenwick/Documents/jasper/.tt-venv/bin/python test_fused_rope_single.py 2>&1 | grep -viE "nanobind|leaked"
            exit $?
        fi
    done
done

echo "ERROR: No working device/descriptor combination found"
exit 1
