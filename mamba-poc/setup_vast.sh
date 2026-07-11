#!/bin/bash
# Setup script for Vast.ai instances
# Run this once after creating your instance

set -e

echo "=== Installing dependencies ==="
pip install -q einops pyyaml wandb numpy

echo "=== Verifying GPU ==="
python -c "
import torch
n = torch.cuda.device_count()
print(f'GPUs: {n}')
for i in range(n):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} — {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB')
"

echo "=== Verifying data generation ==="
python data.py

echo "=== Verifying model param counts ==="
python model.py

echo ""
echo "=== Setup complete! ==="
echo "Next steps:"
echo "  1. python vast_runner.py --smoke-test    # verify training works"
echo "  2. python vast_runner.py --clean         # start training (parallel if 2+ GPUs)"
echo "  3. python vast_runner.py --sequential --clean  # or sequential on single GPU"
echo ""
echo "Monitor with: python vast_runner.py --status"
echo "Outputs save to: ./outputs/"
