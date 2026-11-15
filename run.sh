
#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

echo "======================================"
echo "CUDA LARGE-SCALE IMAGE PROCESSING"
echo "======================================"

echo ""
echo "Checking NVIDIA GPU..."
nvidia-smi

echo ""
echo "Compiling CUDA project..."
make clean
make build

echo ""
echo "Executing image processing..."

mkdir -p results data/output

./image_processor \
    --input data/input \
    --output data/output \
    2>&1 | tee results/output.txt

echo ""
echo "Execution completed successfully!"
echo "Check results/output.txt for execution logs."
