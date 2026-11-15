
#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

echo "======================================"
echo "CUDA LARGE-SCALE IMAGE PROCESSING"
echo "======================================"

echo ""
echo "Step 1: Checking GPU..."
nvidia-smi

echo ""
echo "Step 2: Generating 200 input images..."
python3 generate_dataset.py

echo ""
echo "Step 3: Compiling CUDA application..."
make clean
make build

echo ""
echo "Step 4: Processing all images..."

mkdir -p results data/output

./image_processor \
    --input data/input \
    --output data/output \
    2>&1 | tee results/output.txt

echo ""
echo "Step 5: Verifying output..."

INPUT_COUNT=$(find data/input -name '*.ppm' | wc -l)
OUTPUT_COUNT=$(find data/output -name '*.ppm' | wc -l)

echo "Input images: $INPUT_COUNT"
echo "Output images: $OUTPUT_COUNT"

if [ "$INPUT_COUNT" -ne 200 ] || [ "$OUTPUT_COUNT" -ne 200 ]; then
    echo "ERROR: Image count verification failed!"
    exit 1
fi

echo ""
echo "SUCCESS: All 200 images processed!"
