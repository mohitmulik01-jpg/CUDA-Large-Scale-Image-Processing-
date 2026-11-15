
# Large-Scale Image Processing Using CUDA

## 1. Project Overview

This project implements a GPU-accelerated image processing
pipeline using CUDA C++.

The application processes multiple RGB images using three
parallel image processing operations:

1. Grayscale conversion
2. Gaussian blur
3. Sobel edge detection

The objective is to demonstrate how CUDA GPU parallelism
can efficiently perform image processing across a large
collection of images.

## 2. Technologies Used

- C++
- NVIDIA CUDA C++
- NVIDIA Tesla T4 GPU
- CUDA Runtime API
- Python
- NumPy
- Matplotlib
- Google Colab

## 3. Dataset

The project uses 200 synthetically generated RGB images.

Image specifications:

- Number of images: 200
- Width: 256 pixels
- Height: 256 pixels
- Color channels: 3 (RGB)
- Image format: PPM (P6)

Each image contains generated color gradients, circular
patterns and deterministic noise.

## 4. CUDA Algorithms

### Grayscale Conversion

Converts RGB pixels into grayscale using:

Gray = 0.299R + 0.587G + 0.114B

Each CUDA thread processes one pixel.

### Gaussian Blur

Applies a 3x3 Gaussian filter to reduce image noise.

The filter weights are:

1 2 1
2 4 2
1 2 1

The weighted sum is divided by 16.

### Sobel Edge Detection

Uses horizontal and vertical Sobel operators to detect
image boundaries.

The gradient magnitude is calculated using:

G = sqrt(Gx² + Gy²)

The output represents detected edges.

## 5. GPU Execution

The application uses CUDA kernels and parallel thread
execution to process image pixels.

The processing pipeline is:

RGB Input
    |
Grayscale Conversion
    |
Gaussian Blur
    |
Sobel Edge Detection
    |
Processed Output

## 6. Execution Environment

GPU: NVIDIA Tesla T4
CUDA Toolkit: 13.0
Compiler: NVCC
Platform: Google Colab

## 7. Observed Execution Results

Actual recorded execution:

- Images processed: 200
- GPU kernel time: 5.00541 ms
- Total execution time: 301.859 ms
- Average GPU kernel time per image: 0.025027 ms
- Overall throughput: approximately 662.56 images/second

These measurements are from one recorded execution.
Results may vary depending on GPU availability and
runtime conditions.

GPU kernel time measures the three CUDA kernels.
Total execution time also includes image loading,
memory allocation, transfers and output saving.

## 8. Project Structure

CUDA_Image_Processing_Project/
|
|-- src/
|   |-- main.cu
|
|-- data/
|   |-- input/
|   |-- output/
|
|-- results/
|   |-- output.txt
|   |-- performance.csv
|   |-- performance.png
|
|-- screenshots/
|
|-- report/
|
|-- Makefile
|-- run.sh
|-- analyze_results.py
|-- README.md

## 9. How to Compile

Requirements:
- NVIDIA GPU
- CUDA Toolkit
- NVCC compiler
- C++17 compatible compiler

Compile using:

make build

## 10. How to Execute

Run the application:

./image_processor --input data/input --output data/output

Or execute the complete workflow:

chmod +x run.sh
./run.sh

## 11. Command Line Arguments

--input  : Directory containing input PPM images.
--output : Directory for processed images.

Example:

./image_processor \
    --input data/input \
    --output data/output

## 12. Performance Analysis

The project records GPU kernel execution time and
total application execution time.

Performance results are saved in:

results/performance.csv

A performance graph is generated at:

results/performance.png

## 13. Learning Outcomes

- Understanding CUDA kernel execution.
- Understanding GPU thread and block organization.
- Implementing image processing algorithms using CUDA.
- Managing host-to-device and device-to-host memory transfers.
- Measuring GPU kernel execution time.
- Processing multiple images in a single application.
- Organizing a CUDA project for reproducible execution.

## 14. Conclusion

This project demonstrates a CUDA-based image processing
pipeline that successfully processed 200 images on an
NVIDIA Tesla T4 GPU.

It demonstrates the application of parallel computing
to image processing through grayscale conversion,
Gaussian filtering and Sobel edge detection.
