
import os
import re
import csv
import matplotlib.pyplot as plt

PROJECT = "/content/CUDA_Image_Processing_Project"
LOG_FILE = os.path.join(PROJECT, "results/output.txt")
RESULT_DIR = os.path.join(PROJECT, "results")

with open(LOG_FILE, "r") as file:
    log = file.read()

def extract(pattern):
    match = re.search(pattern, log)
    if not match:
        raise ValueError("Missing measurement: " + pattern)
    return float(match.group(1))

images = int(extract(r"Images processed:\s*(\d+)"))
gpu_ms = extract(r"GPU kernel time:\s*([\d.]+)\s*ms")
total_ms = extract(r"Total execution time:\s*([\d.]+)\s*ms")
average_ms = extract(
    r"Average GPU kernel time per image:\s*([\d.]+)\s*ms"
)

if images <= 0 or gpu_ms <= 0 or total_ms <= 0:
    raise ValueError("Invalid execution measurements")

throughput = images / (total_ms / 1000)

csv_file = os.path.join(RESULT_DIR, "performance.csv")

with open(csv_file, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "Images_Processed",
        "GPU_Kernel_Time_ms",
        "Total_Execution_Time_ms",
        "Average_GPU_Time_Per_Image_ms",
        "Overall_Throughput_images_per_second"
    ])

    writer.writerow([
        images,
        gpu_ms,
        total_ms,
        average_ms,
        throughput
    ])

# Generate performance graph
labels = [
    "GPU Kernel Time",
    "Total Execution Time"
]

values = [gpu_ms, total_ms]

plt.figure(figsize=(9, 5))
bars = plt.bar(labels, values)

plt.ylabel("Execution Time (milliseconds)")
plt.title("CUDA Image Processing Performance")
plt.grid(axis="y", alpha=0.3)

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.3f} ms",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

graph_file = os.path.join(
    RESULT_DIR, "performance.png"
)

plt.savefig(graph_file, dpi=200)
plt.show()

print("========== PERFORMANCE ANALYSIS ==========")
print("Images processed:", images)
print("GPU kernel time:", gpu_ms, "ms")
print("Total execution time:", total_ms, "ms")
print("Average GPU time per image:", average_ms, "ms")
print("Overall throughput:", round(throughput, 2), "images/sec")

print("\nGenerated files:")
print(csv_file)
print(graph_file)
