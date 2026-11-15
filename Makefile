
NVCC = nvcc
TARGET = image_processor
SOURCE = src/main.cu

NVCCFLAGS = -O2 -std=c++17

.PHONY: all build run clean

all: build

build:
	$(NVCC) $(NVCCFLAGS) $(SOURCE) -o $(TARGET)

run: build
	./$(TARGET) --input data/input --output data/output

clean:
	rm -f $(TARGET)
	rm -f data/output/*.ppm
	rm -f results/output.txt
