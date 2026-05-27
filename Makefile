# VaultWatch Makefile
# CPU verifier (EC from scratch, no libsecp256k1)
CXX = g++
CXXFLAGS = -O3 -std=c++17 -pthread -fopenmp -march=native -flto
LDFLAGS = -pthread -fopenmp -flto
TARGET = vaultwatch
OBJS = main.o

all: $(TARGET)

main.o: main.cpp targets.h ec_jacobian.h math256.h
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

$(TARGET): main.o
	$(CXX) $(CXXFLAGS) main.o -o $(TARGET) $(LDFLAGS)

# VaultWatch Integrated (GPU + single-pass)
TARGET_INT = vaultwatch-integrated
CUDA_HOME ?= /usr/local/cuda
NVCC = $(CUDA_HOME)/bin/nvcc
NVFLAGS = -O3 -arch=native -std=c++17 -lineinfo

$(TARGET_INT): vaultwatch-integrated.cu ec_jacobian.h math256.h
	$(NVCC) $(NVFLAGS) vaultwatch-integrated.cu -o $(TARGET_INT)

# Run targets
run-cpu: $(TARGET)
	./$(TARGET) --keys keys.bin --targets patoshi_h160.bin

run-gpu: $(TARGET_INT)
	./$(TARGET_INT) --mode H --ts-start 1288834970 --ts-end 1288924970

run-gpu-m: $(TARGET_INT)
	./$(TARGET_INT) --mode M --ts-start 1288834970 --ts-end 1288924970

run-gpu-all: $(TARGET_INT)
	./$(TARGET_INT) --mode ALL --ts-start 1262304000 --ts-end 1356998400 --progress

# Data preparation
dump-addresses:
	python3 dump_addresses.py

.PHONY: all clean run-cpu run-gpu run-gpu-m run-gpu-all dump-addresses

clean:
	rm -f $(TARGET) $(TARGET_INT) *.o *.bin found.txt