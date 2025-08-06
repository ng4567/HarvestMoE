# Makefile

# The name of the final executable
TARGET = scan-memory

# The source file
SRC = scan-memory.cu

# The compiler
NVCC = nvcc

# Get PyTorch include and library paths
TORCH_INCLUDE := $(shell python -c "import torch; print(' '.join(['-I' + p for p in torch.utils.cpp_extension.include_paths()]))")
TORCH_LIBS := $(shell python -c "import torch; print(' '.join(['-L' + p for p in torch.utils.cpp_extension.library_paths()]))")

# Compiler flags
NVCC_FLAGS = -arch=sm_89 -O2 $(TORCH_INCLUDE) $(TORCH_LIBS) -ltorch -ltorch_cuda

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
