# Makefile

# The name of the final executable
TARGET = scan-memory

# The source file
SRC = scan-memory.cu

# The compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -arch=sm_89 -O2

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
