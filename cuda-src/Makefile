# Makefile

TARGET = scan-memory
SRC = scan-memory.cu
NVCC = nvcc
NVCC_FLAGS = -arch=sm_89 -O2

LIBTORCH_PATH ?= $(LIBTORCH_DIR)

ifeq ($(LIBTORCH_PATH),)
$(error Please set LIBTORCH_DIR to the path to your LibTorch installation)
endif

INCLUDES = -I$(LIBTORCH_PATH)/include -I$(LIBTORCH_PATH)/include/torch/csrc/api/include

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

clean:
	rm -f $(TARGET)
