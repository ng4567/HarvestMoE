#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

struct Expert {
    at::Tensor data;
    size_t     expert_size;
    int        gpu_id;
    bool       allocated = false;

    Expert(size_t size_bytes, int device_id)
      : expert_size(size_bytes), gpu_id(device_id) {}

    bool allocate_expert() {
        cudaSetDevice(gpu_id);
        try {
            data = at::empty(
              { static_cast<int64_t>(expert_size) },
              at::TensorOptions()
                .device(at::kCUDA, gpu_id)
                .dtype(at::kUInt8)
            );
            allocated = true;
            return true;
        } catch (const c10::Error &e) {
            std::cerr << "Allocation failed: " << e.what() << std::endl;
            return false;
        }
    }

    void free_expert() {
        if (allocated) {
            data.reset();
            allocated = false;
        }
    }

    bool copy_expert_to_gpu(const at::Tensor &cpu_tensor) {
        if (!allocated) {
            std::cerr << "Expert must be allocated before copying.\n";
            return false;
        }
        if (!cpu_tensor.device().is_cpu()) {
            std::cerr << "Source must be a CPU tensor.\n";
            return false;
        }
        if (cpu_tensor.numel() != data.numel()) {
            std::cerr << "Element count mismatch.\n";
            return false;
        }
        try {
            cudaSetDevice(gpu_id);
            data.copy_(cpu_tensor, true);
            return true;
        } catch (const c10::Error &e) {
            std::cerr << "Copy failed: " << e.what() << std::endl;
            return false;
        }
    }
};

// Global functions run on GPU
__global__ void hello_from_gpu() {
    printf("Hello from thread %d (block %d)\n", threadIdx.x, blockIdx.x);
}

int get_num_experts_that_fit(size_t expert_size, int gpu_id) {
    cudaError_t err = cudaSetDevice(gpu_id);
    if (err != cudaSuccess) {
        std::cerr << "cudaSetDevice failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    size_t free_mem, total_mem;
    err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemGetInfo failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    int experts_that_fit = free_mem / expert_size;
    if (experts_that_fit <= 0) {
        std::cerr << "No experts fit on GPU " << gpu_id << std::endl;
        return 0;
    }

    std::cout << "GPU " << gpu_id << " has " << free_mem / (1024 * 1024) << " MB free memory." << std::endl;
    std::cout << "GPU " << gpu_id << " fits " << experts_that_fit << " experts of size " << expert_size << " bytes." << std::endl;
    return experts_that_fit;
}

int get_num_gpus() {
    int num_gpus;
    cudaError_t err = cudaGetDeviceCount(&num_gpus);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    return num_gpus;
}


int main() {
    // Wait for GPU to finish before exiting
    cudaDeviceSynchronize();

    // Allocate dummy CPU tensor (200MB of uint8_t)
    size_t expert_size = 200 * 1024 * 1024;
    at::Tensor dummy_cpu_tensor = at::zeros({static_cast<long>(expert_size)}, at::kByte);

    // Create expert on GPU 0
    Expert expert(expert_size, 0);

    if (!expert.allocate_expert()) {
        std::cerr << "Failed to allocate expert on GPU 0" << std::endl;
        return 1;
    }

    if (!expert.copy_expert_to_gpu(dummy_cpu_tensor)) {
        std::cerr << "Failed to copy expert to GPU 0" << std::endl;
        expert.free_expert();
        return 1;
    }

    std::cout << "Successfully copied dummy expert to GPU 0!" << std::endl;

    // Free GPU memory
    expert.free_expert();
    return 0;
}
