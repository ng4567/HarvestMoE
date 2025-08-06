#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>


// Gloabal functions run on GPU
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

    size_t expert_size = 200 * 1024 * 1024;
    int num_gpus = get_num_gpus();
    for (int i = 0; i < num_gpus; i++) {
        get_num_experts_that_fit(expert_size, i);
    }

    return 0;

}