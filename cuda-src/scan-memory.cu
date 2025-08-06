#include <stdio.h>

__global__ void hello_from_gpu() {
    printf("Hello from thread %d (block %d)\n", threadIdx.x, blockIdx.x);
}


// This function runs on the CPU
int main() {
    // Launch the kernel with 2 blocks, 4 threads per block
    hello_from_gpu<<<2, 4>>>();

    // Wait for GPU to finish before exiting
    cudaDeviceSynchronize();

    return 0;
}