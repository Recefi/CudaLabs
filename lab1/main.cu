
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void printKernel() {
    int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("I am from %d block, %d thread (global index: %d)\n", blockIdx.x, threadIdx.x, gIdx);
}
__global__ void calcKernel(int* a, unsigned int size) {
    int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
    a[gIdx] = a[gIdx] + gIdx;
}

cudaError_t printCalcGpu(int* a, unsigned int size) {
    int* gpu_a = nullptr;
    cudaError_t cudaStatus;

    printKernel<<<2, 2>>>();

    cudaStatus = cudaMalloc((void**)&gpu_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(gpu_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    calcKernel<<<2, 2>>>(gpu_a, size);

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize error code: %d\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(a, gpu_a, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(gpu_a);
    return cudaStatus;
}

int main() {
    const int size = 4;
    int a[size] = { 1, 2, 3, 4};

    printf("a = {%d,%d,%d,%d}\n", a[0], a[1], a[2], a[3]);

    cudaError_t cudaStatus = printCalcGpu(a, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "printCalcGpu failed!");
        return 1;
    }

    printf("a = {%d,%d,%d,%d}\n", a[0], a[1], a[2], a[3]);

    return 0;
}
