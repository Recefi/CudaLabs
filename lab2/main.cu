
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

__global__ void saxpyKernel(const int n, const float a, float* x, const int incX, float* y, const int incY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i * incY] += a * x[i * incX];
}
__global__ void daxpyKernel(const int n, const double a, double* x, const int incX, double* y, const int incY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i * incY] += a * x[i * incX];
}

void gpuSaxpy(const int n, const float a, float* x, const int incX, float* y, const int incY,
                                                            const int numBlocks, const int blockSize) {
    cudaError_t cudaStatus;
    float* gpu_x = nullptr;
    float* gpu_y = nullptr;
    int xSize = 1 + (n - 1) * abs(incX);
    int ySize = 1 + (n - 1) * abs(incY);

    cudaStatus = cudaMalloc((void**)&gpu_x, xSize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc x failed!"); goto Error;
    }
    cudaStatus = cudaMalloc((void**)&gpu_y, ySize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc y failed!"); goto Error;
    }

    cudaStatus = cudaMemcpy(gpu_x, x, xSize * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!"); goto Error;
    }
    cudaStatus = cudaMemcpy(gpu_y, y, ySize * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!"); goto Error;
    }

    saxpyKernel<<<numBlocks, blockSize>>>(n, a, gpu_x, incX, gpu_y, incY);

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize error code: %d\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(y, gpu_y, ySize * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(gpu_x);
    cudaFree(gpu_y);
}

void gpuDaxpy(const int n, const double a, double* x, const int incX, double* y, const int incY,
                                                                const int numBlocks, const int blockSize) {
    cudaError_t cudaStatus;
    double* gpu_x = nullptr;
    double* gpu_y = nullptr;
    int xSize = 1 + (n - 1) * abs(incX);
    int ySize = 1 + (n - 1) * abs(incY);

    cudaStatus = cudaMalloc((void**)&gpu_x, xSize * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc x failed!"); goto Error;
    }
    cudaStatus = cudaMalloc((void**)&gpu_y, ySize * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc y failed!"); goto Error;
    }

    cudaStatus = cudaMemcpy(gpu_x, x, xSize * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!"); goto Error;
    }
    cudaStatus = cudaMemcpy(gpu_y, y, ySize * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!"); goto Error;
    }

    daxpyKernel<<<numBlocks, blockSize>>>(n, a, gpu_x, incX, gpu_y, incY);

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize error code: %d\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(y, gpu_y, ySize * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(gpu_x);
    cudaFree(gpu_y);
}
