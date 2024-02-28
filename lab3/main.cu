#include <omp.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

__global__ void matrMultKernel(const int m, const int n, const int k, float* x, float* y, float* z) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < k && row < m)
        for (int i = 0; i < n; i++)
            z[row * k + col] += x[row * n + i] * y[i * k + col];
}

__global__ void blockMatrMultKernel(const int m, const int n, const int k, float* a, float* b, float* c) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    float res = 0;
    __shared__ float aBlock[16 * 16];
    __shared__ float bBlock[16 * 16];

    if (col < k && row < m) {
        for (int i = 0; i < n; i += blockDim.y) {
            aBlock[threadIdx.y * blockDim.x + threadIdx.x] = a[row * n + (i + threadIdx.x)];
            bBlock[threadIdx.y * blockDim.y + threadIdx.x] = b[(i + threadIdx.y) * k + col];
            __syncthreads();

            for (int j = 0; j < blockDim.x; j++)
                res += aBlock[threadIdx.y * blockDim.y + j] * bBlock[j * blockDim.y + threadIdx.x];
            __syncthreads();
        }
        c[row * k + col] += res;
    }
}

void gpuMatrMult(const int m, const int n, const int k, const float* a, const float* b, float* c,
                                                                    const dim3 dimGrid, const dim3 dimBlock) {
    cudaError_t cudaStatus;
    float* gpu_a = nullptr;
    float* gpu_b = nullptr;
    float* gpu_c = nullptr;

    cudaStatus = cudaMalloc((void**)&gpu_a, n * m * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc x failed!"); goto Error;
    }
    cudaStatus = cudaMalloc((void**)&gpu_b, n * k * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc y failed!"); goto Error;
    }
    cudaStatus = cudaMalloc((void**)&gpu_c, m * k * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc z failed!"); goto Error;
    }

    cudaStatus = cudaMemcpy(gpu_a, a, n * m * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy x failed!"); goto Error;
    }
    cudaStatus = cudaMemcpy(gpu_b, b, n * k * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy y failed!"); goto Error;
    }
    cudaStatus = cudaMemcpy(gpu_c, c, m * k * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy z failed!"); goto Error;
    }

    matrMultKernel<<<dimGrid, dimBlock>>>(m, n, k, gpu_a, gpu_b, gpu_c);

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize error code: %d\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(c, gpu_c, m * k * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);
}

void gpuBlockMatrMult(const int m, const int n, const int k, const float* a, const float* b, float* c,
                                                                        const dim3 dimGrid, const dim3 dimBlock) {
    cudaError_t cudaStatus;
    float* gpu_a = nullptr;
    float* gpu_b = nullptr;
    float* gpu_c = nullptr;

    cudaStatus = cudaMalloc((void**)&gpu_a, n * m * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc x failed!"); goto Error;
    }
    cudaStatus = cudaMalloc((void**)&gpu_b, n * k * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc y failed!"); goto Error;
    }
    cudaStatus = cudaMalloc((void**)&gpu_c, m * k * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc z failed!"); goto Error;
    }

    cudaStatus = cudaMemcpy(gpu_a, a, n * m * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy x failed!"); goto Error;
    }
    cudaStatus = cudaMemcpy(gpu_b, b, n * k * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy y failed!"); goto Error;
    }
    cudaStatus = cudaMemcpy(gpu_c, c, m * k * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy z failed!"); goto Error;
    }

    blockMatrMultKernel<<<dimGrid, dimBlock>>>(m, n, k, gpu_a, gpu_b, gpu_c);

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize error code: %d\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(c, gpu_c, m * k * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);
}

void seqMatrMult(const int m, const int n, const int k, float* a, float* b, float* c) {
    for (int i = 0; i < m; ++i)
        for (int p = 0; p < k; ++p)
            for (int j = 0; j < n; ++j)
                c[i * k + p] += a[i * n + j] * b[j * k + p];
}

void ompMatrMult(const int m, const int n, const int k, float* a, float* b, float* c) {
    #pragma omp parallel for
    for (int i = 0; i < m; ++i)
        for (int p = 0; p < k; ++p)
            for (int j = 0; j < n; ++j)
                c[i * k + p] += a[i * n + j] * b[j * k + p];
}

int main() {
    const int m = 1024;
    const int n = 1024;
    const int k = 1024;
    float* a = new float[m * n];
    float* b = new float[n * k];
    float* c = new float[m * k];
    std::fill(a, a + m * n, 1.0f);
    std::fill(b, b + n * k, 1.0f);
    std::fill(c, c + m * k, 0.0f);

    double start = omp_get_wtime();
    seqMatrMult(m, n, k, a, b, c);
    double finish = omp_get_wtime();
    std::cout << "seqMatrMult: " << finish - start << "\n";

    std::fill(c, c + m * k, 0.0f);

    start = omp_get_wtime();
    ompMatrMult(m, n, k, a, b, c);
    finish = omp_get_wtime();
    std::cout << "ompMatrMult: " << finish - start << "\n";

    std::fill(c, c + m * k, 0.0f);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((k + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);

    start = omp_get_wtime();
    gpuMatrMult(m, n, k, a, b, c, dimGrid, dimBlock);
    finish = omp_get_wtime();
    std::cout << "gpuMatrMult: " << finish - start << "\n";

    std::fill(c, c + m * k, 0.0f);

    start = omp_get_wtime();
    gpuBlockMatrMult(m, n, k, a, b, c, dimGrid, dimBlock);
    finish = omp_get_wtime();
    std::cout << "gpuBlockMatrMult: " << finish - start << "\n";

    return 0;
}
