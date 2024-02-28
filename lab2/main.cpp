#include <omp.h>
#include <iostream>
#include <string>

void seqSaxpy(const int n, const float a, float* x, const int incX, float* y, const int incY) {
    for (size_t i = 0; i < n; i++)
        y[i * incY] += a * x[i * incX];
}

void seqDaxpy(const int n, const double a, double* x, const int incX, double* y, const int incY) {
    for (size_t i = 0; i < n; i++)
        y[i * incY] += a * x[i * incX];
}

void ompSaxpy(const int n, const float a, float* x, const int incX, float* y, const int incY) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++)
        y[i * incY] += a * x[i * incX];
}

void ompDaxpy(const int n, const double a, double* x, const int incX, double* y, const int incY) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++)
        y[i * incY] += a * x[i * incX];
}

void gpuSaxpy(const int n, const float a, float* x, const int incX, float* y, const int incY,
                                                            const int numBlocks, const int blockSize);
void gpuDaxpy(const int n, const double a, double* x, const int incX, double* y, const int incY,
                                                            const int numBlocks, const int blockSize);
template <typename T>
std::string chkRes(T* chk, T* res, const int size) {
    for (int i = 0; i < size; i++)
        if (std::abs(chk[i] - res[i]) >= std::numeric_limits<T>::epsilon())
            return "    Not ok";
    return "    Ok";
}

int main() {
    const int n = 250000000;
    const int incX = 1;
    const int incY = 2;
    const int xSize = 1 + (n - 1) * abs(incX);
    const int ySize = 1 + (n - 1) * abs(incY);
    const float a_f = 6.0f;
    float* x_f = new float[xSize];
    float* y_f = new float[ySize];
    float* seqRes_f = new float[ySize];

    std::fill(x_f, x_f + xSize, 2.0f);
    std::fill(y_f, y_f + ySize, 1.0f);
    
    double start = omp_get_wtime();
    seqSaxpy(n, a_f, x_f, incX, y_f, incY);
    double finish = omp_get_wtime();
    std::cout << "seqSaxpy: " << finish - start << "\n";
    std::copy(y_f, y_f + ySize, seqRes_f);

    std::fill(x_f, x_f + xSize, 2.0f);
    std::fill(y_f, y_f + ySize, 1.0f);

    start = omp_get_wtime();
    ompSaxpy(n, a_f, x_f, incX, y_f, incY);
    finish = omp_get_wtime();
    std::cout << "ompSaxpy: " << finish - start << chkRes<float>(y_f, seqRes_f, ySize) << "\n";

    for (int i = 8; i <= 256; i *= 2) {
        std::fill(x_f, x_f + xSize, 2.0f);
        std::fill(y_f, y_f + ySize, 1.0f);
        start = omp_get_wtime();
        gpuSaxpy(n, a_f, x_f, incX, y_f, incY, (n + i - 1) / i, i);
        finish = omp_get_wtime();
        std::cout << "gpuSaxpy(i=" << i << "): " << finish - start << chkRes<float>(y_f, seqRes_f, ySize) << "\n";
    }

    delete[] x_f;
    delete[] y_f;
    delete[] seqRes_f;
    std::cout << "\n\n";

    const double a = 6.0;
    double* x = new double[xSize];
    double* y = new double[ySize];
    double* seqRes = new double[ySize];

    std::fill(x, x + xSize, 2.0);
    std::fill(y, y + ySize, 1.0);

    start = omp_get_wtime();
    seqDaxpy(n, a, x, incX, y, incY);
    finish = omp_get_wtime();
    std::cout << "seqDaxpy: " << finish - start << "\n";
    std::copy(y, y + ySize, seqRes);

    std::fill(x, x + xSize, 2.0);
    std::fill(y, y + ySize, 1.0);

    start = omp_get_wtime();
    ompDaxpy(n, a, x, incX, y, incY);
    finish = omp_get_wtime();
    std::cout << "ompDaxpy: " << finish - start << chkRes<double>(y, seqRes, ySize) << "\n";

    for (int i = 8; i <= 256; i *= 2) {
        std::fill(x, x + xSize, 2.0);
        std::fill(y, y + ySize, 1.0);
        start = omp_get_wtime();
        gpuDaxpy(n, a, x, incX, y, incY, (n + i - 1) / i, i);
        finish = omp_get_wtime();
        std::cout << "gpuDaxpy(i=" << i << "): " << finish - start << chkRes<double>(y, seqRes, ySize) << "\n";
    }

    return 0;
}
