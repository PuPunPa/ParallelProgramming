#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <random>
#include <stdio.h>
#include <iostream>
#include <ctime>

#define GPUErrorAssertion(ans) {gpuAssert((ans), __FILE__, __LINE__);};

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n\r", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void conv(int* a, int* b, int* ker, int n, int m, int kernelSize);

int main()
{
    const int len = 3;
    const int n = 8;
    int size = n * n * sizeof(int);
    int sizeLen = len * len * sizeof(int);

    int a[n * n];
    int b[n * n];
    int c[n * n];

    // Assign random value between 0-255 to every position
    for (int i = 0; i < n * n; i++) {
        a[i] = rand() % 3;
        b[i] = rand() % 3;
    }

    int* devA = 0;
    int* devB = 0;
    int* devC = 0;

    // Allocate Memory
    cudaMalloc((void**)&devA, size);
    cudaMalloc((void**)&devB, size);
    cudaMalloc((void**)&devC, sizeLen);

    c[0] = 0; c[1] = 1; c[2] = 0; c[3] = 0; c[4] = 0; c[5] = 0; c[6] = 0; c[7] = 0; c[8] = 0;

    // Copy to GPU
    cudaMemcpy(devA, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devC, c, sizeLen, cudaMemcpyHostToDevice);

    // Solve Operation
    dim3 block(8, 8);
    conv << <1, block >> > (devA, devB, devC, n, n, len);

    // Print solution
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", c[i * n + j]);
        }
        printf("\n");
    }

    // Clean
    cudaDeviceSynchronize();
    cudaDeviceReset();

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    return 0;
}

__global__ void conv(int* a, int* b, int* ker, int n, int m, int kernelSize) {
    int row = threadIdx.y;
    int col = threadIdx.x;

    int sum = 0;
    if (row > 0 && row < m - 1 && col>0 && col < n - 1) {
        for (int i = 0; i < kernelSize; i++) {
            for (int j = 0; j < kernelSize; j++) {
                sum += (a[(row - 1) * m + i + (col - 1) + j] * ker[i * kernelSize + j]);
            }
        }

        b[row * m + col] = sum;
    }
}