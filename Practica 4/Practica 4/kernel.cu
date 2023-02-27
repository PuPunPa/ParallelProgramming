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

__global__ void mulMatrixGPU(int* a, int* b, int* c, int width, int rows, int cols);

int main()
{
    const int n = 2;
    int size = n * sizeof(n);

    // Declare 2x2 Matrix
    int a[n * n];
    int b[n * n];
    int c[n * n];

    // Assign random value between 0-255 to every position
    for (int i = 0; i < n * n; i++) {
        a[i] = rand() % 226;
        b[i] = rand() % 226;
    }

    int* devA = 0;
    int* devB = 0;
    int* devC = 0;

    // Allocate Memory
    cudaMalloc((void**)&devA, size);
    cudaMalloc((void**)&devB, size);
    cudaMalloc((void**)&devC, size);

    // Copy to GPU
    cudaMemcpy(devA, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devC, c, size, cudaMemcpyHostToDevice);

    // Solve Operation
    dim3 grid(8, 4, 4);
    dim3 block(8, 4, 4);
    mulMatrixGPU << <grid, block >> > (devA, devB, devC, 2, 2, 2);

    // Clean
    cudaDeviceSynchronize();
    cudaDeviceReset();

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    return 0;
}

__global__ void mulMatrixGPU(int* a, int* b, int* c, int width, int rows, int cols) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int suma = 0;
    if (row < rows && col < cols) {
        for (int i = 0; i < width; i++) {
            suma += a[row * width + i] * b[i * width + col];
        }
        c[row * width + col] = suma;
    }

}