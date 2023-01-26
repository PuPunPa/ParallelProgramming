
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void add(int a[], int b[], int c[]) {
    for (int i = 0; i < sizeof(a); i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 5;
    int a[N] = { 2, 2, 2, 1, 11 };
    int b[N] = { 1, 5, 3, 2, 7 };
    int c[N] = { 0 };
    int size = N * sizeof(int);
    int* d_a = 0;
    int* d_b = 0;
    int* d_c = 0;

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice);

    add << <1, N >> > (d_a, d_b, d_c);

    cudaDeviceSynchronize();

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    printf("{ 2, 2, 2, 1, 11 } + { 1, 5, 3, 2, 7 } = { %d, %d, %d, %d, %d }\n",
        c[0], c[1], c[2], c[3], c[4]);

    cudaDeviceReset();

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}