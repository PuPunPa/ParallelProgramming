
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void stream_test(int* in, int* out, int size) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < size) {
        for (int i = 0; i < 25; i++) {
            out[gid] = in[gid] + (in[gid] - 1) * (gid % 10);
        }
    }
}

__global__ void sum_array_overlap(int* a, int* b, int* c, int N) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < N) {
        c[gid] = a[gid] + b[gid];
    }
}

int main()
{
    int size = 1 << 25;
    int block_size = 128;

    size_t NO_BYTES = size * sizeof(int);

    int const NUM_STREAMS = 8;
    int ELEMENTS_PER_STREAM = size / NUM_STREAMS;
    int BYTES_PER_STREAM = NO_BYTES / NUM_STREAMS;

    int* h_a, * h_b, * gpu_result, * cpu_result;

    cudaMallocHost((void**)&h_a, NO_BYTES);
    cudaMallocHost((void**)&h_b, NO_BYTES);
    cudaMallocHost((void**)&gpu_result, NO_BYTES);

    cpu_result = (int*)malloc(NO_BYTES);
    int* d_a, * d_b, * d_c;

    cudaMalloc((void**)&d_a, NO_BYTES);
    cudaMalloc((void**)&d_b, NO_BYTES);
    cudaMalloc((void**)&d_c, NO_BYTES);

    srand((double)time(NULL));
    for (int i = 0; i < size; i++) {
        h_a[i] = rand();
        h_b[i] = rand();
    }

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    dim3 block(block_size);
    dim3 grid(ELEMENTS_PER_STREAM);

    int offset = 0;

    for (int i = 0; i < NUM_STREAMS; i++) {
        offset = i * ELEMENTS_PER_STREAM;
        cudaMemcpyAsync(&d_a[offset], &h_a[offset], BYTES_PER_STREAM, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(&d_b[offset], &h_b[offset], BYTES_PER_STREAM, cudaMemcpyHostToDevice, streams[i]);
        sum_array_overlap << <grid, block, 0, streams[i] >> > (&d_a[offset], &d_b[offset], &d_c[offset], size);
        cudaMemcpyAsync(&gpu_result[offset], &d_c[offset], BYTES_PER_STREAM, cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_STREAMS; i++) {
        offset = i * ELEMENTS_PER_STREAM;
        printf("%d\n", gpu_result[offset]);
    }

    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(gpu_result);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cudaDeviceReset;

    return 0;
}
