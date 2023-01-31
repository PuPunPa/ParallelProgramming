
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *input);

__global__ void idx_calc_tid(int* input) {
    int tid = threadIdx.x;
        printf("*[DEVICE] blockIdx.x; %d, threadIdx.x: %d, gid: %d, data: %d\n\r", tid, input[tid]);
}

__global__ void idx_calc_gid(int* input)
{
    int tid = threadIdx.x;
    int offset = blockIdx.x * blockDim.x;
    int gid = tid + offset;

    printf("*[DEVICE] blockIdx.x; %d, threadIdx.x: %d, gid: %d, data: %d\n\r", blockIdx.x, tid, gid, input[gid]);
}

__global__ void idx_calc_2d(int* input)
{
    int tid = threadIdx.x;
    int row_offset = gridDim.x * blockDim.x * blockIdx.y;
    int block_offset = blockIdx.x * blockDim.x;
    int gid = tid + row_offset + block_offset;

    printf("*[DEVICE] gridDim.x: %d, blockIdx.x: %d, blockIdx.y: %d, threadIdx.x: %d, gid: %d, data: %d\n\r", gridDim.x, blockIdx.x, blockIdx.y, tid, gid, input[gid]);
}

__global__ void idx_calc_2d_2(int* input)
{
    int tid = threadIdx.x + threadIdx.y;

    int n_thBlock = blockDim.x * blockDim.y;
    int block_offset = blockIdx.x * n_thBlock;

    int n_thRow = n_thBlock * gridDim.x;
    int row_offset = n_thRow * blockIdx.y;

    int gid = tid + row_offset + block_offset;

    printf("*[DEVICE] gridDim.x: %d, blockIdx.x: %d, blockIdx.y: %d, threadIdx.x: %d, gid: %d, data: %d\n\r", gridDim.x, blockIdx.x, blockIdx.y, tid, gid, input[gid]);
}

int main()
{
    const int array_size = 16;
    int memsize = sizeof(int) * array_size;
    int h_data[] = { 4, 5, 9, 8, 1, 2, 1, 2, 5, 5, 6, 6, 7, 8, 9, 5 };
    for (int i = 0; i > array_size; i++) {
        printf("[HOST] data: %d", h_data[i]);
    }

    int* d_data;
    dim3 threadsPerBlock(2, 2);
    dim3 blocksInGrid(2, 2);

    cudaMalloc((void**)&d_data, memsize);
    cudaMemcpy(d_data, h_data, memsize, cudaMemcpyHostToDevice);

    //idx_calc_tid << <blocksInGrid, threadsPerBlock >> > (d_data);
    //idx_calc_gid << <blocksInGrid, threadsPerBlock >> > (d_data);
    //idx_calc_2d << <blocksInGrid, threadsPerBlock >> > (d_data);
    idx_calc_2d_2 << <blocksInGrid, threadsPerBlock >> > (d_data);

    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}