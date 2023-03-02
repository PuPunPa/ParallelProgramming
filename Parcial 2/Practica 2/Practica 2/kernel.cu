
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <random>

using namespace std;

#define TILE_DIM 32

template<typename T>

class cuda_ptr
{
public:
    cuda_ptr(const T* host_ptr, const int size) {
        CUDA_ERROR_HANDLER(cudaMalloc(&ptr, size));
        CUDA_ERROR_HANDLER(cudaMemcpy(ptr, host_ptr, size, cudaMemcpyHostToDevice));
    }

    cuda_ptr(const int size) {
        CUDA_ERROR_HANDLER(cudaMalloc(&ptr, size));
    }

    ~cuda_ptr() {
        CUDA_IPC_HANDLER(cudaFree(ptr));
    }

    void copy(const T* host_ptr, const int size) {
        CUDA_ERROR_HANDLER(cudaMemcpy(ptr, host_ptr, size, cudaMemcpyHostToDevice));
    }

    void to_host(T* host_ptr, const int size) {
        CUDA_ERROR_HANDLER(cudaMemcpy(host_ptr, ptr, size, cudaMemcpyDefault));
    }

    T* devptr() {
        return ptr;
    }

    T operator[](int i) {
        return ptr[i];
    }

private:
    T* ptr;
};

__global__ void transpose_shared(double* source, double* dest, int size) {
	__shared__ double tile[TILE_DIM][TILE_DIM];
	int i_in = threadIdx.x + blockDim.x * blockIdx.x;
	int j_in = threadIdx.y + blockDim.y * blockIdx.y;

	int src_idx = j_in * size + i_in;

	int _id_index = threadIdx.y * blockDim.x + threadIdx.x;

	int i_row = _id_index / blockDim.y;
	int i_col = _id_index % blockDim.y;

	int i_out = blockIdx.y * blockDim.y + threadIdx.x;
	int j_out = blockIdx.x * blockDim.y + threadIdx.y;

	int dst_idx = j_out * size + i_out;

	if (i_in < size && j_in < size) {
		tile[threadIdx.y][threadIdx.x] = source[src_idx];

		__syncthreads();

		dest[dst_idx] = tile[threadIdx.x][threadIdx.y];
	}
}

int main() {
    int mat_size = 12;
    int byte_size = mat_size * mat_size * sizeof(double);

    double* mat_input = (double*)malloc(byte_size);
    double* mat_output = (double*)malloc(byte_size);
    memset(mat_output, 0, byte_size);

    srand((unsigned)time(NULL));
    for (int i = 0; i < mat_size * mat_size; i++) {
        mat_input[i] = (double)(rand() % 10);
    }

    int block_size = TILE_DIM;
    int grid_size = (int)ceil((float)mat_size / block_size);
    dim3 grid(grid_size, grid_size);
    dim3 block(block_size, block_size);

    double* a = (double*)malloc(byte_size);
    double* b = (double*)malloc(byte_size);

    // Assign random value between 0-255 to every position
    srand((unsigned)time(NULL));
    for (int i = 0; i < mat_size*mat_size; i++) {
        a[i] = (double)(rand() % 226);
    }

    double* devA = 0;
    double* devB = 0;

    // Allocate Memory
    cudaMalloc((void**)&devA, byte_size);
    cudaMalloc((void**)&devB, byte_size);

    // Copy to GPU
    cudaMemcpy(devA, a, byte_size, cudaMemcpyHostToDevice);

    transpose_shared << <grid, block >> > (devA, devB, mat_size*mat_size);

    cudaMemcpy(b, devB, byte_size, cudaMemcpyDeviceToHost);

    cout << "A { " << endl;
    for (int i = 0; i < mat_size; i++) {
        for (int j = 0; j < mat_size; j++) {
            cout << a[i * mat_size + j] << ", ";
        }
        cout << endl;
    }

    cout << "} \nB { " << endl;
    for (int i = 0; i < mat_size; i++) {
        for (int j = 0; j < mat_size; j++) {
            cout << b[i * mat_size + j] << ", ";
        }
        cout << endl;
    }
    cout << "}";

    // Clean
    cudaDeviceSynchronize();
    cudaDeviceReset();

    cudaFree(devA);
    cudaFree(devB);

    return 0;
}