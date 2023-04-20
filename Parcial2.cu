#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <stdio.h>
#include <iostream>
#include <time.h>

void bubble_sortCPU(int* a, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (a[j] > a[j + 1]) {
                int aux = a[j + 1];
                a[j + 1] = a[j];
                a[j] = aux;
            }
        }
    }
}
__global__ void bubble_sortGPU(int* a, int n) {

    int tid = threadIdx.x;

    for (int i = 0; i < n; i++) {

        int offset = i % 2;
        int leftInd = 2 * tid + offset;
        int rightInd = leftInd + 1;

        if (rightInd < n) {
            if (a[leftInd] > a[rightInd]) {
                int aux = a[leftInd];
                a[leftInd] = a[rightInd];
                a[rightInd] = aux;
            }
        }
        __syncthreads();
    }
}

void medianCPU(int* a, int* b, int n){
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += a[i];
    }
    *b = sum / n;
}

__global__ void medianGPU(int* a, int* b, int n){
    __shared__ int shareMem[1024];
    int tid = threadIdx.x;
    int gid = tid + blockIdx.x * blockDim.x;
    int sum = 0;
    for (int i = 0; i < 64; i++) {
        shareMem[i] = a[gid];
        __syncthreads();
    }
    //*b = sum / n;
}

void standard_devCPU(int* a, int* b, int n){
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += ((a[i] - *b) * (a[i] - *b));
    }
    *b = sqrt(sum / (n-1));
}

__global__ void standard_devGPU(int* a, int n) {}

int main() {

    int size = 10;
    int* host_a;
    int* dev_a;
    int* host_b;
    int* dev_b;

    host_a = (int*)malloc(size * sizeof(int));
    host_b = (int*)malloc(sizeof(int));

    cudaMalloc(&dev_a, size * sizeof(size));
    cudaMalloc(&dev_b, sizeof(int));

    srand(time(NULL));

    for (int i = 0; i < size; i++) {
        int r1 = (rand() % (256));
        host_a[i] = r1;
        printf("%d ", host_a[i]);
    }

    printf("\n");

    cudaMemcpy(dev_a, host_a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid(1);
    dim3 block(size);

    //bubble_sortGPU << <grid, block >> > (dev_a, size);
    //cudaMemcpy(res, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost);
    medianCPU(host_a, host_b, size);

    printf("CPU: \n");
    printf("%d\n", *host_b);

    for (int i = 0; i < size; i++) {
        //printf("%d ", host_a[i]);
    }

    standard_devCPU(host_a, host_b, size);
    printf("%d\n", *host_b);
    printf("\n");

    medianGPU << <grid, block >> > (dev_a, dev_b, size);
    printf("GPU\n");
    printf("%d\n", *dev_b);

    for (int i = 0; i < size; i++) {
        //printf("%d ", res[i]);
    }

    printf("\n");

    return 0;

}
