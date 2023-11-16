#include<stdio.h>
#include<stdlib.h>

#define BLOCK_SIZE 32 

__global__ void transpose(float *input, float *output, size_t N) {

	__shared__ float sharedMemory [BLOCK_SIZE] [BLOCK_SIZE];

	// global index	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	// transposed global memory index
	int ti = threadIdx.x + blockIdx.y * blockDim.x;
	int tj = threadIdx.y + blockIdx.x * blockDim.y;

	// local index
	int local_i = threadIdx.x;
	int local_j = threadIdx.y;

	if (i < N && j < N) {
		// reading from global memory in coalesed manner and performing tanspose in shared memory
		int index = j * N + i;
		sharedMemory[local_i][local_j] = input[index];
	} else {
		sharedMemory[local_i][local_j] = 0.0;
	}

	__syncthreads();

	if (ti < N && tj < N) {
		// writing into global memory in coalesed fashion via transposed data in shared memory
		int transposedIndex = tj * N + ti;
		output[transposedIndex] = sharedMemory[local_j][local_i];
	}
}
