__global__ void copy(float* a, float* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        b[i] = a[i];
    }
}
