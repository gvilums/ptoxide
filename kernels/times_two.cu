__global__ void times_two(float* a, float* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        b[i] = 2 * a[i];
    }
}
