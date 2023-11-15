__device__ __noinline__ float add_op(float a, float b) {
    return a + b;
}


__global__ void add(float* a, float* b, float* c, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = add_op(a[i], b[i]);
    }
}
