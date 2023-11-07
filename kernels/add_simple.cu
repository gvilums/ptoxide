__global__ void add_simple(float* a, float* b, float* c) {
    size_t i = threadIdx.x;
    c[i] = a[i] + b[i];
}
