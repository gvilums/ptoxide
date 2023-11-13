__global__ void gemm(float* a, float* b, float* c, size_t m, size_t k, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < n) {
        float sum = 0.0f;
        for (size_t l = 0; l < k; ++l) {
            sum += a[i * k + l] * b[l * n + j];
        }
        c[i * n + j] = sum;
    }
}
