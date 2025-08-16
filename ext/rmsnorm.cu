// ext/rmsnorm.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename T>
__device__ inline float to_float(T x);
template <>
__device__ inline float to_float<float>(float x) { return x; }
template <>
__device__ inline float to_float<__half>(__half x) { return __half2float(x); }

template <typename T>
__device__ inline T from_float(float x);
template <>
__device__ inline float from_float<float>(float x) { return x; }
template <>
__device__ inline __half from_float<__half>(float x) { return __float2half(x); }

// y = rmsnorm(x, weight, bias, eps)
// x/y: [B, H], weight/bias: [H] (bias optional; pass nullptr to skip)
template <typename T>
__global__ void rmsnorm_kernel(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,    // can be nullptr
    T* __restrict__ y,
    int B, int H,
    float eps
) {
    // 1 block per row (batch), many threads per row
    int b = blockIdx.x;
    if (b >= B) return;

    const T* xrow = x + (size_t)b * H;
    T* yrow = y + (size_t)b * H;

    // 1) compute mean of squares in float
    float sumsq = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float xi = to_float<T>(xrow[i]);
        sumsq += xi * xi;
    }
    // reduce within block
    __shared__ float sh;
    // simple warp-then-block reduction
    // warp reduce
    float val = sumsq;
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    // write warp 0 to shared
    if ((threadIdx.x & 31) == 0) atomicAdd(&sh, val);
    __syncthreads();

    // only one thread finalizes rsigma
    float rsigma;
    if (threadIdx.x == 0) {
        float mean_sq = sh / (float)H;
        rsigma = rsqrtf(mean_sq + eps);
        sh = rsigma; // store back for broadcast
    }
    __syncthreads();
    rsigma = sh;

    // 2) normalize + affine
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float xi = to_float<T>(xrow[i]);
        float gi = weight ? to_float<T>(weight[i]) : 1.f;
        float bi = bias   ? to_float<T>(bias[i])   : 0.f;
        float yi = (xi * rsigma) * gi + bi;
        yrow[i] = from_float<T>(yi);
    }
}

// Launcher (C interface)
extern "C" void rmsnorm_forward(
    const void* x,
    const void* weight,
    const void* bias,   // can be nullptr
    void* y,
    int B, int H,
    float eps,
    int dtype_code,     // 0=float32, 1=float16
    cudaStream_t stream
) {
    dim3 grid(B);
    dim3 block(min(1024, ((H + 31) / 32) * 32)); // round to warp multiple

    if (dtype_code == 0) {
        rmsnorm_kernel<float><<<grid, block, 0, stream>>>(
            (const float*)x, (const float*)weight, (const float*)bias,
            (float*)y, B, H, eps
        );
    } else {
        rmsnorm_kernel<__half><<<grid, block, 0, stream>>>(
            (const __half*)x, (const __half*)weight, (const __half*)bias,
            ( __half*)y, B, H, eps
        );
    }
}
