// ext/rope_apply.cu
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

template<typename T>
__device__ inline float to_float(T x);
template<>
__device__ inline float to_float<float>(float x){ return x; }
template<>
__device__ inline float to_float<__half>(__half x){ return __half2float(x); }

template<typename T>
__device__ inline T from_float(float x);
template<>
__device__ inline float from_float<float>(float x){ return x; }
template<>
__device__ inline __half from_float<__half>(float x){ return __float2half(x); }

// Q,K: [B,H,L,D] (D even), cos,sin: [1,1,L,D]
template<typename T>
__global__ void rope_apply_qk_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ cos_t,
    const T* __restrict__ sin_t,
    T* __restrict__ q_out,
    T* __restrict__ k_out,
    int B, int H, int L, int D
){
    int bh = blockIdx.y;
    int l  = blockIdx.x;
    int d  = threadIdx.x << 1; // process 2 elements (pair) at a time
    if (l >= L || bh >= B*H) return;
    if (d + 1 >= D) return;

    // offsets
    int base = ((bh * L) + l) * D;
    int cbase = l * D; // cos/sin broadcast [1,1,L,D]

    // pair-wise rotate
    float q0 = to_float(q[base + d + 0]);
    float q1 = to_float(q[base + d + 1]);
    float k0 = to_float(k[base + d + 0]);
    float k1 = to_float(k[base + d + 1]);

    float c0 = to_float(cos_t[cbase + d + 0]);
    float c1 = to_float(cos_t[cbase + d + 1]); // equal to c0 typically; kept generic
    float s0 = to_float(sin_t[cbase + d + 0]);
    float s1 = to_float(sin_t[cbase + d + 1]);

    // rotate-half:
    // [x0, x1] -> [-x1, x0]
    float rq0 = (-q1);
    float rq1 = ( q0);
    float rk0 = (-k1);
    float rk1 = ( k0);

    float q_out0 = q0 * c0 + rq0 * s0;
    float q_out1 = q1 * c1 + rq1 * s1;
    float k_out0 = k0 * c0 + rk0 * s0;
    float k_out1 = k1 * c1 + rk1 * s1;

    q_out[base + d + 0] = from_float<T>(q_out0);
    q_out[base + d + 1] = from_float<T>(q_out1);
    k_out[base + d + 0] = from_float<T>(k_out0);
    k_out[base + d + 1] = from_float<T>(k_out1);
}

extern "C" void launch_rope_apply_qk(
    const void* q, const void* k, const void* cos_t, const void* sin_t,
    void* q_out, void* k_out,
    int B, int H, int L, int D, int dtype_code, cudaStream_t stream
){
    dim3 grid(L, B*H, 1);
    dim3 block((D/2)); // one thread per pair
    if (block.x == 0) block.x = 1;
    if (dtype_code == 0) {
        rope_apply_qk_kernel<float><<<grid, block, 0, stream>>>(
            (const float*)q, (const float*)k, (const float*)cos_t, (const float*)sin_t,
            (float*)q_out, (float*)k_out, B, H, L, D
        );
    } else {
        rope_apply_qk_kernel<__half><<<grid, block, 0, stream>>>(
            (const __half*)q, (const __half*)k, (const __half*)cos_t, (const __half*)sin_t,
            ( __half*)q_out, ( __half*)k_out, B, H, L, D
        );
    }
}
