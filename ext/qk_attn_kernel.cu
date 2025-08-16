#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifndef TILE_M
#define TILE_M 16
#endif
#ifndef TILE_N
#define TILE_N 16
#endif
#ifndef TILE_K
#define TILE_K 32
#endif

// C[b, m, n] = scale * sum_k A[b, m, k] * B[b, n, k]
// A: [M, K] row-major
// B: [N, K] row-major (logically K^T)
// C: [M, N] row-major
extern "C" __global__
void qk_batched_gemm_kernel(
    const void* __restrict__ A_,
    const void* __restrict__ B_,
    float* __restrict__ C,
    int BH, int M, int N, int K,
    float scale,
    int a_stride_bytes, int b_stride_bytes, int c_stride_elems,
    int dtype_code // 0=float32, 1=float16
) {
    int bh = blockIdx.z;
    if (bh >= BH) return;

    int m0 = blockIdx.y * TILE_M;
    int n0 = blockIdx.x * TILE_N;

    const char* A_base = (const char*)A_ + (size_t)bh * (size_t)a_stride_bytes;
    const char* B_base = (const char*)B_ + (size_t)bh * (size_t)b_stride_bytes;
    float* C_base = C + (size_t)bh * (size_t)c_stride_elems;

    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];  // NOTE: K x N (safer inner-product access)

    int ty = threadIdx.y; // [0, TILE_M)
    int tx = threadIdx.x; // [0, TILE_N)

    int m = m0 + ty;
    int n = n0 + tx;

    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Load A tile: rows m, cols k0..k0+TILE_K
        if (m < M) {
            for (int kk = tx; kk < TILE_K; kk += TILE_N) {
                int k = k0 + kk;
                float a = 0.0f;
                if (k < K) {
                    if (dtype_code == 0) {
                        const float* A = reinterpret_cast<const float*>(A_base);
                        a = A[m * K + k];
                    } else {
                        const __half* A = reinterpret_cast<const __half*>(A_base);
                        a = __half2float(A[m * K + k]);
                    }
                }
                As[ty][kk] = a;
            }
        } else {
            for (int kk = tx; kk < TILE_K; kk += TILE_N) As[ty][kk] = 0.0f;
        }

        // Load B tile: rows k0..k0+TILE_K, cols n
        if (n < N) {
            for (int kk = ty; kk < TILE_K; kk += TILE_M) {
                int k = k0 + kk;
                float b = 0.0f;
                if (k < K) {
                    if (dtype_code == 0) {
                        const float* B = reinterpret_cast<const float*>(B_base);
                        b = B[n * K + k];
                    } else {
                        const __half* B = reinterpret_cast<const __half*>(B_base);
                        b = __half2float(B[n * K + k]);
                    }
                }
                Bs[kk][tx] = b;  // NOTE: store as K x N
            }
        } else {
            for (int kk = ty; kk < TILE_K; kk += TILE_M) Bs[kk][tx] = 0.0f;
        }

        __syncthreads();

        if (m < M && n < N) {
            #pragma unroll
            for (int kk = 0; kk < TILE_K; ++kk) {
                acc += As[ty][kk] * Bs[kk][tx];  // inner product along K
            }
        }
        __syncthreads();
    }

    if (m < M && n < N) {
        C_base[m * N + n] = acc * scale;
    }
}

extern "C" void launch_qk_batched_gemm(
    const void* A, const void* B, float* C,
    int BH, int M, int N, int K, float scale,
    int a_stride_bytes, int b_stride_bytes, int c_stride_elems,
    int dtype_code, cudaStream_t stream
) {
    dim3 block(TILE_N, TILE_M, 1);
    dim3 grid((N + TILE_N - 1) / TILE_N,
              (M + TILE_M - 1) / TILE_M,
              BH);
    qk_batched_gemm_kernel<<<grid, block, 0, stream>>>(
        A, B, C, BH, M, N, K, scale,
        a_stride_bytes, b_stride_bytes, c_stride_elems, dtype_code
    );
}
