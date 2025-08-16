// ext/av_attn.cu
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifndef TILE_M
#define TILE_M 16  // rows of C per block  (Lq tile)
#endif
#ifndef TILE_N
#define TILE_N 16  // cols of C per block  (D  tile)
#endif
#ifndef TILE_K
#define TILE_K 32  // reduction dim per iter (Lk tile)
#endif

// A: [BH, M=Lq, K=Lk]  (float16/float32)
// B: [BH, K=Lk, N=D]   (float16/float32)
// C: [BH, M=Lq, N=D]   (float16/float32 out, normally match B's dtype)
extern "C" __global__
void av_batched_gemm_kernel(
    const void* __restrict__ A_,
    const void* __restrict__ B_,
    void* __restrict__ C_,
    int BH, int M, int N, int K,
    int a_stride_bytes, int b_stride_bytes, int c_stride_bytes,
    int dtype_code // 0=float32, 1=float16
) {
    int bh = blockIdx.z;
    if (bh >= BH) return;

    int m0 = blockIdx.y * TILE_M;
    int n0 = blockIdx.x * TILE_N;

    const char* A_batch = (const char*)A_ + (size_t)bh * (size_t)a_stride_bytes;
    const char* B_batch = (const char*)B_ + (size_t)bh * (size_t)b_stride_bytes;
    char*       C_batch = (char*)C_ + (size_t)bh * (size_t)c_stride_bytes;

    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    int ty = threadIdx.y; // [0, TILE_M)
    int tx = threadIdx.x; // [0, TILE_N)

    int m = m0 + ty;
    int n = n0 + tx;

    float acc = 0.f;

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // load A tile: [M, K]
        if (m < M) {
            for (int kk = tx; kk < TILE_K; kk += TILE_N) {
                int k = k0 + kk;
                float a = 0.f;
                if (k < K) {
                    if (dtype_code == 0) {
                        const float* A = reinterpret_cast<const float*>(A_batch);
                        a = A[m * K + k];
                    } else {
                        const __half* A = reinterpret_cast<const __half*>(A_batch);
                        a = __half2float(A[m * K + k]);
                    }
                }
                As[ty][kk] = a;
            }
        } else {
            for (int kk = tx; kk < TILE_K; kk += TILE_N) As[ty][kk] = 0.f;
        }

        // load B tile: [K, N]
        if (n < N) {
            for (int kk = ty; kk < TILE_K; kk += TILE_M) {
                int k = k0 + kk;
                float b = 0.f;
                if (k < K) {
                    if (dtype_code == 0) {
                        const float* B = reinterpret_cast<const float*>(B_batch);
                        b = B[k * N + n];
                    } else {
                        const __half* B = reinterpret_cast<const __half*>(B_batch);
                        b = __half2float(B[k * N + n]);
                    }
                }
                Bs[kk][tx] = b;
            }
        } else {
            for (int kk = ty; kk < TILE_K; kk += TILE_M) Bs[kk][tx] = 0.f;
        }

        __syncthreads();

        if (m < M && n < N) {
            #pragma unroll
            for (int kk = 0; kk < TILE_K; ++kk) {
                acc += As[ty][kk] * Bs[kk][tx];
            }
        }
        __syncthreads();
    }

    if (m < M && n < N) {
        if (dtype_code == 0) {
            float* C = reinterpret_cast<float*>(C_batch);
            C[m * N + n] = acc;
        } else {
            __half* C = reinterpret_cast<__half*>(C_batch);
            C[m * N + n] = __float2half(acc);
        }
    }
}

extern "C" void launch_av_batched_gemm(
    const void* A, const void* B, void* C,
    int BH, int M, int N, int K,
    int a_stride_bytes, int b_stride_bytes, int c_stride_bytes,
    int dtype_code, cudaStream_t stream
) {
    dim3 block(TILE_N, TILE_M, 1);
    dim3 grid((N + TILE_N - 1)/TILE_N,
              (M + TILE_M - 1)/TILE_M,
              BH);
    av_batched_gemm_kernel<<<grid, block, 0, stream>>>(
        A, B, C, BH, M, N, K,
        a_stride_bytes, b_stride_bytes, c_stride_bytes,
        dtype_code
    );
}
