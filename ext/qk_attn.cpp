#include <torch/extension.h>
#include <cuda_runtime_api.h>
#include <ATen/cuda/CUDAContext.h>
#if defined(__has_include)
  #if __has_include(<c10/cuda/CUDAException.h>)
    #include <c10/cuda/CUDAException.h>
  #elif __has_include(<ATen/cuda/CUDAException.h>)
    #include <ATen/cuda/CUDAException.h>
  #endif
#endif

extern "C" void launch_qk_batched_gemm(
    const void* A, const void* B, float* C,
    int BH, int M, int N, int K,
    float scale,
    int a_stride_bytes, int b_stride_bytes, int c_stride_elems,
    int dtype_code,
    cudaStream_t stream
);

// Q: [BH, M, K], K: [BH, N, K] → C: [BH, M, N] (float32)
at::Tensor qk_forward(at::Tensor Q, at::Tensor K, double scale) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda(), "Q and K must be CUDA tensors");
    TORCH_CHECK(Q.dim() == 3 && K.dim() == 3, "Q and K must be 3D");
    TORCH_CHECK(Q.scalar_type() == at::kHalf || Q.scalar_type() == at::kFloat, "Q must be fp16/fp32");
    TORCH_CHECK(K.scalar_type() == at::kHalf || K.scalar_type() == at::kFloat, "K must be fp16/fp32");
    TORCH_CHECK(Q.size(0) == K.size(0) && Q.size(2) == K.size(2), "Mismatched BH or head_dim");

    const int64_t BH = Q.size(0);
    const int64_t M  = Q.size(1);
    const int64_t KK = Q.size(2);
    const int64_t N  = K.size(1);

    auto C = at::empty({BH, M, N}, Q.options().dtype(at::kFloat));

    const int dtype_code = (Q.scalar_type() == at::kFloat) ? 0 : 1;
    const int a_stride_bytes = (int)(M * KK * Q.element_size());
    const int b_stride_bytes = (int)(N * KK * K.element_size());
    const int c_stride_elems = (int)(M * N);

    auto stream = at::cuda::getCurrentCUDAStream();
    launch_qk_batched_gemm(
        Q.data_ptr(), K.data_ptr(), C.data_ptr<float>(),
        (int)BH, (int)M, (int)N, (int)KK,
        (float)scale,
        a_stride_bytes, b_stride_bytes, c_stride_elems,
        dtype_code,
        stream.stream()
    );

#if defined(C10_CUDA_KERNEL_LAUNCH_CHECK)
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qk_forward", &qk_forward, "QK^T attention scores (batched)");
}
