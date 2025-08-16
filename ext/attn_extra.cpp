// ext/attn_extra.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

// AV
extern "C" void launch_av_batched_gemm(
    const void* A, const void* B, void* C,
    int BH, int M, int N, int K,
    int a_stride_bytes, int b_stride_bytes, int c_stride_bytes,
    int dtype_code, cudaStream_t stream
);

// RMSNorm
extern "C" void launch_rmsnorm(
    const void* x, const void* residual, const void* weight, void* y,
    int R, int D, float eps, int dtype_code, cudaStream_t stream
);

// RoPE
extern "C" void launch_rope_apply_qk(
    const void* q, const void* k, const void* cos_t, const void* sin_t,
    void* q_out, void* k_out,
    int B, int H, int L, int D, int dtype_code, cudaStream_t stream
);

// -------- AV (A·V) --------
at::Tensor av_forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA");
    TORCH_CHECK(A.dim()==3 && B.dim()==3, "A,B must be 3D [BH,M,K], [BH,K,N]");
    TORCH_CHECK(A.scalar_type()==at::kHalf || A.scalar_type()==at::kFloat, "A must be f16/f32");
    TORCH_CHECK(B.scalar_type()==at::kHalf || B.scalar_type()==at::kFloat, "B must be f16/f32");
    TORCH_CHECK(A.scalar_type()==B.scalar_type(), "A,B dtypes must match");

    int64_t BH=A.size(0), M=A.size(1), K=A.size(2);
    TORCH_CHECK(B.size(0)==BH && B.size(1)==K, "B dims mismatch");
    int64_t N=B.size(2);

    auto out = at::empty({BH, M, N}, A.options());
    int dtype_code = (A.scalar_type()==at::kFloat) ? 0 : 1;
    int a_stride_bytes = (int)(M*K*A.element_size());
    int b_stride_bytes = (int)(K*N*B.element_size());
    int c_stride_bytes = (int)(M*N*out.element_size());

    auto stream = at::cuda::getCurrentCUDAStream();
    launch_av_batched_gemm(
        A.data_ptr(), B.data_ptr(), out.data_ptr(),
        (int)BH, (int)M, (int)N, (int)K,
        a_stride_bytes, b_stride_bytes, c_stride_bytes,
        dtype_code, stream.stream()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// -------- RMSNorm --------
at::Tensor rmsnorm_forward(at::Tensor x, at::Tensor weight, double eps) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "x, weight must be CUDA");
    TORCH_CHECK(x.scalar_type()==at::kHalf || x.scalar_type()==at::kFloat, "x must be f16/f32");
    TORCH_CHECK(weight.scalar_type()==x.scalar_type(), "dtype mismatch");
    TORCH_CHECK(weight.dim()==1, "weight must be [D]");
    int64_t R = x.size(0) * (x.dim()==2 ? 1 : x.size(1));
    int64_t D = x.size(-1);

    auto y = at::empty_like(x);
    int dtype_code = (x.scalar_type()==at::kFloat) ? 0 : 1;
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_rmsnorm(
        x.data_ptr(), nullptr, weight.data_ptr(), y.data_ptr(),
        (int)R, (int)D, (float)eps, dtype_code, stream.stream()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

at::Tensor rmsnorm_residual_forward(at::Tensor residual, at::Tensor x, at::Tensor weight, double eps) {
    TORCH_CHECK(residual.is_cuda() && x.is_cuda() && weight.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(x.scalar_type()==at::kHalf || x.scalar_type()==at::kFloat, "x must be f16/f32");
    TORCH_CHECK(residual.scalar_type()==x.scalar_type() && weight.scalar_type()==x.scalar_type(), "dtype mismatch");
    TORCH_CHECK(weight.dim()==1, "weight must be [D]");
    int64_t R = x.size(0) * (x.dim()==2 ? 1 : x.size(1));
    int64_t D = x.size(-1);

    auto y = at::empty_like(x);
    int dtype_code = (x.scalar_type()==at::kFloat) ? 0 : 1;
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_rmsnorm(
        x.data_ptr(), residual.data_ptr(), weight.data_ptr(), y.data_ptr(),
        (int)R, (int)D, (float)eps, dtype_code, stream.stream()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// -------- RoPE --------
std::tuple<at::Tensor,at::Tensor> rope_apply_qk(at::Tensor q, at::Tensor k, at::Tensor cos_t, at::Tensor sin_t) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && cos_t.is_cuda() && sin_t.is_cuda(), "all must be CUDA");
    TORCH_CHECK(q.dim()==4 && k.dim()==4, "q,k must be [B,H,L,D]");
    TORCH_CHECK(cos_t.dim()==4 && sin_t.dim()==4, "cos,sin must be [1,1,L,D]");
    TORCH_CHECK(q.scalar_type()==at::kHalf || q.scalar_type()==at::kFloat, "dtype must be f16/f32");
    TORCH_CHECK(q.scalar_type()==k.scalar_type() && q.scalar_type()==cos_t.scalar_type() && q.scalar_type()==sin_t.scalar_type(),
        "dtype mismatch");

    int64_t B=q.size(0), H=q.size(1), L=q.size(2), D=q.size(3);
    TORCH_CHECK(k.size(0)==B && k.size(1)==H && k.size(2)==L && k.size(3)==D, "k dims mismatch");
    TORCH_CHECK(cos_t.size(0)==1 && cos_t.size(1)==1 && cos_t.size(2)==L && cos_t.size(3)==D, "cos shape mismatch");

    auto q_out = at::empty_like(q);
    auto k_out = at::empty_like(k);
    int dtype_code = (q.scalar_type()==at::kFloat) ? 0 : 1;

    auto stream = at::cuda::getCurrentCUDAStream();
    launch_rope_apply_qk(
        q.data_ptr(), k.data_ptr(), cos_t.data_ptr(), sin_t.data_ptr(),
        q_out.data_ptr(), k_out.data_ptr(),
        (int)B, (int)H, (int)L, (int)D, dtype_code, stream.stream()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {q_out, k_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("av_forward", &av_forward, "AV batched GEMM");
    m.def("rmsnorm_forward", &rmsnorm_forward, "RMSNorm");
    m.def("rmsnorm_residual_forward", &rmsnorm_residual_forward, "RMSNorm with residual add");
    m.def("rope_apply_qk", &rope_apply_qk, "RoPE apply to Q and K");
}
