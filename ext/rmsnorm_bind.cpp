// ext/rmsnorm_bind.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

extern "C" void rmsnorm_forward(
    const void* x, const void* w, const void* b, void* y,
    int B, int H, float eps, int dtype_code, cudaStream_t stream
);

at::Tensor rmsnorm_forward_bind(at::Tensor x, at::Tensor w, c10::optional<at::Tensor> b_opt, double eps) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "x & weight must be CUDA");
    TORCH_CHECK(x.dim()==2 && w.dim()==1, "x[B,H], weight[H]");
    TORCH_CHECK(x.size(1)==w.size(0), "H mismatch");

    at::Tensor b = b_opt.has_value() ? b_opt.value() : at::Tensor();
    if (b.defined()) {
        TORCH_CHECK(b.is_cuda() && b.dim()==1 && b.size(0)==w.size(0), "bias[H] CUDA");
    }

    int B = (int)x.size(0), H = (int)x.size(1);
    auto y = at::empty_like(x);

    int code = (x.scalar_type()==at::kHalf) ? 1 : 0;
    auto stream = at::cuda::getCurrentCUDAStream();

    rmsnorm_forward(
        x.data_ptr(), w.data_ptr(),
        b.defined()? b.data_ptr() : nullptr,
        y.data_ptr(),
        B, H, (float)eps, code, stream.stream()
    );
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_forward", &rmsnorm_forward_bind, "RMSNorm forward");
}
