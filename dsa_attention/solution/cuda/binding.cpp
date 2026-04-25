#include <torch/extension.h>

extern "C" void kernel(
    void* q_nope,
    void* q_pe,
    void* ckv_cache,
    void* kpe_cache,
    void* sparse_indices,
    float sm_scale,
    void* output,
    void* lse,
    int num_tokens,
    int num_pages);

void forward(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor sparse_indices,
    double sm_scale,
    torch::Tensor output,
    torch::Tensor lse) {
    kernel(
        q_nope.data_ptr(),
        q_pe.data_ptr(),
        ckv_cache.data_ptr(),
        kpe_cache.data_ptr(),
        sparse_indices.data_ptr(),
        static_cast<float>(sm_scale),
        output.data_ptr(),
        lse.data_ptr(),
        static_cast<int>(q_nope.size(0)),
        static_cast<int>(ckv_cache.size(0)));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "DSA sparse attention forward");
}
