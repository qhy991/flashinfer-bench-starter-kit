/**
 * DSA Sparse Attention Kernel (bfloat16) — v8 (Blackwell-tuned)
 *
 * Definition: dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64
 *
 * Optimizations over v3:
 * 1. Vectorized V loads in Pass 2 using __nv_bfloat162 for 2x wider loads,
 *    reducing the number of memory transactions.
 * 2. Prefix-valid sparse indices: real workloads store valid sparse indices
 *    as a contiguous prefix followed by -1 padding. We find n_valid once and
 *    only process [0, n_valid), avoiding the old compact-slot build.
 * 3. Small-prefix path: for n_valid <= 128, compute each logit with a full
 *    warp instead of one scalar thread. The path is noinline to avoid
 *    increasing register pressure on the large-prefix fallback.
 */

#include <cuda_bf16.h>
#include <cstdlib>
#include <math.h>

#define NUM_HEADS      16
#define HEAD_DIM_CKV  512
#define HEAD_DIM_KPE   64
#define PAGE_SIZE      64
#define TOPK         2048

#define BLOCK_SIZE    256
#define WARPS          8
#define WARP_SIZE     32
#define VECT           4
#define SMALL_VALID_THRESHOLD 128
#define SMALL_PATH_MAX_TOKENS 8

static int get_env_int_clamped(const char* name, int default_value, int min_value, int max_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return default_value;

    char* end_ptr = nullptr;
    long parsed = std::strtol(value, &end_ptr, 10);
    if (end_ptr == value) return default_value;
    if (parsed < min_value) return min_value;
    if (parsed > max_value) return max_value;
    return static_cast<int>(parsed);
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

__device__ __noinline__ void dsa_small_prefix_path(
    const __nv_bfloat16* __restrict__ q_nope_ptr,
    const __nv_bfloat16* __restrict__ q_pe_ptr,
    const __nv_bfloat16* __restrict__ ckv_cache,
    const __nv_bfloat16* __restrict__ kpe_cache,
    const int*           __restrict__ idx_ptr,
    const float                       sm_scale,
    __nv_bfloat16*       __restrict__ output,
    float*               __restrict__ lse,
    const int                         token,
    const int                         head,
    const int                         tid,
    const int                         warp_id,
    const int                         lane,
    const int                         n_valid,
    float*                             s_logits,
    float*                             s_wmax,
    float*                             s_wsum)
{
    float warp_local_max = -INFINITY;

    for (int i = warp_id; i < n_valid; i += WARPS) {
        int global_idx = idx_ptr[i];
        int page = global_idx / PAGE_SIZE;
        int off = global_idx % PAGE_SIZE;
        const __nv_bfloat16* k_ckv = ckv_cache + ((long long)page * PAGE_SIZE + off) * HEAD_DIM_CKV;
        const __nv_bfloat16* k_kpe = kpe_cache + ((long long)page * PAGE_SIZE + off) * HEAD_DIM_KPE;

        float logit = 0.0f;
        const __nv_bfloat162* qn2 = (const __nv_bfloat162*)q_nope_ptr;
        const __nv_bfloat162* kc2 = (const __nv_bfloat162*)k_ckv;
        #pragma unroll
        for (int p = lane; p < HEAD_DIM_CKV / 2; p += WARP_SIZE) {
            float2 qf = __bfloat1622float2(qn2[p]);
            float2 kf = __bfloat1622float2(kc2[p]);
            logit += qf.x * kf.x + qf.y * kf.y;
        }

        const __nv_bfloat162* qp2 = (const __nv_bfloat162*)q_pe_ptr;
        const __nv_bfloat162* kp2 = (const __nv_bfloat162*)k_kpe;
        if (lane < HEAD_DIM_KPE / 2) {
            float2 qf = __bfloat1622float2(qp2[lane]);
            float2 kf = __bfloat1622float2(kp2[lane]);
            logit += qf.x * kf.x + qf.y * kf.y;
        }

        logit = warp_reduce_sum(logit);
        if (lane == 0) {
            s_logits[i] = logit;
            warp_local_max = fmaxf(warp_local_max, logit);
        }
    }

    if (lane == 0) s_wmax[warp_id] = warp_local_max;
    __syncthreads();

    float global_max = -INFINITY;
    if (tid == 0) {
        for (int w = 0; w < WARPS; w++)
            global_max = fmaxf(global_max, s_wmax[w]);
        s_wmax[0] = global_max;
    }
    __syncthreads();
    global_max = s_wmax[0];

    float local_sum = 0.0f;
    for (int i = tid; i < n_valid; i += BLOCK_SIZE) {
        float w = expf(s_logits[i] * sm_scale - global_max * sm_scale);
        s_logits[i] = w;
        local_sum += w;
    }

    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) s_wsum[warp_id] = local_sum;
    __syncthreads();

    float total_sum = 0.0f;
    if (tid == 0) {
        for (int w = 0; w < WARPS; w++)
            total_sum += s_wsum[w];
        s_wsum[0] = total_sum;
    }
    __syncthreads();
    total_sum = s_wsum[0];

    if (tid == 0) {
        float lse_val = (global_max * sm_scale + logf(total_sum + 1e-10f)) / logf(2.0f);
        lse[token * NUM_HEADS + head] = lse_val;
    }

    float inv_sum = (total_sum > 0.0f) ? (1.0f / total_sum) : 0.0f;

    const int num_quads = HEAD_DIM_CKV / VECT;
    const int base_d = tid * VECT;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    if (tid < num_quads) {
        for (int i = 0; i < n_valid; ++i) {
            float w = s_logits[i];
            int global_idx = idx_ptr[i];
            int page = global_idx / PAGE_SIZE;
            int off = global_idx % PAGE_SIZE;
            const __nv_bfloat16* v_ptr = ckv_cache + ((long long)page * PAGE_SIZE + off) * HEAD_DIM_CKV;

            const __nv_bfloat162* v2_0 = (const __nv_bfloat162*)(v_ptr + base_d);
            const __nv_bfloat162* v2_1 = (const __nv_bfloat162*)(v_ptr + base_d + 2);
            float2 vf0 = __bfloat1622float2(*v2_0);
            float2 vf1 = __bfloat1622float2(*v2_1);
            acc0 += w * vf0.x;
            acc1 += w * vf0.y;
            acc2 += w * vf1.x;
            acc3 += w * vf1.y;
        }
    }

    __nv_bfloat16* out_ptr = output + ((long long)token * NUM_HEADS + head) * HEAD_DIM_CKV;
    if (tid < num_quads) {
        out_ptr[base_d + 0] = __float2bfloat16_rn(acc0 * inv_sum);
        out_ptr[base_d + 1] = __float2bfloat16_rn(acc1 * inv_sum);
        out_ptr[base_d + 2] = __float2bfloat16_rn(acc2 * inv_sum);
        out_ptr[base_d + 3] = __float2bfloat16_rn(acc3 * inv_sum);
    }
}

template <bool ENABLE_SMALL_PATH>
__global__ void dsa_sparse_attention_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_cache,
    const __nv_bfloat16* __restrict__ kpe_cache,
    const int*           __restrict__ sparse_idx,
    const float                       sm_scale,
    __nv_bfloat16*       __restrict__ output,
    float*               __restrict__ lse,
    const int                         small_valid_threshold,
    const int                         num_tokens,
    const int                         num_pages)
{
    const int token = blockIdx.x;
    const int head  = blockIdx.y;
    if (token >= num_tokens) return;

    (void)num_pages;

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane    = tid % WARP_SIZE;

    extern __shared__ char smem_raw[];
    float* s_logits  = reinterpret_cast<float*>(smem_raw);
    float* s_wmax    = s_logits + TOPK;
    float* s_wsum    = s_wmax + WARPS;

    __shared__ int s_n_valid;

    const __nv_bfloat16* q_nope_ptr = q_nope + ((long long)token * NUM_HEADS + head) * HEAD_DIM_CKV;
    const __nv_bfloat16* q_pe_ptr   = q_pe   + ((long long)token * NUM_HEADS + head) * HEAD_DIM_KPE;
    const int* idx_ptr = sparse_idx + (long long)token * TOPK;

    // Real sparse rows are a valid prefix followed by -1 padding.
    // Binary search avoids scanning/atomicMin over thousands of padding slots.
    if (tid == 0) {
        int lo = 0;
        int hi = TOPK;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (idx_ptr[mid] >= 0) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        s_n_valid = lo;
    }
    __syncthreads();
    const int n_valid = s_n_valid;

    if (n_valid == 0) {
        __nv_bfloat16* out_ptr = output + ((long long)token * NUM_HEADS + head) * HEAD_DIM_CKV;
        for (int d = tid; d < HEAD_DIM_CKV; d += BLOCK_SIZE) {
            out_ptr[d] = __float2bfloat16_rn(0.0f);
        }
        if (tid == 0) lse[token * NUM_HEADS + head] = -INFINITY;
        return;
    }

    if constexpr (ENABLE_SMALL_PATH) {
    if (n_valid <= small_valid_threshold) {
        dsa_small_prefix_path(
            q_nope_ptr, q_pe_ptr, ckv_cache, kpe_cache, idx_ptr, sm_scale,
            output, lse, token, head, tid, warp_id, lane, n_valid,
            s_logits, s_wmax, s_wsum);
        return;
    }
    }

    // Pass 1: logits for the valid prefix only.
    float local_max = -INFINITY;

    for (int i = tid; i < n_valid; i += BLOCK_SIZE) {
        int global_idx = idx_ptr[i];
        float logit = 0.0f;

        int page  = global_idx / PAGE_SIZE;
        int off   = global_idx % PAGE_SIZE;

        const __nv_bfloat16* k_ckv = ckv_cache + ((long long)page * PAGE_SIZE + off) * HEAD_DIM_CKV;
        #pragma unroll 1
        for (int d = 0; d < HEAD_DIM_CKV; d += 8) {
            const __nv_bfloat162* q2 = (const __nv_bfloat162*)(q_nope_ptr + d);
            const __nv_bfloat162* k2 = (const __nv_bfloat162*)(k_ckv + d);
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                float2 qq = __bfloat1622float2(q2[u]);
                float2 kk = __bfloat1622float2(k2[u]);
                logit += qq.x * kk.x + qq.y * kk.y;
            }
        }

        const __nv_bfloat16* k_kpe = kpe_cache + ((long long)page * PAGE_SIZE + off) * HEAD_DIM_KPE;
        #pragma unroll 1
        for (int d = 0; d < HEAD_DIM_KPE; d += 8) {
            const __nv_bfloat162* q2 = (const __nv_bfloat162*)(q_pe_ptr + d);
            const __nv_bfloat162* k2 = (const __nv_bfloat162*)(k_kpe + d);
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                float2 qq = __bfloat1622float2(q2[u]);
                float2 kk = __bfloat1622float2(k2[u]);
                logit += qq.x * kk.x + qq.y * kk.y;
            }
        }

        s_logits[i] = logit;
        local_max = fmaxf(local_max, logit);
    }

    local_max = warp_reduce_max(local_max);
    if (lane == 0) s_wmax[warp_id] = local_max;
    __syncthreads();

    float global_max = -INFINITY;
    if (tid == 0) {
        for (int w = 0; w < WARPS; w++)
            global_max = fmaxf(global_max, s_wmax[w]);
        s_wmax[0] = global_max;
    }
    __syncthreads();
    global_max = s_wmax[0];

    float local_sum = 0.0f;
    for (int i = tid; i < n_valid; i += BLOCK_SIZE) {
        float w = expf(s_logits[i] * sm_scale - global_max * sm_scale);
        s_logits[i] = w;
        local_sum += w;
    }

    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) s_wsum[warp_id] = local_sum;
    __syncthreads();

    float total_sum = 0.0f;
    if (tid == 0) {
        for (int w = 0; w < WARPS; w++)
            total_sum += s_wsum[w];
        s_wsum[0] = total_sum;
    }
    __syncthreads();
    total_sum = s_wsum[0];

    if (tid == 0) {
        float lse_val = (global_max * sm_scale + logf(total_sum + 1e-10f)) / logf(2.0f);
        lse[token * NUM_HEADS + head] = lse_val;
    }

    float inv_sum = (total_sum > 0.0f) ? (1.0f / total_sum) : 0.0f;

    // Pass 2: weighted V accumulation with 4 output dimensions per thread.
    const int num_quads = HEAD_DIM_CKV / VECT;
    const int base_d = tid * VECT;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    if (tid < num_quads) {
        for (int i = 0; i < n_valid; ++i) {
            float w = s_logits[i];
            int global_idx = idx_ptr[i];
            int page  = global_idx / PAGE_SIZE;
            int off   = global_idx % PAGE_SIZE;
            const __nv_bfloat16* v_ptr = ckv_cache + ((long long)page * PAGE_SIZE + off) * HEAD_DIM_CKV;

            const __nv_bfloat162* v2_0 = (const __nv_bfloat162*)(v_ptr + base_d);
            const __nv_bfloat162* v2_1 = (const __nv_bfloat162*)(v_ptr + base_d + 2);
            float2 vf0 = __bfloat1622float2(*v2_0);
            float2 vf1 = __bfloat1622float2(*v2_1);
            acc0 += w * vf0.x;
            acc1 += w * vf0.y;
            acc2 += w * vf1.x;
            acc3 += w * vf1.y;
        }
    }

    __nv_bfloat16* out_ptr = output + ((long long)token * NUM_HEADS + head) * HEAD_DIM_CKV;
    if (tid < num_quads) {
        out_ptr[base_d + 0] = __float2bfloat16_rn(acc0 * inv_sum);
        out_ptr[base_d + 1] = __float2bfloat16_rn(acc1 * inv_sum);
        out_ptr[base_d + 2] = __float2bfloat16_rn(acc2 * inv_sum);
        out_ptr[base_d + 3] = __float2bfloat16_rn(acc3 * inv_sum);
    }
}

// -----------------------------------------------------------------
// Host entry point (PyTorch-compatible C interface)
// -----------------------------------------------------------------
extern "C" void kernel(
    void* q_nope,       // [T, H, 512] bfloat16
    void* q_pe,         // [T, H, 64] bfloat16
    void* ckv_cache,    // [P, 64, 512] bfloat16
    void* kpe_cache,    // [P, 64, 64] bfloat16
    void* sparse_indices, // [T, 2048] int32
    float sm_scale,
    void* output,       // [T, H, 512] bfloat16
    void* lse,          // [T, H] float32
    int num_tokens,
    int num_pages)
{
    dim3 grid(num_tokens, NUM_HEADS);
    dim3 block(BLOCK_SIZE);
    const int small_path_max_tokens = get_env_int_clamped(
        "DSA_ATTN_V8_SMALL_PATH_MAX_TOKENS", SMALL_PATH_MAX_TOKENS, 0, TOPK
    );
    const int small_valid_threshold = get_env_int_clamped(
        "DSA_ATTN_V8_SMALL_VALID_THRESHOLD", SMALL_VALID_THRESHOLD, 0, TOPK
    );

    size_t smem = TOPK * sizeof(float)      // logits / weights
                + WARPS * sizeof(float)     // warp max
                + WARPS * sizeof(float);    // warp sum

    if (small_valid_threshold > 0 && small_path_max_tokens > 0 && num_tokens <= small_path_max_tokens) {
        dsa_sparse_attention_kernel<true><<<grid, block, smem>>>(
            static_cast<__nv_bfloat16*>(q_nope),
            static_cast<__nv_bfloat16*>(q_pe),
            static_cast<__nv_bfloat16*>(ckv_cache),
            static_cast<__nv_bfloat16*>(kpe_cache),
            static_cast<int*>(sparse_indices),
            sm_scale,
            static_cast<__nv_bfloat16*>(output),
            static_cast<float*>(lse),
            small_valid_threshold,
            num_tokens,
            num_pages);
    } else {
        dsa_sparse_attention_kernel<false><<<grid, block, smem>>>(
            static_cast<__nv_bfloat16*>(q_nope),
            static_cast<__nv_bfloat16*>(q_pe),
            static_cast<__nv_bfloat16*>(ckv_cache),
            static_cast<__nv_bfloat16*>(kpe_cache),
            static_cast<int*>(sparse_indices),
            sm_scale,
            static_cast<__nv_bfloat16*>(output),
            static_cast<float*>(lse),
            small_valid_threshold,
            num_tokens,
            num_pages);
    }
}
