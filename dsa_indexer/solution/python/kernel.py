"""
DSA TopK Indexer Kernel (v33 — Hybrid Mixed-Batch Dispatch)

Definition: dsa_topk_indexer_fp8_h64_d128_topk2048_ps64

Key optimizations:
1. Keep v31's grouped mixed-batch path for the large-batch region where it wins.
2. Switch to a lighter per-batch sliced path only for tiny multi-batch workloads.
3. Retain direct slicing and vectorized token-fill.
"""

import torch
import os


NUM_INDEX_HEADS = 64
INDEX_HEAD_DIM = 128
PAGE_SIZE = 64
TOPK = 2048
HEAD_DIM_WITH_SCALE = 132

_col_idx_cache: dict = {}
_offset_cache: dict = {}
DEDUP_THRESHOLD = 128
SMALL_MULTI_BATCH_SIZE = 8
SMALL_MULTI_TOTAL_N = 16


def _get_env_int(name, default, min_value, max_value):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(min_value, min(max_value, parsed))


def dequant_fp8_kv_cache(k_index_cache_fp8):
    """Dequantize full FP8 KV cache -> float32 (test-harness helper)."""
    num_pages, page_size, _, head_dim_sf = k_index_cache_fp8.shape
    head_dim = head_dim_sf - 4
    pu8 = k_index_cache_fp8.view(torch.uint8).reshape(num_pages, page_size * head_dim_sf)
    fp8f = (
        pu8[:, : page_size * head_dim]
        .contiguous()
        .view(num_pages, page_size, head_dim)
        .view(torch.float8_e4m3fn)
        .to(torch.float32)
    )
    sc = (
        pu8[:, page_size * head_dim :]
        .contiguous()
        .view(num_pages, page_size, 4)
        .view(torch.float32)
    )
    return fp8f * sc


def _dequant_gathered_flat(k_index_cache_fp8, flat_indices, page_size, head_dim, head_dim_sf):
    """Gather pages, dequantize to FP32, return flat [n*PS, D]."""
    n = flat_indices.shape[0]
    if n == 0:
        return torch.empty(0, head_dim, dtype=torch.float32, device=k_index_cache_fp8.device)

    pages = k_index_cache_fp8[flat_indices]
    pu8 = pages.view(torch.uint8).reshape(n, page_size * head_dim_sf)
    data_cols = page_size * head_dim

    fp8f = (
        pu8[:, :data_cols]
        .contiguous()
        .view(n, page_size, head_dim)
        .view(torch.float8_e4m3fn)
        .to(torch.float32)
    )
    sc = (
        pu8[:, data_cols:]
        .contiguous()
        .view(n, page_size, 4)
        .view(torch.float32)
    )
    return (fp8f * sc).reshape(n * page_size, head_dim)


def _compute_flat_indices(seq_lens, block_table, num_pages, max_num_pages, device):
    cache_key = (max_num_pages, device)
    col_idx = _col_idx_cache.get(cache_key)
    if col_idx is None:
        col_idx = torch.arange(max_num_pages, device=device, dtype=torch.long)
        _col_idx_cache[cache_key] = col_idx

    n_per_batch_gpu = (seq_lens.to(torch.long) + (PAGE_SIZE - 1)) // PAGE_SIZE
    mask = col_idx.unsqueeze(0) < n_per_batch_gpu.unsqueeze(1)
    flat_indices_int = block_table[mask]
    return flat_indices_int.to(torch.long)


def _gather_and_dequant(k_index_cache_fp8, flat_indices, page_size, head_dim, head_dim_sf, total_n, dedup_threshold):
    if total_n >= dedup_threshold:
        unique_indices, inverse = torch.unique(flat_indices, sorted=False, return_inverse=True)
        if unique_indices.numel() < total_n:
            k_unique = _dequant_gathered_flat(
                k_index_cache_fp8, unique_indices, page_size, head_dim, head_dim_sf
            )
            return k_unique.reshape(-1, page_size, head_dim)[inverse].reshape(-1, head_dim)

    return _dequant_gathered_flat(
        k_index_cache_fp8, flat_indices, page_size, head_dim, head_dim_sf
    )


def _get_page_offsets(page_size, device):
    cache_key = (page_size, device)
    offsets = _offset_cache.get(cache_key)
    if offsets is None:
        offsets = torch.arange(page_size, device=device, dtype=torch.int32)
        _offset_cache[cache_key] = offsets
    return offsets


def _score_and_select_tokens(
    q_b_fp32,
    weights_b,
    k_index_cache_fp8,
    page_indices_b,
    seq_len,
    page_size,
    head_dim,
    head_dim_sf,
    dedup_threshold,
):
    actual_topk = min(TOPK, seq_len)
    K = _gather_and_dequant(
        k_index_cache_fp8,
        page_indices_b,
        page_size,
        head_dim,
        head_dim_sf,
        page_indices_b.numel(),
        dedup_threshold,
    )[:seq_len]
    return _select_topk_tokens(q_b_fp32, weights_b, K, page_indices_b, min(TOPK, seq_len), page_size)


def _select_topk_tokens(q_b_fp32, weights_b, K, page_indices_b, actual_topk, page_size):
    scores = q_b_fp32 @ K.T
    scores.relu_()
    scores *= weights_b[:, None]
    final_scores = scores.sum(dim=0)
    _, topk_idx = torch.topk(final_scores, actual_topk)

    page_idx_per_token = topk_idx // page_size
    offset_per_token = topk_idx % page_size
    global_page_idx = page_indices_b[page_idx_per_token]
    topk_tokens = global_page_idx * page_size + offset_per_token
    return topk_tokens.to(torch.int32), actual_topk


def _write_topk(topk_idx, flat_indices, offsets, n_list, page_size, topk_indices, b, actual_topk):
    page_idx_per_token = topk_idx // page_size
    offset_per_token = topk_idx % page_size
    off = offsets[b]
    page_indices_b = flat_indices[off : off + n_list[b]]
    global_page_idx = page_indices_b[page_idx_per_token]
    topk_tokens = global_page_idx * page_size + offset_per_token
    topk_indices[b, :actual_topk] = topk_tokens.to(torch.int32)


def _run_small_multibatch(
    q_fp32,
    k_index_cache_fp8,
    weights,
    sl_list,
    n_list,
    block_table,
    topk_indices,
    num_pages,
    page_size,
    head_dim,
    head_dim_sf,
    device,
    dedup_threshold,
):
    """Grouped tiny-multibatch path: dequantize once, score per batch."""
    page_spans = []
    page_slices = []
    total_pages = 0
    for b in range(len(sl_list)):
        n_b = n_list[b]
        page_slices.append(block_table[b, :n_b])
        page_spans.append((total_pages, n_b))
        total_pages += n_b

    if total_pages == 0:
        return ()

    flat_indices = torch.cat(page_slices).to(torch.long)
    k_flat = _gather_and_dequant(
        k_index_cache_fp8,
        flat_indices,
        page_size,
        head_dim,
        head_dim_sf,
        total_pages,
        dedup_threshold,
    )

    row_off = 0
    for b, sl in enumerate(sl_list):
        n_b = n_list[b]
        if sl == 0:
            row_off += n_b * page_size
            continue

        page_off, n_b = page_spans[b]
        page_indices_b = flat_indices[page_off : page_off + n_b]
        K = k_flat[row_off : row_off + sl]
        topk_tokens, actual_topk = _select_topk_tokens(
            q_fp32[b],
            weights[b],
            K,
            page_indices_b,
            min(TOPK, sl),
            page_size,
        )
        topk_indices[b, :actual_topk] = topk_tokens
        row_off += n_b * page_size


def _run_grouped_score_batches(
    q_fp32,
    k_index_cache_fp8,
    weights,
    sl_list,
    n_list,
    batch_ids,
    flat_indices,
    offsets,
    topk_indices,
    page_size,
    head_dim,
    head_dim_sf,
    dedup_threshold,
):
    """Process multiple score-sorted batches with one gather+dequant pass."""
    if not batch_ids:
        return ()

    batch_count = len(offsets)
    is_full_batch_cover = (
        len(batch_ids) == batch_count
        and batch_ids[0] == 0
        and batch_ids[-1] == batch_count - 1
    )

    total_pages = 0
    if is_full_batch_cover:
        grouped_flat_indices = flat_indices
        for n_b in n_list:
            total_pages += n_b
    else:
        grouped_page_slices = []
        for b in batch_ids:
            off = offsets[b]
            n_b = n_list[b]
            grouped_page_slices.append(flat_indices[off : off + n_b])
            total_pages += n_b
        grouped_flat_indices = torch.cat(grouped_page_slices)

    grouped_k_flat = _gather_and_dequant(
        k_index_cache_fp8,
        grouped_flat_indices,
        page_size,
        head_dim,
        head_dim_sf,
        total_pages,
        dedup_threshold,
    )

    page_off = 0
    row_off = 0
    for b in batch_ids:
        sl = sl_list[b]
        n_b = n_list[b]
        page_indices_b = grouped_flat_indices[page_off : page_off + n_b]
        K = grouped_k_flat[row_off : row_off + sl]
        topk_tokens, actual_topk = _select_topk_tokens(
            q_fp32[b],
            weights[b],
            K,
            page_indices_b,
            min(TOPK, sl),
            page_size,
        )
        topk_indices[b, :actual_topk] = topk_tokens
        page_off += n_b
        row_off += n_b * page_size

    return ()


@torch.no_grad()
def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices):
    """DSA TopK Indexer v33 — hybrid mixed-batch dispatch."""
    batch_size = q_index_fp8.shape[0]
    num_pages, page_size, _, head_dim_sf = k_index_cache_fp8.shape
    head_dim = head_dim_sf - 4
    max_num_pages = block_table.shape[1]
    device = k_index_cache_fp8.device

    topk_indices.fill_(-1)

    # Ensure inputs are on the correct device
    if seq_lens.device.type != 'cuda':
        seq_lens = seq_lens.to(device)
    if block_table.device.type != 'cuda':
        block_table = block_table.to(device)
    if q_index_fp8.device.type != 'cuda':
        q_index_fp8 = q_index_fp8.to(device)
    if weights.device.type != 'cuda':
        weights = weights.to(device)

    dedup_threshold = _get_env_int(
        "DSA_INDEXER_V33_DEDUP_THRESHOLD", DEDUP_THRESHOLD, 0, 1 << 30
    )
    small_multi_batch_size = _get_env_int(
        "DSA_INDEXER_V33_SMALL_MULTI_BATCH_SIZE", SMALL_MULTI_BATCH_SIZE, 0, 1024
    )
    small_multi_total_n = _get_env_int(
        "DSA_INDEXER_V33_SMALL_MULTI_TOTAL_N", SMALL_MULTI_TOTAL_N, 0, 1 << 20
    )

    sl_list = seq_lens.tolist()

    if batch_size == 1:
        # ── Ultra-fast single-batch path ──
        sl = sl_list[0]
        if sl == 0:
            return ()

        n_pages = (sl + PAGE_SIZE - 1) // PAGE_SIZE
        flat_indices = block_table[0, :n_pages].to(torch.long)
        topk_tokens, actual_topk = _score_and_select_tokens(
            q_index_fp8[0].to(torch.float32),
            weights[0],
            k_index_cache_fp8,
            flat_indices,
            sl,
            page_size,
            head_dim,
            head_dim_sf,
            dedup_threshold,
        )
        topk_indices[0, :actual_topk] = topk_tokens

    else:
        # ── Multi-batch path with per-batch adaptive dispatch ──
        n_list = [(s + PAGE_SIZE - 1) // PAGE_SIZE for s in sl_list]
        offsets = [0] * batch_size
        total_n = 0
        for b in range(batch_size):
            offsets[b] = total_n
            total_n += n_list[b]

        if total_n == 0:
            return ()

        q_fp32 = q_index_fp8.to(torch.float32)

        if batch_size <= small_multi_batch_size and total_n <= small_multi_total_n:
            _run_small_multibatch(
                q_fp32, k_index_cache_fp8, weights, sl_list, n_list, block_table,
                topk_indices, num_pages, page_size, head_dim, head_dim_sf, device, dedup_threshold
            )
            return ()

        flat_indices = _compute_flat_indices(seq_lens, block_table, num_pages, max_num_pages, device)

        # Small sequences still need score-ordered outputs, so they cannot use
        # the old token-order fill shortcut.
        score_sorted_batches = [b for b in range(batch_size) if sl_list[b] <= TOPK and sl_list[b] > 0]
        topk_batches = [b for b in range(batch_size) if sl_list[b] > TOPK]

        _run_grouped_score_batches(
            q_fp32,
            k_index_cache_fp8,
            weights,
            sl_list,
            n_list,
            score_sorted_batches,
            flat_indices,
            offsets,
            topk_indices,
            page_size,
            head_dim,
            head_dim_sf,
            dedup_threshold,
        )

        if not topk_batches:
            return ()

        # For topk batches, gather only their pages
        topk_flat_offsets = []
        topk_total_n = 0
        topk_offsets = {}
        for b in topk_batches:
            off = offsets[b]
            n_b = n_list[b]
            topk_offsets[b] = (topk_total_n, n_b)
            topk_flat_offsets.append(flat_indices[off : off + n_b])
            topk_total_n += n_b

        topk_flat_indices = torch.cat(topk_flat_offsets)

        k_flat = _gather_and_dequant(
            k_index_cache_fp8, topk_flat_indices, page_size, head_dim, head_dim_sf, topk_total_n, dedup_threshold
        )

        for b in topk_batches:
            sl = sl_list[b]
            off_in_topk, n_b = topk_offsets[b]
            row_off = off_in_topk * page_size
            K = k_flat[row_off : row_off + sl]

            scores = q_fp32[b] @ K.T
            scores.relu_()
            scores *= weights[b, :, None]
            final_scores = scores.sum(dim=0)

            actual_topk = min(TOPK, sl)
            _, topk_idx = torch.topk(final_scores, actual_topk)

            page_idx_per_token = topk_idx // PAGE_SIZE
            offset_per_token = topk_idx % PAGE_SIZE
            page_indices_b = topk_flat_indices[off_in_topk : off_in_topk + n_b]
            global_page_idx = page_indices_b[page_idx_per_token]
            topk_tokens = global_page_idx * PAGE_SIZE + offset_per_token
            topk_indices[b, :actual_topk] = topk_tokens.to(torch.int32)

    return ()
