import torch
import triton
import triton.language as tl


@triton.jit
def moe_topk_gating_kernel(
    logits: tl.tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    m: int,
    e: int,
    k: int,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_E: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)

    logits_ptr = tl.make_block_ptr(
        base=logits,
        shape=(m, e),
        strides=(e, 1),
        offsets=(pid * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_E),
        order=(1, 0),
    )

    logits_data = tl.load(logits_ptr, boundary_check=(0, 1), padding_option="zero")
    e_range = tl.arange(0, BLOCK_SIZE_E)
    logits_data = tl.where(e_range < e, logits_data, -float("inf"))

    output_weights = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_K), value=-float("inf"), dtype=tl.float32)
    output_indices = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.int32)

    max_elem = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32)
    k_range = tl.arange(0, BLOCK_SIZE_K)

    for j in range(0, k):
        jth_val = tl.max(logits_data, axis=1, keep_dims=True)
        jth_idx = tl.argmax(logits_data, axis=1, keep_dims=True).to(dtype=tl.int32)

        if j == 0:
            max_elem = jth_val

        jth_pos = k_range[None, :] == j
        output_weights = tl.where(jth_pos, jth_val, output_weights)
        output_indices = tl.where(jth_pos, jth_idx, output_indices)

        logits_pos = e_range[None, :] == jth_idx
        logits_data = tl.where(logits_pos, -float("inf"), logits_data)

    top = tl.exp(output_weights - max_elem)
    bottom = tl.sum(top, axis=1, keep_dims=True)
    output_weights = top / bottom

    topk_weights_ptr = tl.make_block_ptr(
        base=topk_weights,
        shape=(m, k),
        strides=(k, 1),
        offsets=(pid * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )

    topk_indices_ptr = tl.make_block_ptr(
        base=topk_indices,
        shape=(m, k),
        strides=(k, 1),
        offsets=(pid * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )

    tl.store(topk_weights_ptr, output_weights, boundary_check=(0, 1))
    tl.store(topk_indices_ptr, output_indices, boundary_check=(0, 1))


# logits, topk_weights, topk_indices are tensors on the GPU
def solve(
    logits: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    M: int,
    E: int,
    k: int,
):
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_E = triton.next_power_of_2(E)
    BLOCK_SIZE_K = triton.next_power_of_2(k)

    grid = (triton.cdiv(M, BLOCK_SIZE_M),)

    moe_topk_gating_kernel[grid](logits, topk_weights, topk_indices, M, E, k, BLOCK_SIZE_M, BLOCK_SIZE_E, BLOCK_SIZE_K)


if __name__ == "__main__":
    M, E, k = 2, 5, 2

    logits = torch.randn((M, E), device="cuda", dtype=torch.float32)

    topk_weights = torch.empty((M, k), device="cuda", dtype=torch.float32)
    topk_indices = torch.empty((M, k), device="cuda", dtype=torch.int32)

    solve(logits, topk_weights, topk_indices, M, E, k)

    print(f"logits: {logits}")
    print(f"topk_weights: {topk_weights}")
    print(f"topk_indices: {topk_indices}")
    # print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    # print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
