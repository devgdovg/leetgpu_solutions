import torch
import triton
import triton.language as tl


@triton.jit
def sepculative_decoding_verification_kernel(
    draft_tokens: tl.tensor,
    draft_probs: tl.tensor,
    target_probs: tl.tensor,
    uniform_samples: tl.tensor,
    output_tokens: tl.tensor,
    b: int,
    t: int,
    v: int,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
):
    pid = tl.program_id(0)  # along b
    draft_tokens_range = tl.arange(0, BLOCK_SIZE_T)
    draft_tokens_offsets = pid * t + draft_tokens_range
    draft_tokens_mask = draft_tokens_range < t
    draft_tokens_block = tl.load(draft_tokens + draft_tokens_offsets, mask=draft_tokens_mask, other=0)

    uniform_samples_offsets = pid * (t + 1) + draft_tokens_range
    uniform_samples_block = tl.load(uniform_samples + uniform_samples_offsets, mask=draft_tokens_mask, other=0.0)
    resample_prob = tl.load(uniform_samples + pid * (t + 1) + t)

    prob_indices = draft_tokens_block + draft_tokens_range * v + pid * t * v
    """
    p_0 -> 0*v + t0
    p_1 -> 1*v + t1
    ... ...
    p_t -> t*v + t3
    """
    draft_probs_by_index = tl.load(draft_probs + prob_indices, mask=draft_tokens_mask, other=0.0)
    target_probs_by_index = tl.load(target_probs + prob_indices, mask=draft_tokens_mask, other=0.0)

    alpha = tl.minimum(1, target_probs_by_index / draft_probs_by_index)
    accept_or_reject = tl.cumprod(tl.where(uniform_samples_block < alpha, 1, 0).to(dtype=tl.int32), axis=0)
    accept_or_reject = tl.where(draft_tokens_mask, accept_or_reject, 0)
    reject_index = tl.sum(accept_or_reject, axis=0, keep_dims=False)
    # tl.device_assert(reject_index == 3)

    v_range = tl.arange(0, BLOCK_SIZE_V)
    prob_block_mask = v_range < v
    if reject_index == t:
        bonus_prob_offsets = (pid * t + t - 1) * v + v_range
        bonus_prob_block = tl.load(target_probs + bonus_prob_offsets, mask=prob_block_mask, other=0.0)
        cdf = tl.cumsum(bonus_prob_block, axis=0)
    else:
        prob_block_offsets = (pid * t + reject_index) * v + v_range
        draft_probs_block = tl.load(draft_probs + prob_block_offsets, mask=prob_block_mask, other=0.0)
        target_probs_block = tl.load(target_probs + prob_block_offsets, mask=prob_block_mask, other=0.0)
        adj = tl.maximum(0, target_probs_block - draft_probs_block)
        adj_sum = tl.sum(adj, axis=0, keep_dims=False)
        if adj_sum == 0:
            adj = tl.full((BLOCK_SIZE_V,), 1.0 / v, dtype=tl.float32)
        else:
            adj = adj / adj_sum
        cdf = tl.cumsum(adj, axis=0)

    resample_token = tl.sum(tl.where(cdf < resample_prob, 1, 0).to(dtype=tl.int32), axis=0, keep_dims=False)

    output_tokens_block = tl.where(draft_tokens_range < reject_index, draft_tokens_block, 0)
    output_tokens_block = tl.where(draft_tokens_range == reject_index, resample_token, output_tokens_block)

    output_tokens_offsets = uniform_samples_offsets
    output_tokens_mask = draft_tokens_range < (t + 1)

    tl.store(output_tokens + output_tokens_offsets, output_tokens_block, mask=output_tokens_mask)


# draft_tokens, draft_probs, target_probs, uniform_samples, output_tokens are tensors on the GPU
def solve(
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    output_tokens: torch.Tensor,
    B: int,
    T: int,
    V: int,
):
    grid = (B,)
    BLOCK_SIZE_T = triton.next_power_of_2(T + 1)
    BLOCK_SIZE_V = triton.next_power_of_2(V)
    sepculative_decoding_verification_kernel[grid](
        draft_tokens,
        draft_probs,
        target_probs,
        uniform_samples,
        output_tokens,
        B,
        T,
        V,
        BLOCK_SIZE_T,
        BLOCK_SIZE_V,
    )


if __name__ == "__main__":
    B = 2
    T = 3
    V = 4

    # draft_tokens = torch.tensor([1, 2, 0], device="cuda", dtype=torch.int32)
    draft_tokens = torch.tensor([[1, 2, 0], [1, 2, 2]], device="cuda", dtype=torch.int32)
    draft_probs = torch.tensor(
        [
            [[0.1, 0.6, 0.2, 0.1], [0.1, 0.2, 0.5, 0.2], [0.1, 0.2, 0.5, 0.2]],
            [[0.1, 0.6, 0.2, 0.1], [0.1, 0.2, 0.5, 0.2], [0.1, 0.2, 0.5, 0.2]],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    target_probs = torch.tensor(
        [
            [[0.1, 0.5, 0.2, 0.2], [0.3, 0.2, 0.2, 0.3], [0.1, 0.2, 0.5, 0.2]],
            [[0.1, 0.5, 0.2, 0.2], [0.3, 0.2, 0.2, 0.3], [0.1, 0.2, 0.5, 0.2]],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    uniform_samples = torch.tensor([[0.5, 0.7, 0.3, 0.9], [0.5, 0.3, 0.3, 0.9]], device="cuda", dtype=torch.float32)

    # import torch.nn.functional as F
    # torch_result = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    triton_result = torch.empty(
        (
            B,
            T + 1,
        ),
        device="cuda",
        dtype=torch.int32,
    )
    solve(draft_tokens, draft_probs, target_probs, uniform_samples, triton_result, B, T, V)

    print(f"triton_result: {triton_result}")
    # print(f"torch_result: {torch_result}")
    # print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    # print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
