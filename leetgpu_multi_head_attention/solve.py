import torch
import triton
import triton.language as tl


@triton.jit
def multi_head_attn_kernel(
    q: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    output: tl.tensor,
    n: int,
    d_model: int,
    d_head: int,
    REV_SQRT_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    row_offsets = pid_0 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    row_mask = row_offsets < n

    col_range = tl.arange(0, BLOCK_SIZE_D)
    col_offsets = pid_1 * d_head + col_range
    col_mask = (col_offsets < d_model) & (col_range < d_head)
    block_offsets = row_offsets[:, None] * d_model + col_offsets[None, :]
    block_mask = row_mask[:, None] & col_mask[None, :]

    q_block = tl.load(q + block_offsets, mask=block_mask, other=0.0)

    output_ptr = output + block_offsets
    output_mask = block_mask

    top = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_D), dtype=tl.float32)
    bottom = tl.zeros((BLOCK_SIZE_N, 1), tl.float32)
    max_elem = tl.full((BLOCK_SIZE_N, 1), value=-float("inf"), dtype=tl.float32)

    for j in range(0, tl.cdiv(n, BLOCK_SIZE_N)):
        row_offsets = j * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        row_mask = row_offsets < n
        block_offsets = row_offsets[:, None] * d_model + col_offsets[None, :]
        block_mask = row_mask[:, None] & col_mask[None, :]
        k_block = tl.load(k + block_offsets, mask=block_mask, other=0.0)
        v_block = tl.load(v + block_offsets, mask=block_mask, other=0.0)

        qk = tl.dot(q_block, tl.trans(k_block, (1, 0)), input_precision="ieee") * REV_SQRT_D
        qk = tl.where(row_mask[None, :], qk, -float("inf"))

        new_max_elem = tl.maximum(max_elem, tl.max(qk, axis=1, keep_dims=True))
        compensation = tl.exp(max_elem - new_max_elem)
        new_weight = tl.exp(qk - new_max_elem)
        top = top * compensation + tl.dot(new_weight, v_block, input_precision="ieee")
        bottom = bottom * compensation + tl.sum(new_weight, axis=1, keep_dims=True)
        max_elem = new_max_elem

    result = top / bottom

    tl.store(output_ptr, result, mask=output_mask)


# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    N: int,
    d_model: int,
    h: int,
):
    BLOCK_SIZE_N = 32
    d_head = d_model // h
    BLOCK_SIZE_D = 16 if d_head < 16 else triton.next_power_of_2(d_head)
    grid = (triton.cdiv(N, BLOCK_SIZE_N), h)

    import math

    multi_head_attn_kernel[grid](
        Q, K, V, output, N, d_model, d_head, 1.0 / math.sqrt(d_head), BLOCK_SIZE_N, BLOCK_SIZE_D
    )


if __name__ == "__main__":
    N, d, h = 1024, 1024, 32

    q = torch.randn((N, d), device="cuda", dtype=torch.float32)
    k = torch.randn((N, d), device="cuda", dtype=torch.float32)
    v = torch.randn((N, d), device="cuda", dtype=torch.float32)

    import torch.nn.functional as F

    q_torch = q.reshape(N, h, d // h).transpose(1, 0)
    k_torch = k.reshape(N, h, d // h).transpose(1, 0)
    v_torch = v.reshape(N, h, d // h).transpose(1, 0)
    torch_result = F.scaled_dot_product_attention(q_torch, k_torch, v_torch).transpose(1, 0).reshape(N, d)

    triton_result = torch.empty((N, d), device="cuda", dtype=torch.float32)
    solve(q, k, v, triton_result, N, d, h)

    # print(f"triton_result:\n\t {triton_result}")
    # print(f"torch_result:\n\t {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
