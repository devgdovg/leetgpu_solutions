import torch
import triton
import triton.language as tl


@triton.jit
def causal_attention_kernel(
    q: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    output: tl.tensor,
    m: int,
    d: int,
    REV_SQRT_D: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid = tl.program_id(0)

    q_ptr = tl.make_block_ptr(
        base=q,
        shape=(m, d),
        strides=(d, 1),
        offsets=(pid * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_D),
        order=(1, 0),
    )
    q_block = tl.load(q_ptr, boundary_check=(0, 1), padding_option="zero")
    q_rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    q_rows_mask = q_rows < m

    k_ptr = tl.make_block_ptr(
        base=k,
        shape=(m, d),
        strides=(d, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_D),
        order=(1, 0),
    )
    v_ptr = tl.make_block_ptr(
        base=v,
        shape=(m, d),
        strides=(d, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_D),
        order=(1, 0),
    )

    top = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_D), dtype=tl.float32)
    bottom = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32)
    max_elem = tl.full((BLOCK_SIZE_M, 1), value=-float("inf"), dtype=tl.float32)
    for j in range(0, pid + 1):
        k_rows = j * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        k_rows_mask = k_rows < m
        boundary_mask = q_rows_mask[:, None] & k_rows_mask[None, :]
        causal_mask = q_rows[:, None] >= k_rows[None, :]
        final_mask = boundary_mask & causal_mask
        k_block = tl.load(k_ptr, boundary_check=(0, 1), padding_option="zero")
        v_block = tl.load(v_ptr, boundary_check=(0, 1), padding_option="zero")
        qk = tl.dot(q_block, tl.trans(k_block, (1, 0)), input_precision="ieee") * REV_SQRT_D
        qk = tl.where(final_mask, qk, -float("inf"))
        new_max_elem = tl.maximum(max_elem, tl.max(qk, axis=1, keep_dims=True))
        compensation = tl.exp(max_elem - new_max_elem)
        new_weight = tl.exp(qk - new_max_elem)
        top = top * compensation + tl.dot(new_weight, v_block, input_precision="ieee")
        bottom = bottom * compensation + tl.sum(new_weight, axis=1, keep_dims=True)
        max_elem = new_max_elem
        k_ptr = tl.advance(k_ptr, (BLOCK_SIZE_M, 0))
        v_ptr = tl.advance(v_ptr, (BLOCK_SIZE_M, 0))

    result = top / bottom

    output_ptr = tl.make_block_ptr(
        base=output,
        shape=(m, d),
        strides=(d, 1),
        offsets=(pid * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_D),
        order=(1, 0),
    )

    tl.store(output_ptr, result, boundary_check=(0, 1))


# Q, K, V, output are tensors on the GPU
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, d: int):

    BLOCK_SIZE_M = 32

    grid = (triton.cdiv(M, BLOCK_SIZE_M),)

    import math

    causal_attention_kernel[grid](
        Q,
        K,
        V,
        output,
        M,
        d,
        1.0 / math.sqrt(d),
        BLOCK_SIZE_M,
        BLOCK_SIZE_D=16 if d < 16 else triton.next_power_of_2(d),
    )


if __name__ == "__main__":
    M, d = 5000, 128

    q = torch.randn((M, d), device="cuda", dtype=torch.float32)
    k = torch.randn((M, d), device="cuda", dtype=torch.float32)
    v = torch.randn((M, d), device="cuda", dtype=torch.float32)

    import torch.nn.functional as F

    torch_result = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    triton_result = torch.empty((M, d), device="cuda", dtype=torch.float32)
    solve(q, k, v, triton_result, M, d)

    # print(f"triton_result: {triton_result}")
    # print(f"torch_result: {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
