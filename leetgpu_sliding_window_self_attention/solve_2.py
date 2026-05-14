import torch
import triton
import triton.language as tl


@triton.jit
def sliding_win_self_attn_kernel(
    q: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    output: tl.tensor,
    m: int,
    d: int,
    REV_SQRT_D: tl.constexpr,
    WINDOW: tl.constexpr,
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

    kv_row_start = max(0, pid * BLOCK_SIZE_M - WINDOW)
    kv_row_end_exclusive = min(m, (pid + 1) * BLOCK_SIZE_M + WINDOW)

    k_ptr = tl.make_block_ptr(
        base=k,
        shape=(m, d),
        strides=(d, 1),
        offsets=(kv_row_start, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_D),
        order=(1, 0),
    )
    v_ptr = tl.make_block_ptr(
        base=v,
        shape=(m, d),
        strides=(d, 1),
        offsets=(kv_row_start, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_D),
        order=(1, 0),
    )

    top = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_D), dtype=tl.float32)
    bottom = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32)
    max_elem = tl.full((BLOCK_SIZE_M, 1), value=-float("inf"), dtype=tl.float32)

    for j in range(0, tl.cdiv(kv_row_end_exclusive - kv_row_start, BLOCK_SIZE_M)):
        k_block = tl.load(k_ptr, boundary_check=(0, 1), padding_option="zero")
        v_block = tl.load(v_ptr, boundary_check=(0, 1), padding_option="zero")
        qk = tl.dot(q_block, tl.trans(k_block, (1, 0)), input_precision="ieee") * REV_SQRT_D
        kv_rows = j * BLOCK_SIZE_M + kv_row_start + tl.arange(0, BLOCK_SIZE_M)
        boundary_mask = kv_rows[None, :] < m
        window_mask = (kv_rows[None, :] >= (q_rows[:, None] - WINDOW)) & (
            kv_rows[None, :] <= (q_rows[:, None] + WINDOW)
        )
        qk = tl.where(boundary_mask & window_mask, qk, -float("inf"))
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
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    M: int,
    d: int,
    window_size: int,
):

    BLOCK_SIZE_M = 32

    grid = (triton.cdiv(M, BLOCK_SIZE_M),)

    import math

    sliding_win_self_attn_kernel[grid](
        Q,
        K,
        V,
        output,
        M,
        d,
        1.0 / math.sqrt(d),
        window_size,
        BLOCK_SIZE_M,
        BLOCK_SIZE_D=16 if d < 16 else triton.next_power_of_2(d),
    )


if __name__ == "__main__":
    M, d = 10_000, 128
    window_size = 32

    q = torch.randn((M, d), device="cuda", dtype=torch.float32)
    k = torch.randn((M, d), device="cuda", dtype=torch.float32)
    v = torch.randn((M, d), device="cuda", dtype=torch.float32)

    # torch_result = torch.empty((M, d), device="cuda", dtype=torch.float32)
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    def sliding_window_mask(b, h, q_idx, kv_idx):
        return (kv_idx >= q_idx - window_size) & (kv_idx <= q_idx + window_size)

    block_mask = create_block_mask(sliding_window_mask, B=None, H=None, Q_LEN=M, KV_LEN=M)
    compiled_flex = torch.compile(flex_attention)
    q_torch = q.unsqueeze(0).unsqueeze(0)
    k_torch = k.unsqueeze(0).unsqueeze(0)
    v_torch = v.unsqueeze(0).unsqueeze(0)
    torch_result = compiled_flex(q_torch, k_torch, v_torch, block_mask=block_mask).squeeze()

    triton_result = torch.empty((M, d), device="cuda", dtype=torch.float32)
    solve(q, k, v, triton_result, M, d, window_size)

    # print(f"triton_result:\n\t {triton_result}")
    # print(f"torch_result:\n\t {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
