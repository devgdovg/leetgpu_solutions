import torch
import triton
import triton.language as tl


@triton.jit
def grouped_query_attn_kernel(
    q: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    output: tl.tensor,
    num_q_heads: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
    REV_SQRT_D: tl.constexpr,
    BLOCK_SIZE_SEQ: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_0 = tl.program_id(0)  # along heads
    pid_1 = tl.program_id(1)  # along sequence

    q_ptr = tl.make_block_ptr(
        base=q + pid_0 * seq_len * head_dim,
        shape=(seq_len, head_dim),
        strides=(head_dim, 1),
        offsets=(pid_1 * BLOCK_SIZE_SEQ, 0),
        block_shape=(BLOCK_SIZE_SEQ, BLOCK_SIZE_D),
        order=(1, 0),
    )
    q_block = tl.load(q_ptr, boundary_check=(0, 1), padding_option="zero")

    k_ptr = tl.make_block_ptr(
        base=k + pid_0 // (num_q_heads // num_kv_heads) * seq_len * head_dim,
        shape=(seq_len, head_dim),
        strides=(head_dim, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_SEQ, BLOCK_SIZE_D),
        order=(1, 0),
    )

    v_ptr = tl.make_block_ptr(
        base=v + pid_0 // (num_q_heads // num_kv_heads) * seq_len * head_dim,
        shape=(seq_len, head_dim),
        strides=(head_dim, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_SEQ, BLOCK_SIZE_D),
        order=(1, 0),
    )

    top = tl.zeros((BLOCK_SIZE_SEQ, BLOCK_SIZE_D), dtype=tl.float32)
    bottom = tl.zeros((BLOCK_SIZE_SEQ, 1), dtype=tl.float32)
    max_elem = tl.full((BLOCK_SIZE_SEQ, 1), value=-float("inf"), dtype=tl.float32)

    for j in range(tl.cdiv(seq_len, BLOCK_SIZE_SEQ)):
        k_block = tl.load(k_ptr, boundary_check=(0, 1), padding_option="zero")
        v_block = tl.load(v_ptr, boundary_check=(0, 1), padding_option="zero")
        qk = tl.dot(q_block, tl.trans(k_block, (1, 0)), input_precision="ieee") * REV_SQRT_D
        k_row_offsets = j * BLOCK_SIZE_SEQ + tl.arange(0, BLOCK_SIZE_SEQ)
        k_row_mask = k_row_offsets < seq_len
        qk = tl.where(k_row_mask[None, :], qk, -float("inf"))

        new_max_elem = tl.maximum(max_elem, tl.max(qk, axis=1, keep_dims=True))
        compensation = tl.exp(max_elem - new_max_elem)
        new_weight = tl.exp(qk - new_max_elem)
        top = top * compensation + tl.dot(new_weight, v_block, input_precision="ieee")
        bottom = bottom * compensation + tl.sum(new_weight, axis=1, keep_dims=True)
        max_elem = new_max_elem

        k_ptr = tl.advance(k_ptr, (BLOCK_SIZE_SEQ, 0))
        v_ptr = tl.advance(v_ptr, (BLOCK_SIZE_SEQ, 0))

    result = top / bottom

    outut_ptr = tl.make_block_ptr(
        base=output + pid_0 * seq_len * head_dim,
        shape=(seq_len, head_dim),
        strides=(head_dim, 1),
        offsets=(pid_1 * BLOCK_SIZE_SEQ, 0),
        block_shape=(BLOCK_SIZE_SEQ, BLOCK_SIZE_D),
        order=(1, 0),
    )

    tl.store(outut_ptr, result, boundary_check=(0, 1))


# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
):
    BLOCK_SIZE_SEQ = 32
    BLOCK_SIZE_D = 16 if head_dim < 16 else triton.next_power_of_2(head_dim)
    grid = (num_q_heads, triton.cdiv(seq_len, BLOCK_SIZE_SEQ))
    import math

    grouped_query_attn_kernel[grid](
        Q,
        K,
        V,
        output,
        num_q_heads,
        num_kv_heads,
        seq_len,
        head_dim,
        1.0 / math.sqrt(head_dim),
        BLOCK_SIZE_SEQ,
        BLOCK_SIZE_D,
    )


if __name__ == "__main__":

    num_q_heads: int = 32
    num_kv_heads: int = 8
    seq_len: int = 1024
    head_dim: int = 128

    q = torch.randn((num_q_heads, seq_len, head_dim), device="cuda", dtype=torch.float32)
    k = torch.randn((num_kv_heads, seq_len, head_dim), device="cuda", dtype=torch.float32)
    v = torch.randn((num_kv_heads, seq_len, head_dim), device="cuda", dtype=torch.float32)

    import torch.nn.functional as F

    torch_result = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

    triton_result = torch.empty((num_q_heads, seq_len, head_dim), device="cuda", dtype=torch.float32)
    solve(q, k, v, triton_result, num_q_heads, num_kv_heads, seq_len, head_dim)

    # print(f"triton_result:\n\t {triton_result}")
    # print(f"torch_result:\n\t {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
