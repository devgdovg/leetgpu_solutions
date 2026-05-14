import torch
import triton
import triton.language as tl


@triton.jit
def attn_kernel(
    q: torch.Tensor,  # m * d
    k: torch.Tensor,  # n * d
    v: torch.Tensor,  # n * d
    output: torch.Tensor,  # m * d
    m: int,
    n: int,
    d: int,
    rev_sqrt_d: torch.float32,  # type: ignore
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    D_MODEL: tl.constexpr,
):
    pid_0 = tl.program_id(0)

    q_ptr = tl.make_block_ptr(
        base=q,
        shape=(m, d),
        strides=(d, 1),
        offsets=(pid_0 * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, D_MODEL),
        order=(1, 0),
    )
    q_block = tl.load(q_ptr, boundary_check=(0, 1), padding_option="zero")

    k_ptr = tl.make_block_ptr(
        base=k,
        shape=(n, d),
        strides=(d, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_N, D_MODEL),
        order=(1, 0),
    )

    v_ptr = tl.make_block_ptr(
        base=v,
        shape=(n, d),
        strides=(d, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_N, D_MODEL),
        order=(1, 0),
    )

    top = tl.zeros((BLOCK_SIZE_M, D_MODEL), dtype=tl.float32)
    bottom = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32)
    max_elem = tl.full((BLOCK_SIZE_M, 1), -float("inf"), dtype=tl.float32)

    offset_m = pid_0 * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offset_m < m

    for j in range(0, tl.cdiv(n, BLOCK_SIZE_N)):

        k_block = tl.load(k_ptr, boundary_check=(0, 1), padding_option="zero")
        v_block = tl.load(v_ptr, boundary_check=(0, 1), padding_option="zero")

        qk = tl.dot(q_block, tl.trans(k_block, (1, 0))) * rev_sqrt_d

        offset_n = j * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_n = offset_n < n
        mask = mask_m[:, None] & mask_n[None, :]

        qk = tl.where(mask, qk, -float("inf"))

        new_max_elem = tl.maximum(max_elem, tl.max(qk, axis=1, keep_dims=True))
        compensation, new_weight = tl.exp(max_elem - new_max_elem), tl.exp(qk - new_max_elem)
        # top = top * compensation + tl.dot(new_weight.to(dtype=v.dtype.element_ty), v_block)
        top = top * compensation + tl.dot(new_weight, v_block)
        bottom = bottom * compensation + tl.sum(new_weight, axis=1, keep_dims=True)
        max_elem = new_max_elem

        k_ptr = tl.advance(k_ptr, offsets=(BLOCK_SIZE_N, 0))
        v_ptr = tl.advance(v_ptr, offsets=(BLOCK_SIZE_N, 0))

    top = top / bottom

    output_ptr = tl.make_block_ptr(
        base=output,
        shape=(m, d),
        strides=(d, 1),
        offsets=(pid_0 * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, D_MODEL),
        order=(1, 0),
    )

    tl.store(output_ptr, top.to(dtype=output.dtype.element_ty), boundary_check=(0, 1))


# Q, K, V, output are tensors on the GPU
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, N: int, d: int):
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    grid = (triton.cdiv(M, BLOCK_SIZE_M),)
    attn_kernel[grid](
        Q,
        K,
        V,
        output,
        M,
        N,
        d,
        1.0 / (d**0.5),
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        D_MODEL=d if d >= 16 else 16,
    )
    return output
