import torch
import triton
import triton.language as tl


@triton.jit
def qk_dot(
    q_ptr, k_ptr, d: int, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_INNER: tl.constexpr
):
    qk = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(tl.cdiv(d, BLOCK_SIZE_INNER)):
        q_block = tl.load(q_ptr, boundary_check=(0, 1), padding_option="zero")
        k_block = tl.load(k_ptr, boundary_check=(0, 1), padding_option="zero")
        qk = tl.dot(q_block, tl.trans(k_block, (1, 0)), acc=qk, input_precision="ieee")
        q_ptr = tl.advance(q_ptr, (0, BLOCK_SIZE_INNER))
        k_ptr = tl.advance(k_ptr, (0, BLOCK_SIZE_INNER))
    return qk


@triton.jit
def attention_with_linear_bias_kernel(
    q: tl.tensor,  # m * d
    k: tl.tensor,  # n * d
    v: tl.tensor,  # n * d
    out: tl.tensor,  # m * d
    m: int,
    n: int,
    d: int,
    ALPHA: tl.constexpr,
    REV_SQRT_D: tl.constexpr,
    D_MODEL: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_INNER: tl.constexpr,
):
    pid = tl.program_id(0)

    q_ptr = tl.make_block_ptr(
        base=q,
        shape=(m, d),
        strides=(d, 1),
        offsets=(pid * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_INNER),
        order=(1, 0),
    )
    q_offset = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    q_mask = q_offset < m

    k_ptr = tl.make_block_ptr(
        base=k,
        shape=(n, d),
        strides=(d, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_INNER),
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
    max_elem = tl.full((BLOCK_SIZE_M, 1), value=-float("inf"), dtype=tl.float32)

    for j in range(0, tl.cdiv(n, BLOCK_SIZE_N)):
        v_block = tl.load(v_ptr, boundary_check=(0, 1), padding_option="zero")
        qk = qk_dot(q_ptr, k_ptr, d, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_INNER) * REV_SQRT_D
        k_offset = j * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        k_mask = k_offset < n
        delta = ALPHA * (q_offset[:, None] - k_offset[None, :])
        mask = q_mask[:, None] & k_mask[None, :]
        qk = tl.where(mask, qk + delta, -float("inf"))

        new_max_elem = tl.maximum(max_elem, tl.max(qk, axis=1, keep_dims=True))
        compensation, new_weight = tl.exp(max_elem - new_max_elem), tl.exp(qk - new_max_elem)
        top = top * compensation + tl.dot(new_weight, v_block, input_precision="ieee")
        bottom = bottom * compensation + tl.sum(new_weight, axis=1, keep_dims=True)
        max_elem = new_max_elem

        k_ptr = tl.advance(k_ptr, offsets=(BLOCK_SIZE_N, 0))
        v_ptr = tl.advance(v_ptr, offsets=(BLOCK_SIZE_N, 0))

    top = top / bottom

    out_ptr = tl.make_block_ptr(
        base=out,
        shape=(m, d),
        strides=(d, 1),
        offsets=(pid * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, D_MODEL),
        order=(1, 0),
    )

    tl.store(out_ptr, top.to(dtype=out.dtype.element_ty), boundary_check=(0, 1))


# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    M: int,
    N: int,
    d: int,
    alpha: float,
):
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_INNER = 32, 32, 64
    grid = (triton.cdiv(M, BLOCK_SIZE_M),)
    attention_with_linear_bias_kernel[grid](
        Q,
        K,
        V,
        output,
        M,
        N,
        d,
        alpha,
        1.0 / (d**0.5),
        D_MODEL=triton.next_power_of_2(d) if d >= 16 else 16,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_INNER=BLOCK_SIZE_INNER,
    )
    return output


if __name__ == "__main__":
    M, N, d = 4, 4, 4
    alpha = 0.7

    q = torch.randn((M, d), device="cuda", dtype=torch.float32)
    k = torch.randn((N, d), device="cuda", dtype=torch.float32)
    v = torch.randn((N, d), device="cuda", dtype=torch.float32)

    qk = torch.matmul(q, torch.transpose(k, 1, 0)) * (1.0 / (d**0.5))
    bias = alpha * (
        torch.arange(0, M, device="cuda").to(dtype=torch.float32)[:, None]
        - torch.arange(0, N, device="cuda").to(dtype=torch.float32)[None, :]
    )
    import torch.nn.functional as F

    torch_result = torch.matmul(F.softmax(qk + bias, dim=1), v)

    triton_result = torch.empty((M, d), device="cuda", dtype=torch.float32)
    solve(q, k, v, triton_result, M, N, d, alpha)

    # print(f"triton_result: {triton_result}")
    # print(f"torch_result: {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
