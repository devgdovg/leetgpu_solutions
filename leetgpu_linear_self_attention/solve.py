import torch
import triton
import triton.language as tl


@triton.jit
def phi(x):
    return tl.where(x > 0, x + 1, tl.exp(x))


@triton.jit
def kv_reduce_kernel(
    k: tl.tensor,
    v: tl.tensor,
    kv: tl.tensor,
    sum_k: tl.tensor,
    m: int,
    d: int,
    BLOCK_SIZE_SEQ: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_0 = tl.program_id(0)  # along k
    pid_1 = tl.program_id(1)  # along v

    k_ptr = tl.make_block_ptr(
        base=k,
        shape=(m, d),
        strides=(d, 1),
        offsets=(0, pid_0 * BLOCK_SIZE_D),
        block_shape=(BLOCK_SIZE_SEQ, BLOCK_SIZE_D),
        order=(1, 0),
    )

    v_ptr = tl.make_block_ptr(
        base=v,
        shape=(m, d),
        strides=(d, 1),
        offsets=(0, pid_1 * BLOCK_SIZE_D),
        block_shape=(BLOCK_SIZE_SEQ, BLOCK_SIZE_D),
        order=(1, 0),
    )

    accu_kv = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_D), dtype=tl.float32)
    accu_sum_k = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)

    for j in range(0, tl.cdiv(m, BLOCK_SIZE_SEQ)):
        k_block = tl.load(k_ptr, boundary_check=(0, 1), padding_option="zero")
        v_block = tl.load(v_ptr, boundary_check=(0, 1), padding_option="zero")
        k_block = tl.trans(phi(k_block), (1, 0))
        accu_kv = tl.dot(k_block, v_block, acc=accu_kv, input_precision="ieee")
        cols = j * BLOCK_SIZE_SEQ + tl.arange(0, BLOCK_SIZE_SEQ)
        k_block = tl.where((cols < m)[None, :], k_block, 0)
        accu_sum_k = accu_sum_k + tl.sum(k_block, axis=1, keep_dims=False)
        k_ptr = tl.advance(k_ptr, (BLOCK_SIZE_SEQ, 0))
        v_ptr = tl.advance(v_ptr, (BLOCK_SIZE_SEQ, 0))

    kv_ptr = tl.make_block_ptr(
        base=kv,
        shape=(d, d),
        strides=(d, 1),
        offsets=(pid_0 * BLOCK_SIZE_D, pid_1 * BLOCK_SIZE_D),
        block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(kv_ptr, accu_kv, boundary_check=(0, 1))

    if pid_1 == 0:
        sum_k_ptr = tl.make_block_ptr(
            base=sum_k,
            shape=(d,),
            strides=(1,),
            offsets=(pid_0 * BLOCK_SIZE_D),
            block_shape=(BLOCK_SIZE_D,),
            order=(0,),
        )
        tl.store(sum_k_ptr, accu_sum_k, boundary_check=(0,))


@triton.jit
def linear_self_attn_kernel(
    q: tl.tensor,
    kv: tl.tensor,
    sum_k: tl.tensor,
    output: tl.tensor,
    m: int,
    d: int,
    BLOCK_SIZE_SEQ: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_0 = tl.program_id(0)  # along q
    pid_1 = tl.program_id(1)  # along kv

    top = tl.zeros((BLOCK_SIZE_SEQ, BLOCK_SIZE_D), dtype=tl.float32)
    bottom = tl.zeros((BLOCK_SIZE_SEQ, 1), dtype=tl.float32)

    q_ptr = tl.make_block_ptr(
        base=q,
        shape=(m, d),
        strides=(d, 1),
        offsets=(pid_0 * BLOCK_SIZE_SEQ, 0),
        block_shape=(BLOCK_SIZE_SEQ, BLOCK_SIZE_D),
        order=(1, 0),
    )

    kv_ptr = tl.make_block_ptr(
        base=kv,
        shape=(d, d),
        strides=(d, 1),
        offsets=(0, pid_1 * BLOCK_SIZE_D),
        block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_D),
        order=(1, 0),
    )

    sum_k_ptr = tl.make_block_ptr(
        base=sum_k, shape=(d,), strides=(1,), offsets=(0,), block_shape=(BLOCK_SIZE_D,), order=(0,)
    )

    top = tl.zeros((BLOCK_SIZE_SEQ, BLOCK_SIZE_D), dtype=tl.float32)
    bottom = tl.zeros((BLOCK_SIZE_SEQ, 1), dtype=tl.float32)

    for _ in range(0, tl.cdiv(d, BLOCK_SIZE_D)):
        q_block = tl.load(q_ptr, boundary_check=(0, 1), padding_option="zero")
        kv_block = tl.load(kv_ptr, boundary_check=(0, 1), padding_option="zero")
        sum_k_block = tl.load(sum_k_ptr, boundary_check=(0,), padding_option="zero")
        q_block = phi(q_block)
        top = tl.dot(q_block, kv_block, acc=top, input_precision="ieee")
        bottom = bottom + tl.sum(q_block * sum_k_block, axis=1, keep_dims=True)
        q_ptr = tl.advance(q_ptr, (0, BLOCK_SIZE_D))
        kv_ptr = tl.advance(kv_ptr, (BLOCK_SIZE_D, 0))
        sum_k_ptr = tl.advance(sum_k_ptr, (BLOCK_SIZE_D,))

    result = top / (bottom + 1e-6)

    output_ptr = tl.make_block_ptr(
        base=output,
        shape=(m, d),
        strides=(d, 1),
        offsets=(pid_0 * BLOCK_SIZE_SEQ, pid_1 * BLOCK_SIZE_D),
        block_shape=(BLOCK_SIZE_SEQ, BLOCK_SIZE_D),
        order=(1, 0),
    )

    tl.store(output_ptr, result, boundary_check=(0, 1))


# Q, K, V, output are tensors on the GPU
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, d: int):
    BLOCK_SIZE_SEQ = 128
    BLOCK_SIZE_D = 32
    kv = torch.zeros((d, d), device="cuda", dtype=torch.float32)
    sum_k = torch.zeros((d,), device="cuda", dtype=torch.float32)

    grid = (triton.cdiv(d, BLOCK_SIZE_D), triton.cdiv(d, BLOCK_SIZE_D))

    kv_reduce_kernel[grid](K, V, kv, sum_k, M, d, BLOCK_SIZE_SEQ, BLOCK_SIZE_D)

    grid = (triton.cdiv(M, BLOCK_SIZE_SEQ), triton.cdiv(d, BLOCK_SIZE_D))

    linear_self_attn_kernel[grid](Q, kv, sum_k, output, M, d, BLOCK_SIZE_SEQ, BLOCK_SIZE_D)


if __name__ == "__main__":
    M, d = 10_000, 128

    q = torch.randn((M, d), device="cuda", dtype=torch.float32)
    k = torch.randn((M, d), device="cuda", dtype=torch.float32)
    v = torch.randn((M, d), device="cuda", dtype=torch.float32)

    def _phi(x):
        return torch.where(x > 0, x + 1, torch.exp(x))

    _phi_q = _phi(q)
    _phi_k = _phi(k.transpose(1, 0))
    torch_result = torch.matmul(_phi_q, torch.matmul(_phi_k, v)) / torch.matmul(
        _phi(q), torch.sum(_phi_k, dim=1, keepdim=True)
    )

    triton_result = torch.empty((M, d), device="cuda", dtype=torch.float32)
    solve(q, k, v, triton_result, M, d)

    # print(f"triton_result:\n\t {triton_result}")
    # print(f"torch_result:\n\t {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
