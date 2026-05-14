import torch
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(
    a: tl.tensor,
    b: tl.tensor,
    c: tl.tensor,
    m: int,
    n: int,
    k: int,
    alpha: float,
    beta: float,
    stride_am: int,
    stride_ak: int,
    stride_bk: int,
    stride_bn: int,
    stride_cm: int,
    stride_cn: int,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m, num_pid_n = tl.cdiv(m, BLOCK_SIZE_M), tl.cdiv(n, BLOCK_SIZE_N)
    pid_m, pid_n = pid // num_pid_n, pid % num_pid_n
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE)

    accu = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    if alpha != 0:
        a_ptr = tl.make_block_ptr(
            base=a,
            shape=(m, k),
            strides=(stride_am, stride_ak),
            offsets=(pid_m * BLOCK_SIZE_M, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
            order=(1, 0),
        )

        b_ptr = tl.make_block_ptr(
            base=b,
            shape=(k, n),
            strides=(stride_bk, stride_bn),
            offsets=(0, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
            order=(1, 0),
        )

        for _ in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
            a_block = tl.load(a_ptr, boundary_check=(0, 1), padding_option="zero")
            b_block = tl.load(b_ptr, boundary_check=(0, 1), padding_option="zero")
            accu = accu + alpha * tl.dot(a_block, b_block, input_precision="ieee")
            a_ptr = tl.advance(a_ptr, (0, BLOCK_SIZE_K))
            b_ptr = tl.advance(b_ptr, (BLOCK_SIZE_K, 0))

    c_ptr = tl.make_block_ptr(
        base=c,
        shape=(m, n),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )

    if beta != 0:
        c_block = tl.load(c_ptr, boundary_check=(0, 1), padding_option="zero").to(dtype=tl.float32)
        accu = accu + c_block * beta

    tl.store(c_ptr, accu.to(dtype=tl.float16), boundary_check=(0, 1))


# a, b, c are tensors on the GPU
def solve(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    M: int,
    N: int,
    K: int,
    alpha: float,
    beta: float,
):
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    GROUP_SIZE = 8

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    gemm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        alpha,
        beta,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE,
    )


if __name__ == "__main__":
    m, n, k = 1024, 1024, 1024
    x = torch.randn((m, k), device="cuda", dtype=torch.float16)
    y = torch.randn((k, n), device="cuda", dtype=torch.float16)
    z = torch.randn((m, n), device="cuda", dtype=torch.float16)
    alpha, beta = 0.8, 0.7

    torch_result = alpha * (x @ y) + beta * z

    solve(x, y, z, m, n, k, alpha, beta)
    triton_result = z

    # print(f"triton_result: {triton_result}, torch_result: {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
