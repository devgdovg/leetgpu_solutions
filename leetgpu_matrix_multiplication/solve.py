import torch
import triton
import triton.language as tl


@triton.jit
def matrix_multiplication_kernel(
    a,
    b,
    c,
    M,
    N,
    K,
    stride_am,
    stride_an,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_ck,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offset_am = tl.arange(0, BLOCK_SIZE_M) + pid_m * BLOCK_SIZE_M
    mask_am = offset_am < M
    offset_bk = tl.arange(0, BLOCK_SIZE_K) + pid_k * BLOCK_SIZE_K
    mask_bk = offset_bk < K

    accu = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        offset_n = tl.arange(0, BLOCK_SIZE_N) + n * BLOCK_SIZE_N
        mask_n = offset_n < N

        a_ptr = offset_am[:, None] * stride_am + offset_n[None, :]
        a_mask = mask_am[:, None] & mask_n[None, :]
        an = tl.load(a + a_ptr, a_mask, other=0.0)

        b_ptr = offset_n[:, None] * stride_bn + offset_bk[None, :]
        b_mask = mask_n[:, None] & mask_bk[None, :]
        bn = tl.load(b + b_ptr, b_mask, other=0.0)

        accu = tl.dot(an, bn, acc=accu, input_precision="ieee")

    c_ptr = offset_am[:, None] * stride_cm + offset_bk[None, :]
    c_mask = mask_am[:, None] & mask_bk[None, :]

    tl.store(c + c_ptr, accu, c_mask)


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = N, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_ck = K, 1

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_K = 64
    BLOCK_SIZE_N = 64

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(K, BLOCK_SIZE_K))
    matrix_multiplication_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        stride_am,
        stride_an,
        stride_bn,
        stride_bk,
        stride_cm,
        stride_ck,
        BLOCK_SIZE_M,
        BLOCK_SIZE_K,
        BLOCK_SIZE_N,
    )


if __name__ == "__main__":
    m, n, k = 8192, 6144, 4096
    a = torch.randn((m, n), device="cuda", dtype=torch.float16)
    b = torch.randn((n, k), device="cuda", dtype=torch.float16)

    torch_result = a @ b

    triton_result = torch.randn((m, k), device="cuda", dtype=torch.float16)
    solve(a, b, triton_result, m, n, k)

    # print(f"triton_result: {triton_result}, torch_result: {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
