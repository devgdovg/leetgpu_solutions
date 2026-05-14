import torch
import triton
import triton.language as tl


@triton.jit
def _batch_matmul_kernel(
    a: tl.tensor,
    b: tl.tensor,
    output: tl.tensor,
    batch: int,
    m: int,
    n: int,
    k: int,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_0, pid_1, pid_2 = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    a_ptr = tl.make_block_ptr(
        base=a,
        shape=(batch, m, k),
        strides=(m * k, k, 1),
        offsets=(pid_0 * BLOCK_SIZE_BATCH, pid_1 * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_BATCH, BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(2, 1, 0),
    )
    b_ptr = tl.make_block_ptr(
        base=b,
        shape=(batch, k, n),
        strides=(k * n, n, 1),
        offsets=(pid_0 * BLOCK_SIZE_BATCH, 0, pid_2 * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_BATCH, BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(2, 1, 0),
    )

    accu = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
        a_block = tl.load(a_ptr, boundary_check=(0, 1, 2), padding_option="zero")
        b_block = tl.load(b_ptr, boundary_check=(0, 1, 2), padding_option="zero")
        accu = tl.dot(a_block, b_block, acc=accu, input_precision="ieee")  # `input_precision="ieee"` IS IMPORTANT
        a_ptr = tl.advance(a_ptr, (0, 0, BLOCK_SIZE_K))
        b_ptr = tl.advance(b_ptr, (0, BLOCK_SIZE_K, 0))

    output_ptr = tl.make_block_ptr(
        base=output,
        shape=(batch, m, n),
        strides=(m * n, n, 1),
        offsets=(pid_0 * BLOCK_SIZE_BATCH, pid_1 * BLOCK_SIZE_M, pid_2 * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_BATCH, BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(2, 1, 0),
    )

    tl.store(output_ptr, accu.to(dtype=output.dtype.element_ty), boundary_check=(0, 1, 2))


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, BATCH: int, M: int, N: int, K: int):

    BLOCK_SIZE_BATCH, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 4, 32, 64, 32

    grid = (
        triton.cdiv(BATCH, BLOCK_SIZE_BATCH),
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )

    _batch_matmul_kernel[grid](a, b, c, BATCH, M, N, K, BLOCK_SIZE_BATCH, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)


if __name__ == "__main__":
    B, N, M, K = 128, 256, 256, 256

    a = torch.randn((B, M, K), device="cuda", dtype=torch.float32)
    b = torch.ones((B, K, N), device="cuda", dtype=torch.float32)

    torch_result = torch.matmul(a, b)

    triton_result = torch.empty((B, M, N), device="cuda", dtype=torch.float32)
    solve(a, b, triton_result, B, M, N, K)

    # print(f"triton_result: {triton_result}")
    # print(f"torch_result: {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
