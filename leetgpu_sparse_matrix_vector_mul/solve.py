import torch
import triton
import triton.language as tl


@triton.jit
def mul_kernel(
    a: tl.tensor,
    x: tl.tensor,
    output: tl.tensor,
    m: int,
    n: int,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    a_ptr = tl.make_block_ptr(
        base=a,
        shape=(m, n),
        strides=(n, 1),
        offsets=(pid * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    x_ptr = tl.make_block_ptr(
        base=x,
        shape=(n,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_N,),
        order=(0,),
    )
    accu = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32)
    for _ in range(0, tl.cdiv(n, BLOCK_SIZE_N)):
        a_block = tl.load(a_ptr, boundary_check=(0, 1), padding_option="zero").to(dtype=tl.float32)
        x_block = tl.load(x_ptr, boundary_check=(0,), padding_option="zero").to(dtype=tl.float32)
        accu += tl.sum(a_block * x_block, axis=1, keep_dims=True)
        a_ptr = tl.advance(a_ptr, (0, BLOCK_SIZE_N))
        x_ptr = tl.advance(x_ptr, (BLOCK_SIZE_N,))
    output_ptr = tl.make_block_ptr(
        base=output,
        shape=(m,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE_M,),
        block_shape=(BLOCK_SIZE_M,),
        order=(0,),
    )
    tl.store(
        output_ptr,
        tl.reshape(accu, (BLOCK_SIZE_M,)).to(dtype=output.dtype.element_ty),
        boundary_check=(0,),
    )


# A, x, y are tensors on the GPU
def solve(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, M: int, N: int, nnz: int):
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 256
    grid = (triton.cdiv(M, BLOCK_SIZE_M),)
    mul_kernel[grid](A, x, y, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N)


if __name__ == "__main__":
    m, n = 10_000, 10_000
    a = torch.randn((m, n), device="cuda", dtype=torch.float32)
    x = torch.randn((n,), device="cuda", dtype=torch.float32)

    torch_result = torch.matmul(a, x[:, None]).reshape((m,))

    triton_result = torch.empty((m,), device="cuda", dtype=torch.float32)
    solve(a, x, triton_result, m, n, 0)

    # print(f"triton_result: {triton_result}, torch_result: {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
