"""
https://leetgpu.com/challenges/fp16-dot-product
"""

import torch
import triton
import triton.language as tl


@triton.jit
def dot_product_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    n: int,
    BLOCK_SIZE_N: tl.constexpr,
):
    a_ptr = tl.make_block_ptr(
        base=a,
        shape=(n,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_N,),
        order=(0,),
    )
    b_ptr = tl.make_block_ptr(
        base=b,
        shape=(n,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_N,),
        order=(0,),
    )

    accu = tl.zeros((1,), dtype=tl.float32)

    for _ in range(0, tl.cdiv(n, BLOCK_SIZE_N)):
        a_block = tl.load(a_ptr, boundary_check=(0,), padding_option="zero").to(dtype=tl.float32)
        b_block = tl.load(b_ptr, boundary_check=(0,), padding_option="zero").to(dtype=tl.float32)
        accu = accu + tl.sum(a_block * b_block, keep_dims=False).to(dtype=tl.float32)
        a_ptr = tl.advance(a_ptr, (BLOCK_SIZE_N,))
        b_ptr = tl.advance(b_ptr, (BLOCK_SIZE_N,))

    out_ptr = tl.make_block_ptr(
        base=output,
        shape=(1,),
        strides=(1,),
        offsets=(0,),
        block_shape=(1,),
        order=(0,),
    )

    tl.store(out_ptr, accu.to(dtype=output.dtype.element_ty), boundary_check=(0,))


# A, B, result are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, result: torch.Tensor, N: int):
    grid = (1,)
    BLOCK_SIZE_N = 2048
    dot_product_kernel[grid](A, B, result, n=N, BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=8)


if __name__ == "__main__":
    length = 100_000_000
    a = torch.randn(length, dtype=torch.float16, device="cuda")
    b = (torch.rand_like(a) * 10.0).to(dtype=torch.float16, device="cuda")

    triton_result = torch.empty((1,), dtype=torch.float16, device="cuda")
    solve(a, b, triton_result, length)

    torch_result = torch.dot(a, b)

    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
