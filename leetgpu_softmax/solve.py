"""
https://leetgpu.com/challenges/softmax
"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(input: torch.Tensor, output: torch.Tensor, n: int, BLOCK_SIZE_N: tl.constexpr):

    input_ptr = tl.make_block_ptr(
        base=input,
        shape=(n,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_N,),
        order=(0,),
    )

    max_elem = -float("inf")
    bottom = 0.0

    for j in range(0, tl.cdiv(n, BLOCK_SIZE_N)):
        input_block = tl.load(input_ptr, boundary_check=(0,), padding_option="zero").to(dtype=tl.float32)
        offsets = j * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask = offsets < n
        input_block = tl.where(mask, input_block, -float("inf"))

        new_max_elem = tl.maximum(max_elem, tl.max(input_block, keep_dims=False))
        bottom = bottom * tl.exp(max_elem - new_max_elem) + tl.sum(tl.exp(input_block - new_max_elem), keep_dims=False)
        max_elem = new_max_elem

        input_ptr = tl.advance(input_ptr, offsets=(BLOCK_SIZE_N,))

    input_ptr = tl.make_block_ptr(
        base=input,
        shape=(n,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_N,),
        order=(0,),
    )

    output_ptr = tl.make_block_ptr(
        base=output,
        shape=(n,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_N,),
        order=(0,),
    )

    for _ in range(0, tl.cdiv(n, BLOCK_SIZE_N)):
        input_block = tl.load(input_ptr, boundary_check=(0,), padding_option="zero").to(dtype=tl.float32)
        output_block = tl.exp(input_block - max_elem) / bottom
        tl.store(output_ptr, output_block.to(dtype=output.dtype.element_ty), boundary_check=(0,))
        input_ptr = tl.advance(input_ptr, offsets=(BLOCK_SIZE_N,))
        output_ptr = tl.advance(output_ptr, offsets=(BLOCK_SIZE_N,))

    return


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    grid = (1,)
    BLOCK_SIZE_N = 2048
    softmax_kernel[grid](input, output, n=N, BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=8)


if __name__ == "__main__":
    len_x = 500_000
    x = (torch.rand(len_x) * 1000 + 500.0).to(device="cuda", dtype=torch.float32)

    import torch.nn.functional as F

    torch_result = F.softmax(x, dim=0)

    triton_result = torch.empty_like(x)
    solve(x, triton_result, len_x)

    print(f"AllClose: {torch.allclose(torch_result, triton_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(torch_result - triton_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
