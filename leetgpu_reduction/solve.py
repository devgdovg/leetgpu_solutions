"""
https://leetgpu.com/challenges/reduction
"""

import torch
import triton
import triton.language as tl


@triton.jit
def partial_sum_kernel(
    x: torch.Tensor,
    partial_sum: torch.Tensor,
    n: int,
    m: int,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    x_ptr = tl.make_block_ptr(
        base=x,
        shape=(n,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE_N,),
        block_shape=(BLOCK_SIZE_N,),
        order=(0,),
    )
    x_data = tl.load(x_ptr, boundary_check=(0,), padding_option="zero").to(dtype=tl.float32)
    x_sum = tl.sum(x_data, axis=0, keep_dims=False)
    partial_sum_ptr = tl.make_block_ptr(
        base=partial_sum,
        shape=(m,),
        strides=(1,),
        offsets=(pid,),
        block_shape=(1,),
        order=(0,),
    )
    tl.store(partial_sum_ptr, x_sum, boundary_check=(0,))


@triton.jit
def final_sum_kernel(input: torch.Tensor, output: torch.Tensor, valid_size: int, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < valid_size
    data = tl.load(input + offsets, mask=mask, other=0.0).to(dtype=tl.float32)
    output_ptr = tl.make_block_ptr(
        base=output,
        shape=(1,),
        strides=(1,),
        offsets=(0,),
        block_shape=(1,),
        order=(0,),
    )
    tl.store(
        output_ptr,
        tl.sum(data, axis=0, keep_dims=False).to(dtype=output.dtype.element_ty),
        boundary_check=(0,),
    )


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    MAX_SIZE = 100_000_000
    BLOCK_SIZE_N = 32768
    partial_sum_size = triton.next_power_of_2(MAX_SIZE) // BLOCK_SIZE_N
    partial_sum = torch.zeros(partial_sum_size, device=input.device, dtype=torch.float32)

    grid = (triton.cdiv(N, BLOCK_SIZE_N),)

    partial_sum_kernel[grid](
        x=input, partial_sum=partial_sum, n=N, m=partial_sum_size, BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=8
    )

    valid_partial_sum_size = triton.cdiv(N, BLOCK_SIZE_N)

    final_sum_kernel[(1,)](
        partial_sum,
        output,
        valid_size=valid_partial_sum_size,
        BLOCK_SIZE=partial_sum_size,
    )


if __name__ == "__main__":
    x_len = 100_000_000
    x = (torch.rand(x_len) * 2000 - 1000).to(dtype=torch.float32, device="cuda")

    torch_result = torch.sum(x, dim=0)

    triton_result = torch.empty((1,), dtype=torch.float32, device="cuda")
    solve(x, triton_result, x_len)

    print(f"triton_result: {triton_result}, torch_result: {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
