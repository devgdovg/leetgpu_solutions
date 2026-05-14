import torch
import triton
import triton.language as tl


@triton.jit
def partial_sum_kernel(
    input: tl.tensor,
    partial_result: tl.tensor,
    n: int,
    s: int,
    e: int,
    partial_result_size: int,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_mask = offsets < n
    subarray_start_mask = offsets >= s
    subarray_end_mask = offsets <= e
    mask = offsets_mask & subarray_start_mask & subarray_end_mask
    data = tl.load(input + offsets, mask=mask, other=0.0)
    block_sum = tl.sum(data, keep_dims=False)
    partial_result_ptr = tl.make_block_ptr(
        base=partial_result,
        shape=(partial_result_size,),
        strides=(1,),
        offsets=(pid,),
        block_shape=(1,),
        order=(0,),
    )
    tl.store(partial_result_ptr, block_sum.to(dtype=partial_result.dtype.element_ty), boundary_check=(0,))


@triton.jit
def final_sum_kernel(partial_result: tl.tensor, output: tl.tensor, partial_result_size: int, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < partial_result_size
    data = tl.load(partial_result + offsets, mask=mask, other=0.0)
    final_sum = tl.sum(data, keep_dims=False)
    tl.store(output + tl.arange(0, 1), final_sum.to(dtype=output.dtype.element_ty))


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, S: int, E: int):

    BLOCK_SIZE_N = 16384

    partial_result_size = triton.cdiv(N, BLOCK_SIZE_N)
    grid = (partial_result_size,)
    partial_result = torch.zeros((partial_result_size,), device="cuda", dtype=torch.float32)

    partial_sum_kernel[grid](
        input,
        partial_result,
        N,
        S,
        E,
        partial_result_size,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    final_sum_kernel[(1,)](
        partial_result,
        output,
        partial_result_size,
        BLOCK_SIZE=triton.next_power_of_2(partial_result_size),
    )


if __name__ == "__main__":
    N, S, E = 5, 1, 3

    input = torch.randn((N,), device="cuda", dtype=torch.float32)

    torch_result = torch.sum(input[S : E + 1])

    triton_result = torch.zeros((1), device="cuda", dtype=torch.float32)
    solve(input, triton_result, N, S, E)

    # print(f"triton_result: {triton_result}")
    # print(f"torch_result: {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
