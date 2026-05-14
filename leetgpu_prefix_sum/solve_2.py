import torch
import triton
import triton.language as tl


@triton.jit
def stage_1_kernel(input: tl.tensor, output: tl.tensor, block_sums: tl.tensor, n: int, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_ptr = tl.make_block_ptr(
        base=input,
        shape=(n,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    block_data = tl.load(block_ptr, boundary_check=(0,), padding_option="zero")
    block_sum = tl.sum(block_data, axis=0, keep_dims=False)
    block_data = tl.cumsum(block_data)
    output_ptr = tl.make_block_ptr(
        base=output,
        shape=(n,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    tl.store(output_ptr, block_data, boundary_check=(0,))
    tl.store(block_sums + pid, block_sum)


@triton.jit
def stage_2_kernel(
    output: tl.tensor,
    block_sums: tl.tensor,
    n: int,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SUMS_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    output_ptr = tl.make_block_ptr(
        base=output,
        shape=(n,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    output_block = tl.load(output_ptr, boundary_check=(0,), padding_option="zero")
    block_sums_ptr = tl.arange(0, BLOCK_SUMS_SIZE)
    block_sums_data = tl.load(block_sums + block_sums_ptr, mask=block_sums_ptr < pid, other=0)
    output_block = output_block + tl.sum(block_sums_data, axis=0, keep_dims=False)
    tl.store(output_ptr, output_block, boundary_check=(0,))


# data and output are tensors on the GPU
def solve(data: torch.Tensor, output: torch.Tensor, n: int):
    BLOCK_SIZE = 1024
    block_count = triton.cdiv(n, BLOCK_SIZE)
    block_sums = torch.empty((block_count,), device="cuda", dtype=torch.float32)
    stage_1_kernel[(block_count,)](data, output, block_sums, n, BLOCK_SIZE)
    stage_2_kernel[(block_count,)](output, block_sums, n, BLOCK_SIZE, triton.next_power_of_2(block_count))


if __name__ == "__main__":
    N = 250_000

    x = torch.randn((N,), device="cuda", dtype=torch.float32)

    torch_result = torch.cumsum(x, dim=0)

    triton_result = torch.empty((N,), device="cuda", dtype=torch.float32)
    solve(x, triton_result, N)

    # print(f"triton_result: {triton_result}")
    # print(f"torch_result: {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
