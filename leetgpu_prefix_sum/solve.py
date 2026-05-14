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
def stage_2_kernel(block_sums: tl.tensor, block_count: int, LOAD_RANGE: tl.constexpr):
    data_offsets = tl.arange(0, LOAD_RANGE)
    mask = data_offsets < block_count
    data = tl.load(block_sums + data_offsets, mask=mask, other=0)
    data = tl.cumsum(data) - data
    tl.store(block_sums + data_offsets, data, mask=mask)


@triton.jit
def stage_3_kernel(block_sums: tl.tensor, output: tl.tensor, n: int, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    output_ptr = tl.make_block_ptr(
        base=output,
        shape=(n,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    output_data = tl.load(output_ptr, boundary_check=(0,), padding_option="zero")
    output_data = output_data + tl.load(block_sums + pid)
    tl.store(output_ptr, output_data, boundary_check=(0,))


# data and output are tensors on the GPU
def solve(data: torch.Tensor, output: torch.Tensor, n: int):
    BLOCK_SIZE = 512
    BLOCK_COUNT = triton.cdiv(n, BLOCK_SIZE)
    block_sums = torch.empty((BLOCK_COUNT,), device="cuda", dtype=torch.float32)
    stage_1_kernel[(BLOCK_COUNT,)](data, output, block_sums, n, BLOCK_SIZE)
    stage_2_kernel[(1,)](block_sums, BLOCK_COUNT, triton.next_power_of_2(BLOCK_COUNT))
    stage_3_kernel[(BLOCK_COUNT,)](block_sums, output, n, BLOCK_SIZE)


if __name__ == "__main__":
    N = 22

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
