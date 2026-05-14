import torch
import triton
import triton.language as tl


@triton.jit
def substream_size_kernel(input: tl.tensor, substream_sizes: tl.tensor, n: int, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    input_ptr = tl.make_block_ptr(
        base=input,
        shape=(n,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    input_block = tl.load(input_ptr, boundary_check=(0,), padding_option="zero")
    substream_size = tl.sum(input_block > 0)
    tl.store(substream_sizes + pid, substream_size)


@triton.jit
def substream_offset_kernel(substream_sizes: tl.tensor, n: int, BLOCK_SIZE: tl.constexpr):
    r = tl.arange(0, BLOCK_SIZE)
    m = r < n
    data = tl.load(substream_sizes + r, mask=m, other=0)
    data = tl.cumsum(data) - data
    tl.store(substream_sizes + r, data, mask=m)


@triton.jit
def final_compact_kernel(
    input: tl.tensor,
    substream_offsets: tl.tensor,
    output: tl.tensor,
    n: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    input_ptr = tl.make_block_ptr(
        base=input,
        shape=(n,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    input_block = tl.load(input_ptr, boundary_check=(0,), padding_option="zero")
    substream_offset = tl.load(substream_offsets + pid)
    pos = input_block > 0
    input_offsets = tl.cumsum(pos) - pos
    output_offsets = input_offsets + substream_offset
    output_mask = pos & (output_offsets < n)
    tl.store(output + output_offsets, input_block, mask=output_mask)


# A, out are tensors on the GPU
def solve(A: torch.Tensor, N: int, out: torch.Tensor):
    BLOCK_SIZE = 2048
    grid = triton.cdiv(N, BLOCK_SIZE)

    substream_sizes = torch.empty((grid,), device=A.device, dtype=torch.int32)
    substream_size_kernel[(grid,)](A, substream_sizes, N, BLOCK_SIZE)

    substream_offset_kernel[(1,)](substream_sizes, grid, triton.next_power_of_2(grid))

    final_compact_kernel[(grid,)](A, substream_sizes, out, N, BLOCK_SIZE)


if __name__ == "__main__":
    x_len = 50_000_000
    x_len_non_zeros = 35_823_949
    # x_len = 40
    # x_len_non_zeros = 28
    x_len_zeros = x_len - x_len_non_zeros

    x_zero = torch.zeros((x_len_zeros,), device="cuda", dtype=torch.float32)
    x_non_zero = torch.randn((x_len_non_zeros,), device="cuda", dtype=torch.float32)
    x = torch.concat((x_zero, x_non_zero), dim=0)
    x = x[torch.randperm(x.size(0))]

    # print(f"input:\n {x}")

    torch_result = torch.concat((x[x > 0], torch.zeros_like(x[x <= 0])), dim=0)

    triton_result = torch.zeros_like(x, device="cuda", dtype=torch.float32)
    solve(x, x_len, triton_result)

    # print(f"torch_result:\n {torch_result}")
    # print(f"triton_result:\n {triton_result}")

    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
