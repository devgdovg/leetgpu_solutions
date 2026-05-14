import torch
import triton
import triton.language as tl


@triton.jit
def substream_compaction_kernel(
    input: tl.tensor,
    substreams: tl.tensor,
    substream_sizes: tl.tensor,
    n: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    MAX_INDEX = 999_999_999
    input_ptr = tl.make_block_ptr(
        base=input,
        shape=(n,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )

    input_block = tl.load(input_ptr, boundary_check=(0,), padding_option="zero")
    indices = tl.arange(0, BLOCK_SIZE)
    indices = tl.sort(tl.where(input_block > 0, indices, MAX_INDEX), dim=0)
    substream_size = tl.sum(tl.where(indices == MAX_INDEX, 0, 1), axis=0, keep_dims=False)
    indices = tl.where(indices == MAX_INDEX, 0, indices)
    substream_data = tl.gather(input_block, index=indices, axis=0)
    substream_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    substream_mask = tl.arange(0, BLOCK_SIZE) < substream_size
    tl.store(substreams + substream_offsets, substream_data, mask=substream_mask)
    tl.store(substream_sizes + pid, substream_size)


@triton.jit
def final_compact_kernel(
    substreams: tl.tensor,
    substream_sizes: tl.tensor,
    output: tl.tensor,
    n: int,
    BLOCK_SIZE: tl.constexpr,
):
    output_idx = tl.zeros((1,), dtype=tl.int32)
    substream_ptr = tl.make_block_ptr(
        base=substreams,
        shape=(tl.cdiv(n, BLOCK_SIZE), BLOCK_SIZE),
        strides=(BLOCK_SIZE, 1),
        offsets=(0, 0),
        block_shape=(1, BLOCK_SIZE),
        order=(1, 0),
    )
    substream_size_ptr = tl.make_block_ptr(
        base=substream_sizes,
        shape=(tl.cdiv(n, BLOCK_SIZE),),
        strides=(1,),
        offsets=(0,),
        block_shape=(1,),
        order=(0,),
    )
    for _ in range(0, tl.cdiv(n, BLOCK_SIZE)):
        substream_data = tl.load(substream_ptr, boundary_check=(0, 1), padding_option="zero")
        substream_size = tl.load(substream_size_ptr, boundary_check=(0,))
        output_offsets = output_idx + tl.arange(0, BLOCK_SIZE)
        output_mask = tl.arange(0, BLOCK_SIZE) < substream_size
        tl.store(output + output_offsets, tl.reshape(substream_data, (BLOCK_SIZE,)), mask=output_mask)
        output_idx += substream_size
        substream_ptr = tl.advance(substream_ptr, (1, 0))
        substream_size_ptr = tl.advance(substream_size_ptr, (1,))


# A, out are tensors on the GPU
def solve(A: torch.Tensor, N: int, out: torch.Tensor):
    BLOCK_SIZE = 8192
    substreams = torch.zeros((triton.cdiv(N, BLOCK_SIZE), BLOCK_SIZE), device=A.device, dtype=A.dtype)
    substream_sizes = torch.zeros((triton.cdiv(N, BLOCK_SIZE),), device=A.device, dtype=torch.int32)
    substream_compaction_kernel[(triton.cdiv(N, BLOCK_SIZE),)](
        A, substreams, substream_sizes, N, BLOCK_SIZE, num_warps=8
    )

    final_compact_kernel[(1,)](substreams, substream_sizes, out, N, BLOCK_SIZE, num_warps=8)


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
