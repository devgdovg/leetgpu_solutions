import torch
import triton
import triton.language as tl


@triton.jit
def stream_compaction_kernel(input: tl.tensor, output: tl.tensor, n: int, BLOCK_SIZE: tl.constexpr):
    output_idx = 0
    MAX_INDEX = 999_999_999
    input_ptr = tl.make_block_ptr(
        base=input, shape=(n,), strides=(1,), offsets=(0,), block_shape=(BLOCK_SIZE,), order=(0,)
    )
    for _ in range(0, tl.cdiv(n, BLOCK_SIZE)):
        input_block = tl.load(input_ptr, boundary_check=(0,), padding_option="zero")
        indices = tl.arange(0, BLOCK_SIZE)
        indices = tl.sort(tl.where(input_block > 0, indices, MAX_INDEX), dim=0)
        compact_size = tl.sum(tl.where(indices == MAX_INDEX, 0, 1), axis=0, keep_dims=False)
        indices = tl.where(indices == MAX_INDEX, 0, indices)
        output_block = tl.gather(input_block, index=indices, axis=0)
        output_offsets = output_idx + tl.arange(0, BLOCK_SIZE)
        output_mask = output_offsets < output_idx + compact_size
        tl.store(output + output_offsets, output_block, mask=output_mask)
        output_idx += compact_size
        input_ptr = tl.advance(input_ptr, (BLOCK_SIZE,))


# A, out are tensors on the GPU
def solve(A: torch.Tensor, N: int, out: torch.Tensor):
    BLOCK_SIZE = 4096
    stream_compaction_kernel[(1,)](A, out, N, BLOCK_SIZE)


if __name__ == "__main__":
    x_len = 100_000_000
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
