import torch
import triton
import triton.language as tl


@triton.jit
def clip_kernel(input, output, lo, hi, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    _offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    _mask = _offsets < N
    input_block = tl.load(input + _offsets, mask=_mask, other=0)
    output_block = tl.where(input_block > lo, input_block, lo)
    output_block = tl.where(output_block < hi, output_block, hi)
    tl.store(output + _offsets, output_block, mask=_mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, lo: float, hi: float, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    clip_kernel[grid](input, output, lo, hi, N, BLOCK_SIZE=BLOCK_SIZE)


if __name__ == "__main__":
    N = 100_000

    x = torch.randn((N,), device="cuda", dtype=torch.float32)
    low = -0.2
    high = 0.4

    torch_result = torch.clamp(x, low, high)

    triton_result = torch.empty((N,), device="cuda", dtype=torch.float32)
    solve(x, triton_result, low, high, N)

    # print(f"triton_result: {triton_result}")
    # print(f"torch_result: {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
