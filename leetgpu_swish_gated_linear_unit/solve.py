import torch
import triton
import triton.language as tl


@triton.jit
def swiglu_kernel(input, output, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    x1_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x1_mask = x1_offsets < N // 2
    x2_offsets = x1_offsets + (N // 2)
    x2_mask = x2_offsets < N

    x1 = tl.load(input + x1_offsets, mask=x1_mask, other=0)
    x2 = tl.load(input + x2_offsets, mask=x2_mask, other=0)

    silu = x1 / (1 + tl.exp(-x1))
    swiglu = silu * x2

    ouput_offsets = x1_offsets
    output_mask = x1_mask

    tl.store(output + ouput_offsets, swiglu, mask=output_mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N // 2, BLOCK_SIZE),)
    swiglu_kernel[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE)


if __name__ == "__main__":
    N = 100_000

    input = torch.randn((N,), device="cuda", dtype=torch.float32)

    x1, x2 = input[: N // 2], input[N // 2 :]

    import torch.nn.functional as F

    torch_result = F.silu(x1) * x2

    triton_result = torch.empty((N // 2,), device="cuda", dtype=torch.float32)
    solve(input, triton_result, N)

    # print(f"triton_result: {triton_result}")
    # print(f"torch_result: {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
