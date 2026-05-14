import torch
import triton
import triton.language as tl


@triton.jit
def leaky_relu_kernel(input, output, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    range = tl.arange(0, BLOCK_SIZE) + start
    mask = range < n_elements

    block = tl.load(input + range, mask, other=0)
    tl.store(output + range, tl.where(block > 0, block, 0.01 * block), mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    leaky_relu_kernel[grid](input, output, N, BLOCK_SIZE)


if __name__ == "__main__":
    n = 50_000_000
    input = torch.randn((n,), device="cuda", dtype=torch.float32)

    import torch.nn.functional as F

    torch_result = F.leaky_relu(input, negative_slope=0.01)

    triton_result = torch.empty_like(input, device="cuda", dtype=torch.float32)
    solve(input, triton_result, n)

    # print(f"triton_result: {triton_result}, torch_result: {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
