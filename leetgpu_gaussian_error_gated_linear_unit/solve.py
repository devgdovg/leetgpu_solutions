import torch
import triton
import triton.language as tl


@triton.jit
def geglu(input, output, N, REV_SQRT_2: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    x1_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x2_offsets = x1_offsets + (N // 2)
    x1_mask = x1_offsets < (N // 2)
    x2_mask = x2_offsets < N

    x1 = tl.load(input + x1_offsets, mask=x1_mask, other=0.0)
    x2 = tl.load(input + x2_offsets, mask=x2_mask, other=0.0)

    result = x1 * 0.5 * x2 * (1 + tl.erf(x2 * REV_SQRT_2))
    tl.store(output + x1_offsets, result, mask=x1_mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N // 2, BLOCK_SIZE),)
    geglu[grid](input, output, N, 1.0 / 2**0.5, BLOCK_SIZE)


if __name__ == "__main__":
    n = 1_000_000
    x = torch.randn((n,), device="cuda", dtype=torch.float32)

    import torch.nn.functional as F

    def torch_geglu(a):
        a1, a2 = a.chunk(2, dim=-1)
        return a1 * F.gelu(a2)

    torch_result = torch_geglu(x)

    triton_result = torch.empty((n // 2,), device="cuda", dtype=torch.float32)
    solve(x, triton_result, n)

    # print(f"triton_result:\n\t {triton_result}")
    # print(f"torch_result:\n\t {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
