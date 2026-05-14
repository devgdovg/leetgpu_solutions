import torch
import triton
import triton.language as tl


@triton.jit
def reverse_kernel(input, N, BLOCK_SIZE: tl.constexpr):
    pid_0 = tl.program_id(0)

    left_start = pid_0 * BLOCK_SIZE
    left_range = tl.arange(0, BLOCK_SIZE) + left_start
    left_mask = left_range < (N // 2)

    right_start = N - (pid_0 * BLOCK_SIZE) - BLOCK_SIZE
    right_range = tl.arange(0, BLOCK_SIZE) + right_start
    right_mask = right_range >= ((N + 1) // 2)

    l_block = tl.load(input + left_range, left_mask)
    r_block = tl.load(input + right_range, right_mask)

    tl.store(input + left_range, tl.flip(r_block, dim=0), left_mask)
    tl.store(input + right_range, tl.flip(l_block, dim=0), right_mask)


# input is a tensor on the GPU
def solve(input: torch.Tensor, N: int):
    BLOCK_SIZE = 128
    if N == 1:
        return
    n_blocks = triton.cdiv(N, BLOCK_SIZE * 2)
    grid = (n_blocks,)

    reverse_kernel[grid](input, N, BLOCK_SIZE)


if __name__ == "__main__":
    n = 25_000_000
    arr = torch.randn((n,), device="cuda", dtype=torch.float32)

    torch_result = torch.flip(arr, dims=(0,))

    solve(arr, n)
    triton_result = arr

    # print(f"triton_result: {triton_result}, torch_result: {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
