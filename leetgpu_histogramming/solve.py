import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(
    input: tl.tensor,
    output: tl.tensor,
    n: int,
    num_bins: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    data = tl.load(input + offsets, mask=mask, other=-1)
    for p in range(pid, pid + num_bins):
        bin = p % num_bins
        bin_count = tl.sum(data == bin)
        tl.atomic_add(output + bin, bin_count)


# input, histogram are tensors on the GPU
def solve(input: torch.Tensor, histogram: torch.Tensor, N: int, num_bins: int):
    BLOCK_SIZE = 2048
    histogram_kernel[(triton.cdiv(N, BLOCK_SIZE),)](input, histogram, N, num_bins, BLOCK_SIZE)


if __name__ == "__main__":
    N, num_bins = 50_000_000, 256

    import numpy as np

    arr = np.random.randint(num_bins, size=N)
    counts, _ = np.histogram(arr, num_bins)

    torch_result = torch.from_numpy(counts).to(device="cuda", dtype=torch.int32)

    input = torch.from_numpy(arr).to(device="cuda", dtype=torch.int32)
    triton_result = torch.zeros((num_bins,), device="cuda", dtype=torch.int32)
    solve(input, triton_result, N, num_bins)

    # print(f"triton_result: {triton_result}")
    # print(f"torch_result: {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
