import torch
import triton
import triton.language as tl


@triton.jit
def subarray_sum_kernel(input: tl.tensor, output: tl.tensor, s: int, e: int, BLOCK_SIZE: tl.constexpr):
    accu = tl.zeros((1,), dtype=tl.float32)
    for j in range(0, tl.cdiv(e - s + 1, BLOCK_SIZE)):
        offsets = tl.arange(0, BLOCK_SIZE) + j * BLOCK_SIZE + s
        mask = offsets <= e
        data = tl.load(input + offsets, mask=mask, other=0.0)
        accu += tl.sum(data, keep_dims=False)

    tl.store(output + tl.arange(0, 1), accu.to(dtype=output.dtype.element_ty))


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, S: int, E: int):

    subarray_sum_kernel[(1,)](input, output, S, E, BLOCK_SIZE=1024)


if __name__ == "__main__":
    N, S, E = 100_000_000, 1_234_567, 87_654_321

    input = torch.randn((N,), device="cuda", dtype=torch.float32)

    torch_result = torch.sum(input[S : E + 1])

    triton_result = torch.zeros((1), device="cuda", dtype=torch.float32)
    solve(input, triton_result, N, S, E)

    # print(f"triton_result: {triton_result}")
    # print(f"torch_result: {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
