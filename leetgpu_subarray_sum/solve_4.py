import torch
import triton
import triton.language as tl


@triton.autotune(configs=[triton.Config({}, num_warps=16)], key=["s", "e"])
@triton.jit
def _sum_kernel(
    input: tl.tensor,
    output: tl.tensor,
    s: int,
    e: int,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) + s
    end_mask = offsets <= e
    data = tl.load(input + offsets, mask=end_mask, other=0.0)
    block_sum = tl.sum(data, keep_dims=False)
    tl.atomic_add(output + tl.arange(0, 1), block_sum.to(dtype=output.dtype.element_ty))


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, S: int, E: int):

    BLOCK_SIZE_N = 16384

    grid = (triton.cdiv(E - S + 1, BLOCK_SIZE_N),)

    _sum_kernel[grid](
        input,
        output,
        S,
        E,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )


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
