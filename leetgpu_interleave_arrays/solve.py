import torch
import triton
import triton.language as tl


@triton.jit
def interleave_kernel(A_ptr, B_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a_block = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b_block = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    out_a_offsets = offsets * 2
    out_b_offsets = out_a_offsets + 1
    tl.store(output_ptr + out_a_offsets, a_block, mask=out_a_offsets < 2 * N)
    tl.store(output_ptr + out_b_offsets, b_block, mask=out_b_offsets < 2 * N)


# A, B, output are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    interleave_kernel[grid](A, B, output, N, BLOCK_SIZE=BLOCK_SIZE)


if __name__ == "__main__":
    N = 50_000_000

    a = torch.randn((N,), device="cuda", dtype=torch.float32)
    b = torch.randn((N,), device="cuda", dtype=torch.float32)

    torch_result = torch.stack((a, b), dim=1).flatten()

    triton_result = torch.empty((2 * N,), device="cuda", dtype=torch.float32)
    solve(a, b, triton_result, N)

    # print(f"triton_result: {triton_result}")
    # print(f"torch_result: {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
