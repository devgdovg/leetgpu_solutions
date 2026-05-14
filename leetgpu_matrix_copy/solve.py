import torch
import triton
import triton.language as tl


@triton.jit
def copy_kernel(a_ptr, b_ptr, N, B0: tl.constexpr, B1: tl.constexpr):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    x_start = pid_0 * B0
    x_range = tl.arange(0, B0) + x_start
    x_mask = x_range < N

    y_start = pid_1 * B1
    y_range = tl.arange(0, B1) + y_start
    y_mask = y_range < N

    range = x_range[:, None] * N + y_range[None, :]
    mask = x_mask[:, None] & y_mask[None, :]

    block = tl.load(a_ptr + range, mask, other=0)

    tl.store(b_ptr + range, block, mask)

    return


# a, b are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, N: int):

    B0 = 64
    B1 = 64

    grid = (triton.cdiv(N, B0), triton.cdiv(N, B1))

    copy_kernel[grid](a, b, N, B0, B1)
    return


if __name__ == "__main__":
    n = 4096
    input = torch.randn((n, n), device="cuda", dtype=torch.float32)

    torch_result = input.detach().clone()
    triton_result = torch.empty_like(input, device="cuda", dtype=torch.float32)
    solve(input, triton_result, n)

    # print(f"triton_result: {triton_result}, torch_result: {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
