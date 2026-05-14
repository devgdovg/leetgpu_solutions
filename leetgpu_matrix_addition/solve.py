import torch
import triton
import triton.language as tl


@triton.jit
def matrix_add_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    x_start = pid_0 * BLOCK_SIZE
    y_start = pid_1 * BLOCK_SIZE
    range = tl.arange(0, BLOCK_SIZE)
    x_range = range + x_start
    y_range = range + y_start
    x_mask = x_range < n_elements
    y_mask = y_range < n_elements

    block_range = x_range[:, None] * n_elements + y_range[None, :]
    block_mask = x_mask[:, None] & y_mask[None, :]

    a_block = tl.load(a + block_range, block_mask, other=0)
    b_block = tl.load(b + block_range, block_mask, other=0)

    c_block = a_block + b_block

    tl.store(c + block_range, c_block, block_mask)


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 64
    n_elements = N
    G = triton.cdiv(n_elements, BLOCK_SIZE)
    grid = (G, G)
    matrix_add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE)


if __name__ == "__main__":
    n = 4096
    a = torch.randn((n, n), device="cuda", dtype=torch.float32)
    b = torch.randn((n, n), device="cuda", dtype=torch.float32)

    torch_result = a + b
    triton_result = torch.randn((n, n), device="cuda", dtype=torch.float32)
    solve(a, b, triton_result, n)

    # print(f"triton_result: {triton_result}, torch_result: {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
