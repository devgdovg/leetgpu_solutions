import torch
import triton
import triton.language as tl


@triton.jit
def sigmoid_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    mask = offsets < n_elements
    x_block = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(dtype=tl.float32)
    y_block = 1.0 / (tl.exp(-x_block) + 1.0)
    tl.store(y_ptr + offsets, y_block.to(dtype=y_ptr.dtype.element_ty), mask=mask)
    pass


# X, Y are tensors on the GPU
def solve(X: torch.Tensor, Y: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    sigmoid_kernel[grid](X, Y, N, BLOCK_SIZE)


if __name__ == "__main__":
    x_len = 100_000_000
    x = torch.rand(x_len, device="cuda", dtype=torch.float32)

    torch_result = torch.sigmoid(x)

    triton_result = torch.empty_like(x)
    solve(x, triton_result, x_len)

    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
