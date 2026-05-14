import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(0)

    start = pid * BLOCK_SIZE
    range = tl.arange(0, BLOCK_SIZE) + start
    mask = range < n_elements

    aa = tl.load(a + range, mask)
    bb = tl.load(b + range, mask)

    tl.store(c + range, aa + bb, mask)


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE)
