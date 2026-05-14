import torch
import triton
import triton.language as tl


@triton.jit
def _sum_kernel(
    input: tl.tensor,
    output: tl.tensor,
    n: int,
    m: int,
    s_row: int,
    e_row: int,
    s_col: int,
    e_col: int,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    pid_0, pid_1 = tl.program_id(0), tl.program_id(1)

    offset_row = s_row + pid_0 * BLOCK_SIZE_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    mask_row = offset_row <= e_row

    offset_col = s_col + pid_1 * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)
    mask_col = offset_col <= e_col

    offsets = offset_row[:, None] * m + offset_col[None, :]
    mask = mask_row[:, None] & mask_col[None, :]

    data = tl.load(input + offsets, mask=mask, other=0.0).to(dtype=tl.float32)

    block_sum = tl.sum(tl.sum(data, axis=1, keep_dims=False), keep_dims=False)

    tl.atomic_add(output + tl.arange(0, 1), block_sum.to(dtype=output.dtype.element_ty))


# input, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    output: torch.Tensor,
    N: int,
    M: int,
    S_ROW: int,
    E_ROW: int,
    S_COL: int,
    E_COL: int,
):
    BLOCK_SIZE_ROW, BLOCK_SIZE_COL = 64, 64
    grid = (
        triton.cdiv(E_ROW - S_ROW + 1, BLOCK_SIZE_ROW),
        triton.cdiv(E_COL - S_COL + 1, BLOCK_SIZE_COL),
    )

    _sum_kernel[grid](
        input,
        output,
        N,
        M,
        S_ROW,
        E_ROW,
        S_COL,
        E_COL,
        BLOCK_SIZE_ROW=BLOCK_SIZE_ROW,
        BLOCK_SIZE_COL=BLOCK_SIZE_COL,
    )


if __name__ == "__main__":
    N, M = 10_000, 10_000
    S_ROW, E_ROW = 123, 8765
    S_COL, E_COL = 234, 9876

    input = 1 + 9 * torch.rand((N, M), device="cuda", dtype=torch.float32)

    torch_result = torch.sum(input[S_ROW : E_ROW + 1, S_COL : E_COL + 1])

    triton_result = torch.zeros((1), device="cuda", dtype=torch.float32)
    solve(input, triton_result, N, M, S_ROW, E_ROW, S_COL, E_COL)

    print(f"triton_result: {triton_result}")
    print(f"torch_result: {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
