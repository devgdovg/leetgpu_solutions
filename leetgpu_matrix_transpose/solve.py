import torch
import triton
import triton.language as tl


@triton.jit
def matrix_transpose_kernel(
    input,
    output,
    rows,
    cols,
    stride_ir,
    stride_ic,
    stride_or,
    stride_oc,
    B0: tl.constexpr,
    B1: tl.constexpr,
):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    x_start = pid_0 * B0
    x_range = tl.arange(0, B0) + x_start
    x_mask = x_range < rows

    y_start = pid_1 * B1
    y_range = tl.arange(0, B1) + y_start
    y_mask = y_range < cols

    block = tl.load(
        input + (x_range[:, None] * stride_ir + y_range[None, :]),
        x_mask[:, None] & y_mask[None, :],
        other=0,
    )

    tl.store(
        output + (y_range[:, None] * stride_or + x_range[None, :]),
        tl.trans(block, (1, 0)),
        y_mask[:, None] & x_mask[None, :],
    )


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    stride_ir, stride_ic = cols, 1
    stride_or, stride_oc = rows, 1

    B0, B1 = 64, 64

    grid = (triton.cdiv(rows, B0), triton.cdiv(cols, B1))
    matrix_transpose_kernel[grid](input, output, rows, cols, stride_ir, stride_ic, stride_or, stride_oc, B0, B1)


if __name__ == "__main__":
    rows, cols = 7000, 6000
    input = torch.randn((rows, cols), device="cuda", dtype=torch.float16)

    torch_result = torch.transpose(input, 1, 0)

    triton_result = torch.empty((cols, rows), device="cuda", dtype=torch.float16)
    solve(input, triton_result, rows, cols)

    # print(f"triton_result: {triton_result}, torch_result: {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
