import torch
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input: tl.tensor,
    kernel: tl.tensor,
    output: tl.tensor,
    input_rows: int,
    input_cols: int,
    kernel_rows: int,
    kernel_cols: int,
    output_rows: int,
    output_cols: int,
    KERNEL_ROW_DIM: tl.constexpr,
    KERNEL_COL_DIM: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BATCH_SIZE + tl.arange(0, BATCH_SIZE)
    row_ids, col_ids = offsets // output_cols, offsets % output_cols

    kr = tl.arange(0, KERNEL_ROW_DIM)
    kc = tl.arange(0, KERNEL_COL_DIM)

    row_offsets = row_ids[:, None] + kr[None, :]
    col_offsets = col_ids[:, None] + kc[None, :]
    img_ptr = row_offsets[:, :, None] * input_cols + col_offsets[:, None, :]
    img_mask = (row_offsets < input_rows)[:, :, None] & (col_offsets < input_cols)[:, None, :]

    img_block = tl.load(input + img_ptr, mask=img_mask, other=0.0)
    img_flatten = tl.reshape(img_block, (BATCH_SIZE, KERNEL_ROW_DIM * KERNEL_COL_DIM))

    kernel_ptr = kr[:, None] * kernel_cols + kc[None, :]
    kernel_mask = (kr < kernel_rows)[:, None] & (kc < kernel_cols)[None, :]

    kernel_block = tl.load(kernel + kernel_ptr, mask=kernel_mask, other=0.0)
    kernel_flatten = tl.reshape(kernel_block, (KERNEL_ROW_DIM * KERNEL_COL_DIM, 1))

    result = tl.dot(img_flatten, kernel_flatten, input_precision="ieee")
    output_mask = offsets < output_rows * output_cols
    tl.store(
        output + offsets,
        result.reshape((BATCH_SIZE,)).to(dtype=output.dtype.element_ty),
        mask=output_mask,
    )


# input, kernel, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_rows: int,
    input_cols: int,
    kernel_rows: int,
    kernel_cols: int,
):
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1
    BATCH_SIZE = 32
    grid = (triton.cdiv(output_rows * output_cols, BATCH_SIZE),)

    kernel_row_dim = triton.next_power_of_2(kernel_rows)
    if kernel_row_dim < 4:
        kernel_row_dim = 4

    kernel_col_dim = triton.next_power_of_2(kernel_cols)
    if kernel_col_dim < 4:
        kernel_col_dim = 4

    conv2d_kernel[grid](
        input,
        kernel,
        output,
        input_rows,
        input_cols,
        kernel_rows,
        kernel_cols,
        output_rows,
        output_cols,
        KERNEL_ROW_DIM=kernel_row_dim,
        KERNEL_COL_DIM=kernel_col_dim,
        BATCH_SIZE=BATCH_SIZE,
    )


if __name__ == "__main__":
    input_row, input_col, kernel_row, kernel_col = 3072, 3072, 15, 15
    output_row, output_col = input_row - kernel_row + 1, input_col - kernel_col + 1
    x = torch.randn((input_row, input_col), device="cuda", dtype=torch.float32)
    y = torch.randn((kernel_row, kernel_col), device="cuda", dtype=torch.float32)

    import torch.nn.functional as F

    torch_result = F.conv2d(x.unsqueeze(0).unsqueeze(0), y.unsqueeze(0).unsqueeze(0)).squeeze()

    triton_result = torch.empty((output_row, output_col), device="cuda", dtype=torch.float32)
    solve(x, y, triton_result, input_row, input_col, kernel_row, kernel_col)

    # print(f"triton_result: {triton_result}, torch_result: {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
