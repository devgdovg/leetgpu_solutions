import torch
import triton
import triton.language as tl


@triton.jit
def gaussian_blur_kernel(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_rows: int,
    input_cols: int,
    kernel_rows: int,
    kernel_cols: int,
    BATCH_SIZE: tl.constexpr,
    SPAN_ROW_BLOCK_SIZE: tl.constexpr,
    SPAN_COL_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    pixels = pid * BATCH_SIZE + tl.arange(0, BATCH_SIZE)
    pixel_rows, pixel_cols = pixels // input_cols, pixels % input_cols

    span_r = tl.arange(0, SPAN_ROW_BLOCK_SIZE) - (kernel_rows >> 1)
    mask_span_r = span_r <= (kernel_rows >> 1)
    pixel_row_span = pixel_rows[:, None] + span_r[None, :]
    pixel_row_mask = pixel_row_span >= 0
    pixel_row_mask = pixel_row_mask & (pixel_row_span < input_rows)
    pixel_row_mask = pixel_row_mask & mask_span_r

    span_c = tl.arange(0, SPAN_COL_BLOCK_SIZE) - (kernel_cols >> 1)
    mask_span_c = span_c <= (kernel_cols >> 1)
    pixel_col_span = pixel_cols[:, None] + span_c[None, :]
    pixel_col_mask = pixel_col_span >= 0
    pixel_col_mask = pixel_col_mask & (pixel_col_span < input_cols)
    pixel_col_mask = pixel_col_mask & mask_span_c

    pixel_span = pixel_row_span[:, :, None] * input_cols + pixel_col_span[:, None, :]
    pixel_mask = pixel_row_mask[:, :, None] & pixel_col_mask[:, None, :]

    pixels_data = tl.load(input + pixel_span, mask=pixel_mask, other=0)

    kernel_ptr = tl.make_block_ptr(
        base=kernel,
        shape=(kernel_rows, kernel_cols),
        strides=(kernel_cols, 1),
        offsets=(0, 0),
        block_shape=(SPAN_ROW_BLOCK_SIZE, SPAN_COL_BLOCK_SIZE),
        order=(1, 0),
    )
    kernel_data = tl.load(kernel_ptr, boundary_check=(0, 1), padding_option="zero")
    blurred = tl.sum(tl.sum(pixels_data * kernel_data, axis=2, keep_dims=False), axis=1, keep_dims=False)

    tl.store(output + pixels, tl.reshape(blurred, (BATCH_SIZE,)), mask=pixels < input_rows * input_cols)


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
    BATCH_SIZE = 32
    grid = triton.cdiv(input_rows * input_cols, BATCH_SIZE)

    gaussian_blur_kernel[(grid,)](
        input,
        kernel,
        output,
        input_rows,
        input_cols,
        kernel_rows,
        kernel_cols,
        BATCH_SIZE,
        triton.next_power_of_2(kernel_rows),
        triton.next_power_of_2(kernel_cols),
    )


if __name__ == "__main__":
    input_row, input_col, kernel_row, kernel_col = 512, 512, 7, 7
    x = torch.randn((input_row, input_col), device="cuda", dtype=torch.float32)
    kernel = torch.randn((kernel_row, kernel_col), device="cuda", dtype=torch.float32)

    import torch.nn.functional as F

    torch_result = F.conv2d(
        x.unsqueeze(0).unsqueeze(0),
        kernel.unsqueeze(0).unsqueeze(0),
        padding=(kernel_row // 2, kernel_col // 2),
    ).squeeze()

    # print(f"torch_result:\n\t {torch_result}")

    triton_result = torch.empty((input_row, input_col), device="cuda", dtype=torch.float32)
    solve(x, kernel, triton_result, input_row, input_col, kernel_row, kernel_col)

    # print(f"triton_result:\n\t {triton_result}")

    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
