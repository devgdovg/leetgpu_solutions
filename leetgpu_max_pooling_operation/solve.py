import torch
import triton
import triton.language as tl


@triton.jit
def max_pooling_kernel(
    input: tl.tensor,
    output: tl.tensor,
    input_size_h: int,
    input_size_w: int,
    output_size_h: int,
    output_size_w: int,
    kernel_size: int,
    stride: int,
    padding: int,
    BLOCK_SIZE_KERNEL: tl.constexpr,
    BLOCK_SIZE_OUTPUT: tl.constexpr,
):
    pid = tl.program_id(0)
    slice = tl.program_id(1)

    output_offset = pid * BLOCK_SIZE_OUTPUT + tl.arange(0, BLOCK_SIZE_OUTPUT)
    output_idx_h = output_offset // output_size_w
    output_idx_w = output_offset % output_size_w

    input_idx_h = output_idx_h[:, None] * stride + (tl.arange(0, BLOCK_SIZE_KERNEL) - padding)[None, :]
    input_idx_w = output_idx_w[:, None] * stride + (tl.arange(0, BLOCK_SIZE_KERNEL) - padding)[None, :]

    input_ptrs = input_idx_h[:, :, None] * input_size_w + input_idx_w[:, None, :]
    input_mask_h = (input_idx_h >= 0) & (input_idx_h < input_size_h)
    input_mask_w = (input_idx_w >= 0) & (input_idx_w < input_size_w)
    input_mask = input_mask_h[:, :, None] & input_mask_w[:, None, :]

    kernel_mask = tl.arange(0, BLOCK_SIZE_KERNEL) < kernel_size
    kernel_mask = kernel_mask[:, None] & kernel_mask[None, :]

    input_ptrs = input_ptrs + slice * input_size_h * input_size_w
    input_data = tl.load(input + input_ptrs, mask=input_mask & kernel_mask, other=-float("inf"))

    output_data = tl.max(
        tl.reshape(input_data, (BLOCK_SIZE_OUTPUT, BLOCK_SIZE_KERNEL * BLOCK_SIZE_KERNEL)),
        axis=1,
        keep_dims=False,
    )

    output_mask = output_offset < output_size_h * output_size_w
    output_ptr = output_offset + slice * output_size_h * output_size_w
    tl.store(output + output_ptr, output_data.to(dtype=output.dtype.element_ty), mask=output_mask)


# input, output are tensors on the GPU
def solve(input, output, N, C, H, W, kernel_size, stride, padding):
    if (H + 2 * padding < kernel_size) or (W + 2 * padding < kernel_size):
        # illegal
        return

    output_size_h = (H + 2 * padding - kernel_size) // stride + 1
    output_size_w = (W + 2 * padding - kernel_size) // stride + 1

    BLOCK_SIZE_OUTPUT = 1024
    grid = (
        triton.cdiv(output_size_h * output_size_w, BLOCK_SIZE_OUTPUT),
        N * C,
    )
    max_pooling_kernel[grid](
        input,
        output,
        H,
        W,
        output_size_h,
        output_size_w,
        kernel_size,
        stride,
        padding,
        BLOCK_SIZE_KERNEL=triton.next_power_of_2(kernel_size),
        BLOCK_SIZE_OUTPUT=BLOCK_SIZE_OUTPUT,
    )


if __name__ == "__main__":
    N, C, H, W = 4, 16, 1024, 1024
    kernel_size, stride, padding = 3, 2, 1

    input = torch.randn((N, C, H, W), device="cuda", dtype=torch.float32)
    # input = torch.arange(0, N*C*H*W, device="cuda", dtype=torch.float32).reshape((N, C, H, W))
    # print(f"input: {input}")

    torch_fn = torch.nn.MaxPool2d(kernel_size, stride, padding=padding)
    torch_result = torch_fn(input)

    output_size_h = (H + 2 * padding - kernel_size) // stride + 1
    output_size_w = (W + 2 * padding - kernel_size) // stride + 1
    triton_result = torch.ones((N, C, output_size_h, output_size_w), device="cuda", dtype=torch.float32)
    solve(input, triton_result, N, C, H, W, kernel_size, stride, padding)

    # print(f"triton_result: {triton_result}")
    # print(f"torch_result: {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
