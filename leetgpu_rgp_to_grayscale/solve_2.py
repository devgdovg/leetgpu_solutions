import torch
import triton
import triton.language as tl


@triton.jit
def rgb_to_grayscale_kernel(input, output, width, height, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    output_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = output_offsets < width * height
    input_offsets = output_offsets * 3
    r = tl.load(input + input_offsets, mask=mask, other=0.0)
    g = tl.load(input + input_offsets + 1, mask=mask, other=0.0)
    b = tl.load(input + input_offsets + 2, mask=mask, other=0.0)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    tl.store(output + output_offsets, gray.to(dtype=output.dtype.element_ty), mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, width: int, height: int):
    total_pixels = width * height
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_pixels, BLOCK_SIZE),)
    rgb_to_grayscale_kernel[grid](input, output, width, height, BLOCK_SIZE)


if __name__ == "__main__":
    rgb = torch.tensor(
        [255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 128.0, 128.0, 128.0],
        device="cuda",
        dtype=torch.float32,
    )
    height, width = 2, 2
    output = torch.empty((height * width,), device="cuda", dtype=torch.float32)

    solve(rgb, output, width, height)

    print(output)
