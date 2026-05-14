import torch
import triton
import triton.language as tl


@triton.jit
def rgb_to_grayscale_kernel(input, output, convert, width, height, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    input_ptr = tl.make_block_ptr(
        base=input,
        shape=(width * height, 3),
        strides=(3, 1),
        offsets=(pid * BLOCK_SIZE, 0),
        block_shape=(BLOCK_SIZE, 4),
        order=(1, 0),
    )
    rgb = tl.load(input_ptr, boundary_check=(0, 1), padding_option="zero")
    convert_ptr = tl.make_block_ptr(
        base=convert,
        shape=(3,),
        strides=(1,),
        offsets=(0,),
        block_shape=(4,),
        order=(0,),
    )
    convert_scale = tl.load(convert_ptr, boundary_check=(0,), padding_option="zero")
    gray = tl.sum(rgb * convert_scale, axis=1, keep_dims=False)
    # gray = tl.dot(rgb, tl.reshape(convert_scale, (16, 1)), input_precision='ieee')
    # gray = tl.reshape(gray, (BLOCK_SIZE,))

    output_ptr = tl.make_block_ptr(
        base=output,
        shape=(width * height,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )

    tl.store(output_ptr, gray.to(dtype=output.dtype.element_ty), boundary_check=(0,))


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, width: int, height: int):
    total_pixels = width * height
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_pixels, BLOCK_SIZE),)
    convert = torch.tensor([[0.299, 0.587, 0.114]], device="cuda", dtype=torch.float32)
    rgb_to_grayscale_kernel[grid](input, output, convert, width, height, BLOCK_SIZE)


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
