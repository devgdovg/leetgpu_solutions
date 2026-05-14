import torch
import triton
import triton.language as tl


@triton.jit
def invert_kernel(image, width, height, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE * 4 + tl.arange(0, 4 * BLOCK_SIZE)
    mask = offsets < 4 * width * height
    data = tl.load(image + offsets, mask=mask, other=0)
    rgb = (offsets % 4) < 3
    data = tl.where(rgb, 255 - data, data)
    tl.store(image + offsets, data, mask=mask)


# image is a tensor on the GPU
def solve(image: torch.Tensor, width: int, height: int):
    BLOCK_SIZE = 1024
    n_pixels = width * height
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)

    invert_kernel[grid](image, width, height, BLOCK_SIZE)


if __name__ == "__main__":
    width, height = 4096, 5120
    image = torch.randint(0, 256, (height * width * 4,), device="cuda", dtype=torch.uint8)

    mask = (torch.arange(len(image), device="cuda") % 4) < 3
    torch_result = torch.where(mask, 255 - image, image)

    solve(image, width, height)
    triton_result = image

    # print(f"triton_result: {triton_result}, torch_result: {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
