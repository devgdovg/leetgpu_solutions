import torch
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(input, kernel, output, input_size, kernel_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    output_range = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    output_mask = output_range < (input_size - kernel_size + 1)

    accu = tl.zeros((BLOCK_SIZE, 1), dtype=tl.float32)

    for j in range(0, tl.cdiv(kernel_size, BLOCK_SIZE)):
        kernel_range = tl.arange(0, BLOCK_SIZE) + j * BLOCK_SIZE
        kernel_mask = kernel_range < kernel_size

        input_range = output_range[:, None] + kernel_range[None, :]
        input_mask = output_mask[:, None] & kernel_mask[None, :]

        inp = tl.load(input + input_range, input_mask, other=0)
        ker = tl.load(kernel + kernel_range, kernel_mask, other=0)

        accu += tl.sum(inp * ker, axis=1, keep_dims=True)

    tl.store(output + output_range, tl.reshape(accu, (BLOCK_SIZE,)), output_mask)


# input, kernel, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_size: int,
    kernel_size: int,
):
    BLOCK_SIZE = 64
    n_blocks = triton.cdiv(input_size - kernel_size + 1, BLOCK_SIZE)
    grid = (n_blocks,)

    conv1d_kernel[grid](input, kernel, output, input_size, kernel_size, BLOCK_SIZE)


if __name__ == "__main__":
    input_size = 1_500_000
    kernel_size = 2_047
    input = torch.randn((input_size,), device="cuda", dtype=torch.float32)
    kernel = torch.randn((kernel_size,), device="cuda", dtype=torch.float32)

    import torch.nn.functional as F

    torch_result = F.conv1d(input.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=0).squeeze()

    triton_result = torch.empty((input_size - kernel_size + 1,), device="cuda", dtype=torch.float32)
    solve(input, kernel, triton_result, input_size, kernel_size)

    # print(f"triton_result: {triton_result}, torch_result: {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
