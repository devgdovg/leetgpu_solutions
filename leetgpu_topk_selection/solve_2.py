import torch
import triton
import triton.language as tl


@triton.jit
def topk_by_block(input: tl.tensor, output: tl.tensor, n: int, k: int, BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(0)

    input_offsets = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    input_mask = input_offsets < n
    input_data = tl.load(input + input_offsets, mask=input_mask, other=-float("inf"))

    top_k = tl.sort(input_data, dim=0, descending=True)

    output_range = tl.arange(0, BLOCK_SIZE_N)
    output_offsets = pid * k + output_range
    output_mask = output_range < k
    tl.store(output + output_offsets, top_k, mask=output_mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, k: int):
    BLOCK_SIZE = 4096
    current_input = input
    current_input_size = N
    stop = False
    while not stop:
        block_count = triton.cdiv(current_input_size, BLOCK_SIZE)
        if block_count == 1:
            current_output = output
            stop = True
        else:
            current_output = torch.empty((block_count * k,), device=input.device, dtype=torch.float32)
        topk_by_block[(block_count,)](current_input, current_output, current_input_size, k, BLOCK_SIZE)
        current_input = current_output
        current_input_size = block_count * k


if __name__ == "__main__":
    N, k = 50_000_000, 100

    input = torch.randn((N,), device="cuda", dtype=torch.float32)

    torch_result = torch.topk(input, k, dim=0).values

    triton_result = torch.empty((k,), device="cuda", dtype=torch.float32)
    solve(input, triton_result, N, k)

    # print(f"triton_result: {triton_result}")
    # print(f"torch_result: {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
