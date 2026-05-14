import torch
import triton
import triton.language as tl


@triton.jit
def partial_topk_kernel(
    input: tl.tensor,
    partial_result: tl.tensor,
    n: int,
    k: int,
    BLOCK_SIZE: tl.constexpr,
    K_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    input_ptr = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    input_mask = input_ptr < n
    input_block = tl.load(input + input_ptr, mask=input_mask, other=-float("inf"))
    topk = tl.topk(input_block, K_BLOCK_SIZE, dim=0)

    partial_result_ptr = tl.make_block_ptr(
        base=partial_result,
        shape=(tl.cdiv(n, BLOCK_SIZE), k),
        strides=(k, 1),
        offsets=(pid, 0),
        block_shape=(1, K_BLOCK_SIZE),
        order=(1, 0),
    )
    tl.store(partial_result_ptr, tl.reshape(topk, (1, K_BLOCK_SIZE)), boundary_check=(0, 1))


@triton.jit
def final_topk_kernel(
    partial_result: tl.tensor,
    final_result: tl.tensor,
    n: int,
    k: int,
    LOAD_BLOCK_SIZE: tl.constexpr,
    STORE_BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, LOAD_BLOCK_SIZE)
    mask = offsets < n
    data = tl.load(partial_result + offsets, mask=mask, other=-float("inf"))
    topk = tl.topk(data, STORE_BLOCK_SIZE, dim=0)
    output_offsets = tl.arange(0, STORE_BLOCK_SIZE)
    output_mask = output_offsets < k
    tl.store(final_result + output_offsets, topk, mask=output_mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, k: int):

    BLOCK_SIZE = 16384

    grid = triton.cdiv(N, BLOCK_SIZE)

    partial_result = torch.empty((grid, k), device="cuda", dtype=torch.float32)

    partial_topk_kernel[(grid,)](input, partial_result, N, k, BLOCK_SIZE, triton.next_power_of_2(k))

    final_topk_kernel[(1,)](partial_result, output, N, k, triton.next_power_of_2(grid * k), triton.next_power_of_2(k))


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
