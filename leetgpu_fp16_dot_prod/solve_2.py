import torch
import triton
import triton.language as tl


@triton.jit
def partial_sum_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    partial_sum: torch.Tensor,
    n: int,
    partial_sum_size: int,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    x_ptr = tl.make_block_ptr(
        base=x,
        shape=(n,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE_N,),
        block_shape=(BLOCK_SIZE_N,),
        order=(0,),
    )
    y_ptr = tl.make_block_ptr(
        base=y,
        shape=(n,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE_N,),
        block_shape=(BLOCK_SIZE_N,),
        order=(0,),
    )
    x_data = tl.load(x_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
    y_data = tl.load(y_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)

    partial_prod_sum = tl.sum(x_data * y_data, axis=0, keep_dims=False)

    partial_prod_sum_ptr = tl.make_block_ptr(
        base=partial_sum,
        shape=(partial_sum_size,),
        strides=(1,),
        offsets=(pid,),
        block_shape=(1,),
        order=(0,),
    )
    tl.store(partial_prod_sum_ptr, partial_prod_sum, boundary_check=(0,))


@triton.jit
def final_sum_kernel(
    partial_sum: torch.Tensor,
    mse: torch.Tensor,
    valid_partial_sum_size: int,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < valid_partial_sum_size
    data = tl.load(partial_sum + offsets, mask=mask, other=0.0)
    result = tl.sum(data, axis=0, keep_dims=False)
    mse_ptr = tl.make_block_ptr(
        base=mse,
        shape=(1,),
        strides=(1,),
        offsets=(0,),
        block_shape=(1,),
        order=(0,),
    )
    tl.store(mse_ptr, result.to(dtype=mse.dtype.element_ty), boundary_check=(0,))


# a, b, result are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, result: torch.Tensor, n: int):
    BLOCK_SIZE_N = 32768
    MAX_SIZE = 100_000_000

    partial_sum_size = triton.next_power_of_2(MAX_SIZE) // BLOCK_SIZE_N

    partial_sum = torch.zeros(partial_sum_size).to(dtype=torch.float32, device="cuda")

    grid = (triton.cdiv(n, BLOCK_SIZE_N),)

    partial_sum_kernel[grid](a, b, partial_sum, n, partial_sum_size, BLOCK_SIZE_N)

    valid_partial_sum_size = triton.cdiv(n, BLOCK_SIZE_N)

    final_sum_kernel[(1,)](partial_sum, result, valid_partial_sum_size, BLOCK_SIZE=partial_sum_size)


if __name__ == "__main__":
    x_len = 100_000_000
    x = torch.randn(x_len, dtype=torch.float16, device="cuda")
    y = (torch.rand_like(x) * 10.0).to(dtype=torch.float16, device="cuda")

    torch_result = torch.dot(x, y)

    triton_result = torch.empty((1,), dtype=torch.float16, device="cuda")
    solve(x, y, triton_result, x_len)

    print(f"triton_result: {triton_result}, torch_result: {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
