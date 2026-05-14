import torch
import triton
import triton.language as tl


@triton.jit
def batch_norm_kernel(
    input: tl.tensor,
    gamma: tl.tensor,
    beta: tl.tensor,
    output: tl.tensor,
    n: int,
    c: int,
    eps: float,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid = tl.program_id(0)
    input_ptr = tl.make_block_ptr(
        base=input,
        shape=(n, c),
        strides=(c, 1),
        offsets=(0, pid * BLOCK_SIZE_C),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_C),
        order=(1, 0),
    )

    accu = tl.zeros((1, BLOCK_SIZE_C), dtype=tl.float32)
    accu_square = tl.zeros((1, BLOCK_SIZE_C), dtype=tl.float32)
    for _ in range(0, tl.cdiv(n, BLOCK_SIZE_N)):
        input_data = tl.load(input_ptr, boundary_check=(0, 1), padding_option="zero")
        accu = accu + tl.sum(input_data, axis=0, keep_dims=True)
        accu_square = accu_square + tl.sum(input_data * input_data, axis=0, keep_dims=True)
        input_ptr = tl.advance(input_ptr, (BLOCK_SIZE_N, 0))
    miu = accu / n
    std = accu_square / n - miu * miu

    accu = tl.zeros((1, BLOCK_SIZE_C), dtype=tl.float32)

    input_ptr = tl.make_block_ptr(
        base=input,
        shape=(n, c),
        strides=(c, 1),
        offsets=(0, pid * BLOCK_SIZE_C),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_C),
        order=(1, 0),
    )
    output_ptr = tl.make_block_ptr(
        base=output,
        shape=(n, c),
        strides=(c, 1),
        offsets=(0, pid * BLOCK_SIZE_C),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_C),
        order=(1, 0),
    )
    gamma_ptr = tl.make_block_ptr(
        base=gamma,
        shape=(c,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE_C,),
        block_shape=(BLOCK_SIZE_C,),
        order=(0,),
    )
    gamma_data = tl.load(gamma_ptr, boundary_check=(0,), padding_option="zero")
    beta_ptr = tl.make_block_ptr(
        base=beta,
        shape=(c,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE_C,),
        block_shape=(BLOCK_SIZE_C,),
        order=(0,),
    )
    beta_data = tl.load(beta_ptr, boundary_check=(0,), padding_option="zero")
    for _ in range(0, tl.cdiv(n, BLOCK_SIZE_N)):
        input_data = tl.load(input_ptr, boundary_check=(0, 1), padding_option="zero")
        norm_data = (input_data - miu) / tl.sqrt(std + eps)
        output_data = norm_data * gamma_data + beta_data
        tl.store(output_ptr, output_data.to(dtype=output.dtype.element_ty), boundary_check=(0, 1))
        input_ptr = tl.advance(input_ptr, (BLOCK_SIZE_N, 0))
        output_ptr = tl.advance(output_ptr, (BLOCK_SIZE_N, 0))


# input, gamma, beta, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    N: int,
    C: int,
    eps: float,
):
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_C = 32
    grid = (triton.cdiv(C, BLOCK_SIZE_C),)
    batch_norm_kernel[grid](
        input,
        gamma,
        beta,
        output,
        N,
        C,
        eps,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )


if __name__ == "__main__":
    N, C = 10_000, 1024

    input = torch.randn((N, C), device="cuda", dtype=torch.float32)
    gamma = torch.randn((C), device="cuda", dtype=torch.float32)
    beta = torch.randn((C), device="cuda", dtype=torch.float32)
    eps = 1e-5

    torch_fn = torch.nn.BatchNorm1d(C, eps=eps, affine=True, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        torch_fn.weight.copy_(gamma)
        torch_fn.bias.copy_(beta)

    torch_result = torch_fn(input)

    print("torch done.")

    triton_result = torch.empty_like(input)
    solve(input, gamma, beta, triton_result, N, C, eps)

    # print(f"triton_result: {triton_result}")
    # print(f"torch_result: {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
