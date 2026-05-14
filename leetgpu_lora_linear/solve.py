import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    x: tl.tensor,
    y1: tl.tensor,
    y2: tl.tensor,
    output1: tl.tensor,
    output2: tl.tensor,
    scale: float,
    accu: tl.constexpr,
    m: int,
    n1: int,
    n2: int,
    k: int,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_0, pid_1 = tl.program_id(0), tl.program_id(1)

    if pid_1 < tl.cdiv(n1, BLOCK_SIZE_N):
        n = n1
        y = y1
        pid_y = pid_1
        output = output1
    else:
        n = n2
        y = y2
        pid_y = pid_1 - tl.cdiv(n1, BLOCK_SIZE_N)
        output = output2

    x_ptr = tl.make_block_ptr(
        base=x,
        shape=(m, k),
        strides=(k, 1),
        offsets=(pid_0 * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )

    y_ptr = tl.make_block_ptr(
        base=y,
        shape=(n, k),
        strides=(k, 1),
        offsets=(pid_y * BLOCK_SIZE_N, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
        order=(1, 0),
    )

    output_ptr = tl.make_block_ptr(
        base=output,
        shape=(m, n),
        strides=(n, 1),
        offsets=(pid_0 * BLOCK_SIZE_M, pid_y * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )

    result = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
        x_block = tl.load(x_ptr, boundary_check=(0, 1), padding_option="zero")
        y_block = tl.load(y_ptr, boundary_check=(0, 1), padding_option="zero")
        result = tl.dot(x_block, tl.trans(y_block, (1, 0)), input_precision="ieee", acc=result)
        # result = tl.dot(x_block, tl.trans(y_block, (1, 0)), acc=result)
        x_ptr = tl.advance(x_ptr, (0, BLOCK_SIZE_K))
        y_ptr = tl.advance(y_ptr, (0, BLOCK_SIZE_K))
    if scale != 1.0:
        result = result * scale

    if accu == 1:
        orig = tl.load(output_ptr, boundary_check=(0, 1), padding_option="zero")
        result = result + orig

    tl.store(output_ptr, result, boundary_check=(0, 1))


# x, W, A, B, output are tensors on the GPU
def solve(
    x: torch.Tensor,
    W: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    output: torch.Tensor,
    batch: int,
    d_in: int,
    d_out: int,
    rank: int,
    lora_scale: float,
):
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 32, 64, 32
    lora_down_result = torch.empty((batch, rank), device=x.device, dtype=x.dtype)
    grid_1 = (
        triton.cdiv(batch, BLOCK_SIZE_M),
        triton.cdiv(d_out, BLOCK_SIZE_N) + triton.cdiv(rank, BLOCK_SIZE_N),
    )

    matmul_kernel[grid_1](
        x,
        W,
        A,
        output,
        lora_down_result,
        1.0,
        0,
        batch,
        d_out,
        rank,
        d_in,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )

    DUMMY_TENSOR = torch.empty((batch, d_out), device="cuda", dtype=torch.float32)
    grid_2 = (triton.cdiv(batch, BLOCK_SIZE_M), triton.cdiv(d_out, BLOCK_SIZE_N))
    matmul_kernel[grid_2](
        lora_down_result,
        B,
        DUMMY_TENSOR,
        output,
        DUMMY_TENSOR,
        lora_scale,
        1,
        batch,
        d_out,
        d_out,
        rank,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )


if __name__ == "__main__":
    batch = 2
    d_in = 4
    d_out = 3
    rank = 2
    scale = 0.5

    x = torch.randn((batch, d_in), device="cuda", dtype=torch.float32)
    w = torch.randn((d_out, d_in), device="cuda", dtype=torch.float32)
    lora_a = torch.randn((rank, d_in), device="cuda", dtype=torch.float32)
    lora_b = torch.randn((d_out, rank), device="cuda", dtype=torch.float32)

    torch_result = torch.matmul(x, w.transpose(1, 0)) + scale * torch.matmul(
        torch.matmul(x, lora_a.transpose(1, 0)), lora_b.transpose(1, 0)
    )

    triton_result = torch.empty((batch, d_out), device="cuda", dtype=torch.float32)
    solve(x, w, lora_a, lora_b, triton_result, batch, d_in, d_out, rank, scale)

    print(f"triton_result: {triton_result.shape}")
    print(f"torch_result: {torch_result.shape}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")
