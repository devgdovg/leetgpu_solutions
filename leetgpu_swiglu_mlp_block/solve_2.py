import torch
import triton
import triton.language as tl


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def fused_matmul_swiglu_kernel(
    x: tl.tensor,
    y1: tl.tensor,
    y2: tl.tensor,
    output: tl.tensor,
    m: int,
    n: int,
    k: int,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SWIZZLE_GROUP_SIZE: tl.constexpr,
    FUSE_SWIGLU: tl.constexpr,
):
    # pid = tl.program_id(0)
    # col_block_count = tl.cdiv(n, BLOCK_SIZE_N)
    # pid_m, pid_n = pid // col_block_count, pid % col_block_count
    # pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, tl.cdiv(m, BLOCK_SIZE_M), col_block_count, SWIZZLE_GROUP_SIZE)

    pid_m, pid_n = tl.program_id(0), tl.program_id(1)

    x_ptr = tl.make_block_ptr(
        base=x,
        shape=(m, k),
        strides=(k, 1),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )
    y1_ptr = tl.make_block_ptr(
        base=y1,
        shape=(k, n),
        strides=(n, 1),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(1, 0),
    )
    x_y1_accu = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if FUSE_SWIGLU == 1:
        y2_ptr = tl.make_block_ptr(
            base=y2,
            shape=(k, n),
            strides=(n, 1),
            offsets=(0, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
            order=(1, 0),
        )
        x_y2_accu = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
        x_block = tl.load(x_ptr, boundary_check=(0, 1), padding_option="zero")
        y1_block = tl.load(y1_ptr, boundary_check=(0, 1), padding_option="zero")
        x_y1_accu = tl.dot(x_block, y1_block, acc=x_y1_accu, input_precision="ieee")
        if FUSE_SWIGLU == 1:
            y2_block = tl.load(y2_ptr, boundary_check=(0, 1), padding_option="zero")
            x_y2_accu = tl.dot(x_block, y2_block, acc=x_y2_accu, input_precision="ieee")
        x_ptr = tl.advance(x_ptr, (0, BLOCK_SIZE_K))
        y1_ptr = tl.advance(y1_ptr, (BLOCK_SIZE_K, 0))
        if FUSE_SWIGLU == 1:
            y2_ptr = tl.advance(y2_ptr, (BLOCK_SIZE_K, 0))

    if FUSE_SWIGLU == 1:
        x_y1_accu = x_y1_accu * silu(x_y2_accu)

    output_ptr = tl.make_block_ptr(
        base=output,
        shape=(m, n),
        strides=(n, 1),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )

    tl.store(output_ptr, x_y1_accu, boundary_check=(0, 1))


# x, W_gate, W_up, W_down, output are tensors on the GPU
def solve(
    x: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    output: torch.Tensor,
    M: int,
    d_model: int,
    d_ffn: int,
):
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 64
    SWIZZLE_GROUP_SIZE = 8

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(d_ffn, BLOCK_SIZE_N))

    swiglu_output = torch.empty((M, d_ffn), device="cuda", dtype=torch.float32)
    fused_matmul_swiglu_kernel[grid](
        x,
        W_up,
        W_gate,
        swiglu_output,
        M,
        d_ffn,
        d_model,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        SWIZZLE_GROUP_SIZE,
        FUSE_SWIGLU=1,
    )

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(d_model, BLOCK_SIZE_N))
    DUMMY_TENSOR = torch.empty((0,), device="cuda", dtype=torch.float32)
    fused_matmul_swiglu_kernel[grid](
        swiglu_output,
        W_down,
        DUMMY_TENSOR,
        output,
        M,
        d_model,
        d_ffn,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        SWIZZLE_GROUP_SIZE,
        FUSE_SWIGLU=0,
    )


if __name__ == "__main__":
    M, d_model, d_ffn = 512, 4096, 8192
    # M, d_model, d_ffn = 32, 32, 32

    x = torch.randn((M, d_model), device="cuda", dtype=torch.float32)
    w_gate = torch.randn((d_model, d_ffn), device="cuda", dtype=torch.float32)
    w_up = torch.randn((d_model, d_ffn), device="cuda", dtype=torch.float32)
    w_down = torch.randn((d_ffn, d_model), device="cuda", dtype=torch.float32)

    import torch.nn.functional as F

    swiglu = F.silu(torch.matmul(x, w_gate)) * torch.matmul(x, w_up)
    torch_result = torch.matmul(swiglu, w_down)

    triton_result = torch.randn((M, d_model), device="cuda", dtype=torch.float32)
    solve(x, w_gate, w_up, w_down, triton_result, M, d_model, d_ffn)

    # print(f"triton_result: {triton_result}")
    # print(f"torch_result: {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
