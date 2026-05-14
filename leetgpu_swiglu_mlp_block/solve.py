import torch
import triton
import triton.language as tl


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def swiglu_mlp_kernel(
    x: tl.tensor,
    gate: tl.tensor,
    up: tl.tensor,
    down: tl.tensor,
    output: tl.tensor,
    m: int,
    d_model: int,
    d_ffn: int,
    BLOCK_SIZE_OUT: tl.constexpr,
    BLOCK_SIZE_IN: tl.constexpr,
    BLOCK_SIZE_DMODEL: tl.constexpr,
):
    pid_0 = tl.program_id(0)  # along rows of x
    output_block = tl.zeros((BLOCK_SIZE_OUT, BLOCK_SIZE_DMODEL), dtype=tl.float32)
    for j in range(0, tl.cdiv(d_ffn, BLOCK_SIZE_OUT)):
        x_ptr = tl.make_block_ptr(
            base=x,
            shape=(m, d_model),
            strides=(d_model, 1),
            offsets=(pid_0 * BLOCK_SIZE_OUT, 0),
            block_shape=(BLOCK_SIZE_OUT, BLOCK_SIZE_IN),
            order=(1, 0),
        )
        gate_ptr = tl.make_block_ptr(
            base=gate,
            shape=(d_model, d_ffn),
            strides=(d_ffn, 1),
            offsets=(0, j * BLOCK_SIZE_OUT),
            block_shape=(BLOCK_SIZE_IN, BLOCK_SIZE_OUT),
            order=(1, 0),
        )
        up_ptr = tl.make_block_ptr(
            base=up,
            shape=(d_model, d_ffn),
            strides=(d_ffn, 1),
            offsets=(0, j * BLOCK_SIZE_OUT),
            block_shape=(BLOCK_SIZE_IN, BLOCK_SIZE_OUT),
            order=(1, 0),
        )
        x_gate_block = tl.zeros((BLOCK_SIZE_OUT, BLOCK_SIZE_OUT), dtype=tl.float32)
        x_up_block = tl.zeros((BLOCK_SIZE_OUT, BLOCK_SIZE_OUT), dtype=tl.float32)
        for _ in range(0, tl.cdiv(d_model, BLOCK_SIZE_IN)):
            x_block = tl.load(x_ptr, boundary_check=(0, 1), padding_option="zero")
            gate_block = tl.load(gate_ptr, boundary_check=(0, 1), padding_option="zero")
            up_block = tl.load(up_ptr, boundary_check=(0, 1), padding_option="zero")
            x_gate_block = tl.dot(x_block, gate_block, acc=x_gate_block, input_precision="ieee")
            x_up_block = tl.dot(x_block, up_block, acc=x_up_block, input_precision="ieee")
            x_ptr = tl.advance(x_ptr, (0, BLOCK_SIZE_IN))
            gate_ptr = tl.advance(gate_ptr, (BLOCK_SIZE_IN, 0))
            up_ptr = tl.advance(up_ptr, (BLOCK_SIZE_IN, 0))

        swiglu_block = silu(x_gate_block) * x_up_block

        down_ptr = tl.make_block_ptr(
            base=down,
            shape=(d_ffn, d_model),
            strides=(d_model, 1),
            offsets=(j * BLOCK_SIZE_OUT, 0),
            block_shape=(BLOCK_SIZE_OUT, BLOCK_SIZE_DMODEL),
            order=(1, 0),
        )
        down_block = tl.load(down_ptr, boundary_check=(0, 1), padding_option="zero")
        output_block = tl.dot(swiglu_block, down_block, acc=output_block, input_precision="ieee")

    output_ptr = tl.make_block_ptr(
        base=output,
        shape=(m, d_model),
        strides=(d_model, 1),
        offsets=(pid_0 * BLOCK_SIZE_OUT, 0),
        block_shape=(BLOCK_SIZE_OUT, BLOCK_SIZE_DMODEL),
        order=(1, 0),
    )
    tl.store(output_ptr, output_block, boundary_check=(0, 1))


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
    BLOCK_SIZE_OUT = 64
    BLOCK_SIZE_IN = 64

    grid = (triton.cdiv(M, BLOCK_SIZE_OUT),)

    swiglu_mlp_kernel[grid](
        x,
        W_gate,
        W_up,
        W_down,
        output,
        M,
        d_model,
        d_ffn,
        BLOCK_SIZE_OUT,
        BLOCK_SIZE_IN,
        BLOCK_SIZE_DMODEL=16 if d_model < 16 else triton.next_power_of_2(d_model),
        num_warps=8,
    )


if __name__ == "__main__":
    M, d_model, d_ffn = 512, 128, 14336
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
