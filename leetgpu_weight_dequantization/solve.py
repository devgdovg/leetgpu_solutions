import torch
import triton
import triton.language as tl


@triton.jit
def dequant_kernel(x: tl.tensor, scale: tl.tensor, output: tl.tensor, m: int, n: int, TILE_SIZE: tl.constexpr):
    pid_0, pid_1 = tl.program_id(0), tl.program_id(1)
    x_ptr = tl.make_block_ptr(
        base=x,
        shape=(m, n),
        strides=(n, 1),
        offsets=(pid_0 * TILE_SIZE, pid_1 * TILE_SIZE),
        block_shape=(TILE_SIZE, TILE_SIZE),
        order=(1, 0),
    )
    scale_m = tl.cdiv(m, TILE_SIZE)
    scale_n = tl.cdiv(n, TILE_SIZE)
    scale_ptr = tl.make_block_ptr(
        base=scale,
        shape=(scale_m, scale_n),
        strides=(scale_n, 1),
        offsets=(pid_0, pid_1),
        block_shape=(1, 1),
        order=(1, 0),
    )
    x_tile = tl.load(x_ptr, boundary_check=(0, 1), padding_option="zero")
    scale_tile = tl.load(scale_ptr, boundary_check=(0, 1), padding_option="zero")
    output_tile = x_tile * scale_tile
    output_ptr = tl.make_block_ptr(
        base=output,
        shape=(m, n),
        strides=(n, 1),
        offsets=(pid_0 * TILE_SIZE, pid_1 * TILE_SIZE),
        block_shape=(TILE_SIZE, TILE_SIZE),
        order=(1, 0),
    )
    tl.store(output_ptr, output_tile, boundary_check=(0, 1))


# X, S, Y are tensors on the GPU
def solve(X: torch.Tensor, S: torch.Tensor, Y: torch.Tensor, M: int, N: int, TILE_SIZE: int):
    grid = (triton.cdiv(M, TILE_SIZE), triton.cdiv(N, TILE_SIZE))
    dequant_kernel[grid](X, S, Y, M, N, TILE_SIZE)


if __name__ == "__main__":
    M, N, TILE_SIZE = 8100, 8100, 16

    x = torch.randint(-10, 10, size=(M, N), device="cuda")
    s = torch.rand((triton.cdiv(M, TILE_SIZE), triton.cdiv(N, TILE_SIZE)), device="cuda", dtype=torch.float32)

    s_expanded = s.repeat_interleave(TILE_SIZE, dim=0).repeat_interleave(TILE_SIZE, dim=1)
    s_expanded = s_expanded[:M, :N]
    torch_result = x * s_expanded

    triton_result = torch.empty_like(x, device="cuda", dtype=torch.float32)
    solve(x, s, triton_result, M, N, TILE_SIZE)

    print(f"triton_result: {triton_result.shape}")
    print(f"torch_result: {torch_result.shape}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
