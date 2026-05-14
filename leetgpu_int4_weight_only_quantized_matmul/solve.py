import torch
import triton
import triton.language as tl

"""
Version 1
unpack & dequant的流程:
1. [load] 加载packed weight
2. [load] 加载scale
3. [r_shift] 得到even位置的一半unpacked权重(未减去offset)
4. [substract] 减去8, 得到even位置的一半unpacked权重(已减去offset)
5. [and] 得到odd位置的一半unpacked权重(未减去offset)
6. [substract] 减去8, 得到odd位置的一半unpacked权重(已减去offset)
7. [interleave] 得到完整的unpacked权重
8. [multiply] unpacked权重与scale相乘, 得到完整的dequant权重
"""


@triton.jit
def unpack(packed_int8_weight: tl.tensor, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    int_even_pos = (packed_int8_weight.to(dtype=tl.uint8) >> 4).to(dtype=tl.int8) - 8
    int_odd_pos = (packed_int8_weight.to(dtype=tl.uint8) & 0x0F).to(dtype=tl.int8) - 8
    return tl.interleave(int_even_pos, int_odd_pos).to(dtype=tl.int8)


@triton.jit
def dequant(
    unpacked_int8_weight: tl.tensor,
    scales: tl.tensor,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_G: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    s = tl.reshape(scales, (BLOCK_SIZE_N, BLOCK_SIZE_G, 1))
    s = tl.broadcast_to(s, (BLOCK_SIZE_N, BLOCK_SIZE_G, BLOCK_SIZE_K // BLOCK_SIZE_G))
    w = unpacked_int8_weight.to(tl.float32) * tl.reshape(s, (BLOCK_SIZE_N, BLOCK_SIZE_K)).to(tl.float32)
    return w


@triton.jit
def w4a16_matmul_kernel(
    x: tl.tensor,  # fp16
    wq: tl.tensor,  # int4 packed as uint8
    scales: tl.tensor,  # fp16
    y: tl.tensor,  # fp16
    m: int,
    n: int,
    k: int,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_G: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    x_ptr = tl.make_block_ptr(
        base=x,
        shape=(m, k),
        strides=(k, 1),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(0, 1),
    )

    wq_ptr = tl.make_block_ptr(
        base=wq,
        shape=(n, k >> 1),
        strides=(k >> 1, 1),
        offsets=(pid_n * BLOCK_SIZE_N, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K >> 1),
        order=(0, 1),
    )

    s_ptr = tl.make_block_ptr(
        base=scales,
        shape=(n, k // GROUP_SIZE),
        strides=(k // GROUP_SIZE, 1),
        offsets=(pid_n * BLOCK_SIZE_N, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_G),
        order=(0, 1),
    )

    accu = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for j in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
        x_block = tl.load(x_ptr, boundary_check=(0, 1), padding_option="zero")
        wq_block = tl.load(wq_ptr, boundary_check=(0, 1), padding_option="zero")
        unpacked_wq_block = unpack(wq_block, BLOCK_SIZE_N, BLOCK_SIZE_K)
        s_block = tl.load(s_ptr, boundary_check=(0, 1), padding_option="zero")
        w_block = dequant(unpacked_wq_block, s_block, BLOCK_SIZE_N, BLOCK_SIZE_G, BLOCK_SIZE_K)

        accu = tl.dot(x_block.to(tl.float32), tl.trans(w_block, (1, 0)), acc=accu, input_precision="ieee")

        x_ptr = tl.advance(x_ptr, (0, BLOCK_SIZE_K))
        wq_ptr = tl.advance(wq_ptr, (0, BLOCK_SIZE_K >> 1))
        if (((j + 1) * BLOCK_SIZE_K) % GROUP_SIZE) == 0:
            s_ptr = tl.advance(s_ptr, (0, BLOCK_SIZE_G))

    y_ptr = tl.make_block_ptr(
        base=y,
        shape=(m, n),
        strides=(n, 1),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(0, 1),
    )

    tl.store(y_ptr, accu.to(dtype=tl.float16), boundary_check=(0, 1))


# x, w_q, scales, y are tensors on the GPU
def solve(
    x: torch.Tensor,
    w_q: torch.Tensor,
    scales: torch.Tensor,
    y: torch.Tensor,
    M: int,
    N: int,
    K: int,
    group_size: int,
):
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 64

    if BLOCK_SIZE_K >= group_size:
        assert (BLOCK_SIZE_K % group_size) == 0
        BLOCK_SIZE_G = BLOCK_SIZE_K // group_size
    else:
        assert (group_size % BLOCK_SIZE_K) == 0
        BLOCK_SIZE_G = 1

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    w4a16_matmul_kernel[grid](
        x,
        w_q,
        scales,
        y,
        M,
        N,
        K,
        group_size,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        BLOCK_SIZE_G,
    )


if __name__ == "__main__":

    m, n, k = 4096, 4096, 4096
    group_size = 128
    x = torch.ones((m, k), device="cuda", dtype=torch.float16)
    wq = torch.ones((n, k // 2), device="cuda", dtype=torch.uint8)
    scales = torch.randn((n, k // group_size), device="cuda", dtype=torch.float16)

    unpacked_wq = torch.stack((wq >> 4, wq & 0x0F), dim=-1).reshape(n, -1).to(dtype=torch.int8)
    w_dequant = (unpacked_wq - 8) * (scales[:, :, None].broadcast_to(n, k // group_size, group_size).reshape(n, k))
    torch_result = x @ torch.transpose(w_dequant, 1, 0)

    triton_result = torch.empty((m, n), device="cuda", dtype=torch.float16)
    solve(x, wq, scales, triton_result, m, n, k, group_size)

    # print(f"torch_result:\n\t {torch_result}")
    # print(f"triton_result:\n\t {triton_result}")

    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
