"""
https://leetgpu.com/challenges/rotary-positional-embedding
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rope_kernel(
    q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    output: torch.Tensor,
    m: int,
    d: int,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_0, pid_1 = tl.program_id(0), tl.program_id(1)

    m_offsets = pid_0 * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    m_mask = m_offsets < m

    d_offsets = pid_1 * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    d_mask = d_offsets < d

    block_ptr = m_offsets[:, None] * d + d_offsets[None, :]
    block_mask = m_mask[:, None] & d_mask[None, :]

    rotated_d_offsets = (d_offsets + (d >> 1)) % d  # https://chatgpt.com/c/69e877f6-2b84-8323-b4a7-6c599a910e8a
    rotated_d_mask = rotated_d_offsets < d

    rotated_block_ptr = m_offsets[:, None] * d + rotated_d_offsets[None, :]
    rotated_block_mask = m_mask[:, None] & rotated_d_mask[None, :]

    q_block = tl.load(q + block_ptr, mask=block_mask, other=0.0)
    rotated_q_block = tl.load(q + rotated_block_ptr, mask=rotated_block_mask, other=0.0)
    cos_block = tl.load(cos + block_ptr, mask=block_mask, other=0.0)
    sin_block = tl.load(sin + block_ptr, mask=block_mask, other=0.0)

    rotated_q_block = tl.where(d_offsets < (d >> 1), -rotated_q_block, rotated_q_block)
    output_block = q_block * cos_block + rotated_q_block * sin_block

    tl.store(output + block_ptr, output_block.to(dtype=output.dtype.element_ty), mask=block_mask)


# Q, cos, sin, output are tensors on the GPU
def solve(Q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, output: torch.Tensor, M: int, D: int):
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_D = 128
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(D, BLOCK_SIZE_D))
    rope_kernel[grid](Q, cos, sin, output, M, D, BLOCK_SIZE_M, BLOCK_SIZE_D)
