import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    input: torch.Tensor,
    gamma: float,
    beta: float,
    output: torch.Tensor,
    n: int,
    eps: float,
    LENGTH: tl.constexpr,
):
    offsets = tl.arange(0, LENGTH)
    mask = offsets < n
    data = tl.load(input + offsets, mask, other=0.0).to(dtype=tl.float32)
    rms = tl.sqrt(tl.sum(data * data, axis=0, keep_dims=False) / n + eps)
    data = (data / rms) * gamma + beta
    tl.store(output + offsets, data.to(dtype=output.dtype.element_ty), mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, gamma: float, beta: float, output: torch.Tensor, N: int, eps: float):
    grid = (1,)
    rms_norm_kernel[grid](input, gamma, beta, output, N, eps, LENGTH=triton.next_power_of_2(N), num_warps=8)


def torch_rmsnorm(input: torch.Tensor, n: int, eps: float, gamma: float, beta: float):
    rms_norm = nn.RMSNorm(n, eps=eps, elementwise_affine=True, device=input.device, dtype=input.dtype)
    new_weights = torch.full_like(input, gamma)
    with torch.no_grad():
        rms_norm.weight.copy_(new_weights)
    return rms_norm(input) + beta


if __name__ == "__main__":
    x_len = 100_000
    x = torch.randn(x_len, dtype=torch.float32, device="cuda")
    eps = 1e-5
    gamma = 10.5
    beta = 0.5

    torch_result = torch_rmsnorm(x, x_len, eps, gamma, beta)
    triton_result = torch.empty_like(x, device="cuda", dtype=torch.float32)
    solve(x, gamma, beta, triton_result, x_len, eps)

    print(f"AllClose: {torch.allclose(torch_result, triton_result)}")
    print(f"MaxDiffAbs: {torch.max(torch.abs(torch_result - triton_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10_000)
