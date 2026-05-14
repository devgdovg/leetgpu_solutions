import torch
import triton
import triton.language as tl


@triton.jit
def exp_sum_kernel(
    logits: tl.tensor,
    labels: tl.tensor,
    exp_sum: tl.tensor,
    n: int,
    c: int,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid = tl.program_id(0)
    logits_ptr = tl.make_block_ptr(
        base=logits,
        shape=(n, c),
        strides=(c, 1),
        offsets=(pid * BLOCK_SIZE_N, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_C),
        order=(1, 0),
    )
    offsets_n = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    labels_ptr = tl.make_block_ptr(
        base=labels,
        shape=(n,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE_N,),
        block_shape=(BLOCK_SIZE_N,),
        order=(0,),
    )
    labels_block = tl.load(labels_ptr, boundary_check=(0,), padding_option="zero")

    target_elem_ptr = offsets_n * c + labels_block
    target_elem_mask = (offsets_n < n) & (labels_block < c)
    target_elem = tl.load(logits + target_elem_ptr, mask=target_elem_mask, other=-float("inf"))
    target_elem = tl.reshape(target_elem, (BLOCK_SIZE_N, 1))

    accu = tl.zeros((BLOCK_SIZE_N, 1), dtype=tl.float32)
    max_elem = tl.full((BLOCK_SIZE_N, 1), value=-float("inf"), dtype=tl.float32)

    for j in range(0, tl.cdiv(c, BLOCK_SIZE_C)):
        logits_block = tl.load(logits_ptr, boundary_check=(0, 1), padding_option="zero")
        offsets_c = j * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
        boundary_mask = ((offsets_n < n)[:, None]) & ((offsets_c < c)[None, :])
        logits_block = tl.where(boundary_mask, logits_block, -float("inf"))
        new_max_elem = tl.maximum(max_elem, tl.max(logits_block, axis=1, keep_dims=True))
        accu = accu * tl.exp(max_elem - new_max_elem) + tl.sum(
            tl.exp(logits_block - new_max_elem), axis=1, keep_dims=True
        )
        max_elem = new_max_elem
        logits_ptr = tl.advance(logits_ptr, (0, BLOCK_SIZE_C))

    accu = tl.log(accu * tl.exp(max_elem - target_elem))

    exp_sum_ptr = tl.make_block_ptr(
        base=exp_sum,
        shape=(n,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE_N,),
        block_shape=(BLOCK_SIZE_N,),
        order=(0,),
    )
    tl.store(
        exp_sum_ptr,
        tl.reshape(accu, (BLOCK_SIZE_N,)).to(dtype=exp_sum.dtype.element_ty),
        boundary_check=(0,),
    )


@triton.jit
def avg_kernel(input: tl.tensor, output: tl.tensor, n: int, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    data = tl.load(input + offsets, mask=mask, other=0.0)
    result = tl.sum(data, axis=0, keep_dims=False) / n
    tl.store(output + tl.arange(0, 1), result.to(output.dtype.element_ty))


# logits, true_labels, loss are tensors on the GPU
def solve(logits: torch.Tensor, true_labels: torch.Tensor, loss: torch.Tensor, N: int, C: int):
    exp_sum = torch.empty((N,), device="cuda", dtype=torch.float32)
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_C = 64
    grid = (triton.cdiv(N, BLOCK_SIZE_N),)
    exp_sum_kernel[grid](logits, true_labels, exp_sum, N, C, BLOCK_SIZE_N, BLOCK_SIZE_C)

    avg_kernel[(1,)](exp_sum, loss, N, BLOCK_SIZE=triton.next_power_of_2(N))


if __name__ == "__main__":
    n, c = 10_000, 1_000
    logits = torch.randn((n, c), device="cuda", dtype=torch.float32)
    labels = torch.randint(low=0, high=c, size=(n,), device="cuda", dtype=torch.long)

    loss_fn = torch.nn.CrossEntropyLoss()
    torch_result = loss_fn(logits, labels)

    triton_result = torch.empty((1,), device="cuda", dtype=torch.float32)
    solve(logits, labels, triton_result, n, c)

    # print(f"triton_result: {triton_result}, torch_result: {torch_result}")
    print(f"AllClose: {torch.allclose(triton_result, torch_result)}")
    print(f"MaxDiff: {torch.max(torch.abs(triton_result - torch_result))}")

    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
