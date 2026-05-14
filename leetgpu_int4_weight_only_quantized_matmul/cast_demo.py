import torch
import triton
import triton.language as tl


@triton.jit
def cast_kernel(input: tl.tensor, output: tl.tensor, BLOCK_SIZE: tl.constexpr):  # int32
    input_ptr = tl.arange(0, BLOCK_SIZE)
    input_data = tl.load(input + input_ptr)

    output_data = input_data.cast(dtype=tl.float16, bitcast=True)
    output_ptr = tl.arange(0, BLOCK_SIZE * 2)

    tl.store(output + output_ptr, output_data)


if __name__ == "__main__":

    input = torch.full((16,), fill_value=0x64006400, device="cuda", dtype=torch.uint32)
    output = torch.zeros((32,), device="cuda", dtype=torch.float16)

    cast_kernel[(1,)](input, output, 16)

    print(f"output:\n\t {output}")
    print("Ctrl+C to exit")
    import time

    time.sleep(10000)
