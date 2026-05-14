import torch
import triton
import triton.language as tl


@triton.jit
def kernel(A, B, C, D, BLOCK: tl.constexpr):
    a = tl.load(A + tl.arange(0, BLOCK))  # uint8 tensor
    b = tl.load(B + tl.arange(0, BLOCK))  # float32 tensor

    # For each (a,b) in zip(a,b), perform the following:
    # - Let ai be `a` converted to int32.
    # - Let af be `a` converted to float.
    # - Let m be the max of ai and b.
    # - Return ai and mi.
    # Do the above 4 elements at a time.
    (c, d) = tl.inline_asm_elementwise(
        asm="""
        {
            // Unpack `a` into `ai`.
            .reg .b8 tmp<4>;
            mov.b32 {tmp0, tmp1, tmp2, tmp3}, $8;
            cvt.u32.u8 $0, tmp0;
            cvt.u32.u8 $1, tmp1;
            cvt.u32.u8 $2, tmp2;
            cvt.u32.u8 $3, tmp3;
        }
        // Convert `ai` to float.
        cvt.rn.f32.s32 $4, $0;
        cvt.rn.f32.s32 $5, $1;
        cvt.rn.f32.s32 $6, $2;
        cvt.rn.f32.s32 $7, $3;
        // Take max of `ai` and `b`.
        max.f32 $4, $4, $9;
        max.f32 $5, $5, $10;
        max.f32 $6, $6, $11;
        max.f32 $7, $7, $12;
        """,
        constraints=(
            # 8 output registers, namely
            #   $0=ai0, $1=ai1, $2=ai2, $3=ai3,
            #   $4=m0,  $5=m1,  $6=m2,  $7=m3.
            "=r,=r,=r,=r,=r,=r,=r,=r,"
            # 5 input registers, namely
            #   $8=ai,
            #   $9=b0, $10=b1, $11=b2, $12=b3.
            # The four elements from `a` are all packed into one register.
            "r,r,r,r,r"
        ),
        args=[a, b],
        dtype=(tl.int32, tl.float32),
        is_pure=True,
        pack=4,
    )
    tl.store(C + tl.arange(0, BLOCK), c)
    tl.store(D + tl.arange(0, BLOCK), d)


if __name__ == "__main__":
    BLOCK = 16
    A = torch.randint(0, 256, (BLOCK,), device="cuda", dtype=torch.uint8)
    B = 100.0 * torch.randn((BLOCK,), device="cuda", dtype=torch.float32)
    C = torch.empty((BLOCK,), device="cuda", dtype=torch.uint32)
    D = torch.empty((BLOCK,), device="cuda", dtype=torch.float32)
    kernel[(1,)](A, B, C, D, BLOCK)

    print(f"A:\n\t {A}")
    print(f"B:\n\t {B}")
    print(f"C:\n\t {C}")
    print(f"D:\n\t {D}")
