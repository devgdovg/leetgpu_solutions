ncu --set full \
    --kernel-name-base function \
    --kernel-name "regex:w4a16_matmul_kernel" \
    --launch-skip 1 \
    --launch-count 1 \
    -o my_triton_report \
    --force-overwrite \
    python solve_3.py
