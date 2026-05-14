# LeetGPU Solutions

This repository contains solution implementations for problems on [LeetGPU.com](https://leetgpu.com), intended for learning and research purposes.

## Overview

As of May 2026, the repository includes solutions for more than 50 problems, primarily written in Triton.

Discussions, improvements, and contributions for additional problems or further optimizations are welcome.

## Repository Structure

Each directory with the prefix `leetgpu_${problem_name}` corresponds to a specific problem on LeetGPU.com.

Within each directory, there may be one or more solution files.

- **`solve.py`**: A Triton-based solution implementation.
  - **`solve`** function: The entry point defined by the problem specification, responsible for invoking Triton kernels.
  - **`@triton.jit`** decorated funtions: Triton kernels for solving the problem.
  - **`if __name__ == "__main__"`** block: Local executable code for testing.

## How to Contribute

Coming soon.

## License

All rights reserved.

Unless otherwise specified, all files and assets in this repository are provided for educational and research purposes only.

Unauthorized commercial use, redistribution, and derivative works are prohibited.
