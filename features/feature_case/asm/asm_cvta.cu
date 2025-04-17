// ====------ asm_cvta.cu ---------------------------------- *- CUDA -* ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define TEST(FN)                                                               \
  {                                                                            \
    if (FN()) {                                                                \
      printf("Test " #FN " PASS\n");                                           \
    } else {                                                                   \
      printf("Test " #FN " FAIL\n");                                           \
      return 1;                                                                \
    }                                                                          \
  }

__global__ void read_shared_value(int *output) {
  __shared__ int shared_data[1]; // Shared memory allocation

  if (threadIdx.x == 0) {
    shared_data[0] = 42;
  }
  __syncthreads();

  unsigned long long shared_addr_u64;
  int value;

  asm volatile(
      "cvta.to.shared.u64 %0, %2;\n\t" // Properly uses input operand %2
      "ld.shared.u32 %1, [%0];\n\t"    // Correctly assigns to output %1
      : "=l"(shared_addr_u64), "=r"(value)
      : "l"(shared_data));

  if (threadIdx.x == 0) {
    output[0] = value;
  }
}

bool cvta_to_shared_u64_test() {
  int *d_output;
  int h_output = 0;

  // Allocate device memory
  cudaMalloc(&d_output, sizeof(int));

  // Launch the kernel
  read_shared_value<<<1, 1>>>(d_output);

  // Copy result back to host
  cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

  bool passed = true;

  if (h_output != 42) {
    passed = false;
    std::cout << "Read value from shared memory: " << h_output << std::endl;
  }

  cudaFree(d_output);
  return passed;
}

int main() {
  TEST(cvta_to_shared_u64_test);
  return 0;
}
