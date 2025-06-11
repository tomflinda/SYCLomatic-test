// ====------ asm_ld.cu ----------------------------------- *- CUDA -* ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <cuda_runtime.h>

#define TEST(FN)                                                               \
  {                                                                            \
    if (FN()) {                                                                \
      printf("Test " #FN " PASS\n");                                           \
    } else {                                                                   \
      printf("Test " #FN " FAIL\n");                                           \
      return 1;                                                                \
    }                                                                          \
  }

__device__ inline void load_global_short4(short4 &a, const short4 *addr) {
  short x, y, z, w;
  asm("ld.cg.global.v4.s16 {%0, %1, %2, %3}, [%4+0];"
      : "=h"(x), "=h"(y), "=h"(z), "=h"(w)
      : "l"(addr));
  a.x = x;
  a.y = y;
  a.z = z;
  a.w = w;
}

__global__ void test_kernel(short4 *d_out, const short4 *d_in) {
  short4 val;
  load_global_short4(val, d_in);
  *d_out = val;
}

__device__ inline void load_global_short2(short2 &a, const short2 *addr) {
  short x, y, z, w;
  asm("ld.cg.global.v2.s16 {%0, %1}, [%2+0];" : "=h"(x), "=h"(y) : "l"(addr));
  a.x = x;
  a.y = y;
}

__global__ void test_kernel(short2 *d_out, const short2 *d_in) {
  short2 val;
  load_global_short2(val, d_in);
  *d_out = val;
}

bool test_1() {
  short4 h_in = {1, 2, 3, 4};
  short4 h_out;
  short4 *d_in, *d_out;

  cudaMalloc(&d_in, sizeof(short4));
  cudaMalloc(&d_out, sizeof(short4));
  cudaMemcpy(d_in, &h_in, sizeof(short4), cudaMemcpyHostToDevice);

  test_kernel<<<1, 1>>>(d_out, d_in);
  cudaMemcpy(&h_out, d_out, sizeof(short4), cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);

  return (h_out.x == h_in.x && h_out.y == h_in.y && h_out.z == h_in.z &&
          h_out.w == h_in.w)
             ? true
             : false;
}

bool test_2() {
  short2 h_in = {1, 2};
  short2 h_out;
  short2 *d_in, *d_out;

  cudaMalloc(&d_in, sizeof(short2));
  cudaMalloc(&d_out, sizeof(short2));
  cudaMemcpy(d_in, &h_in, sizeof(short2), cudaMemcpyHostToDevice);

  test_kernel<<<1, 1>>>(d_out, d_in);
  cudaMemcpy(&h_out, d_out, sizeof(short2), cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);

  return (h_out.x == h_in.x && h_out.y == h_in.y) ? true : false;
}

__device__ __forceinline__ int ld_flag_volatile(int *flag_addr) {
  int flag;
  asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;"
               : "=r"(flag)
               : "l"(flag_addr));
  return flag;
}

__global__ void test_ld_flag_acquire(int *flag_addr, int *out_value) {
  int val = ld_flag_volatile(flag_addr);
  *out_value = val;
}

bool test_3() {

  int h_flag_value = 999;
  int h_result = 0;

  int *d_flag_addr;
  int *d_result;

  cudaMalloc(&d_flag_addr, sizeof(int));
  cudaMalloc(&d_result, sizeof(int));

  cudaMemcpy(d_flag_addr, &h_flag_value, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(d_result, 0, sizeof(int));

  test_ld_flag_acquire<<<1, 1>>>(d_flag_addr, d_result);
  cudaDeviceSynchronize();

  cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_flag_addr);
  cudaFree(d_result);

  return (h_result == h_flag_value) ? true : false;
}

int main() {
  TEST(test_1);
  TEST(test_2);
  TEST(test_3);
  return 0;
}