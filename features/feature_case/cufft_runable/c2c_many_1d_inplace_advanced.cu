// ===--- c2c_many_1d_inplace_advanced.cu --------------------*- CUDA -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===---------------------------------------------------------------------===//

#include "cufft.h"
#include "cufftXt.h"
#include "common.h"
#include <cstring>
#include <iostream>



// forward
// input
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
// |   c   |   0   |   c   |   0   |   c   |   0   |   0   |   0   |   0   |   c   |   0   |   c   |   0   |   c   |   0   |   0   |   0   |   0   |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
// |________________________n______________________|               |       |________________________n______________________|               |       |
// |_____________________________nembed____________________________|       |_____________________________nembed____________________________|       |
// |___________________________________batch0______________________________|___________________________________batch1______________________________|
// output
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
// |   c   |   0   |   c   |   0   |   c   |   0   |   0   |   0   |   0   |   c   |   0   |   c   |   0   |   c   |   0   |   0   |   0   |   0   |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
// |________________________n______________________|               |       |________________________n______________________|               |       |
// |_____________________________nembed____________________________|       |_____________________________nembed____________________________|       |
// |___________________________________batch0______________________________|___________________________________batch1______________________________|
bool c2c_many_1d_inplace_advanced() {
  cufftHandle plan_fwd;
  cufftCreate(&plan_fwd);
  float2 forward_idata_h[18];
  std::memset(forward_idata_h, 0, sizeof(float2) * 18);
  forward_idata_h[0] =  float2{0, 1};
  forward_idata_h[2] =  float2{2, 3};
  forward_idata_h[4] =  float2{4, 5};
  forward_idata_h[9] =  float2{0, 1};
  forward_idata_h[11] = float2{2, 3};
  forward_idata_h[13] = float2{4, 5};

  float2* data_d;
  cudaMalloc(&data_d, sizeof(float2) * 18);
  cudaMemcpy(data_d, forward_idata_h, sizeof(float2) * 18, cudaMemcpyHostToDevice);

  size_t workSize;
  long long int n[1] = {3};
  long long int inembed[1] = {4};
  long long int onembed[1] = {4};
  cufftMakePlanMany64(plan_fwd, 1, n, inembed, 2, 9, onembed, 2, 9, CUFFT_C2C, 2, &workSize);
  cufftExecC2C(plan_fwd, data_d, data_d, CUFFT_FORWARD);
  cudaDeviceSynchronize();
  float2 forward_odata_h[18];
  cudaMemcpy(forward_odata_h, data_d, sizeof(float2) * 18, cudaMemcpyDeviceToHost);

  float2 forward_odata_ref[18];
  forward_odata_ref[0] =  float2{6,9};
  forward_odata_ref[1] =  float2{2,3};
  forward_odata_ref[2] =  float2{-4.73205,-1.26795};
  forward_odata_ref[3] =  float2{0,1};
  forward_odata_ref[4] =  float2{-1.26795,-4.73205};
  forward_odata_ref[5] =  float2{4,5};
  forward_odata_ref[6] =  float2{0,0};
  forward_odata_ref[7] =  float2{0,0};
  forward_odata_ref[8] =  float2{0,0};
  forward_odata_ref[9] =  float2{6,9};
  forward_odata_ref[10] = float2{0,0};
  forward_odata_ref[11] = float2{-4.73205,-1.26795};
  forward_odata_ref[12] = float2{0,0};
  forward_odata_ref[13] = float2{-1.26795,-4.73205};
  forward_odata_ref[14] = float2{0,0};
  forward_odata_ref[15] = float2{0,0};
  forward_odata_ref[16] = float2{0,0};
  forward_odata_ref[17] = float2{0,0};

  cufftDestroy(plan_fwd);

  std::vector<int> indices = {0, 2, 4,
                              9, 11, 13};
  if (!compare(forward_odata_ref, forward_odata_h, indices)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, indices);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, indices);

    cudaFree(data_d);

    return false;
  }

  cufftHandle plan_bwd;
  cufftCreate(&plan_bwd);
  cufftMakePlanMany64(plan_bwd, 1, n, onembed, 2, 9, inembed, 2, 9, CUFFT_C2C, 2, &workSize);
  cufftExecC2C(plan_bwd, data_d, data_d, CUFFT_INVERSE);
  cudaDeviceSynchronize();
  float2 backward_odata_h[18];
  cudaMemcpy(backward_odata_h, data_d, sizeof(float2) * 18, cudaMemcpyDeviceToHost);

  float2 backward_odata_ref[18];
  backward_odata_ref[0] =  float2{0, 3};
  backward_odata_ref[1] =  float2{0, 0};
  backward_odata_ref[2] =  float2{6, 9};
  backward_odata_ref[3] =  float2{0, 0};
  backward_odata_ref[4] =  float2{12, 15};
  backward_odata_ref[5] =  float2{0, 0};
  backward_odata_ref[6] =  float2{0, 0};
  backward_odata_ref[7] =  float2{0, 0};
  backward_odata_ref[8] =  float2{0, 0};
  backward_odata_ref[9] =  float2{0, 3};
  backward_odata_ref[10] = float2{0, 0};
  backward_odata_ref[11] = float2{6, 9};
  backward_odata_ref[12] = float2{0, 0};
  backward_odata_ref[13] = float2{12, 15};
  backward_odata_ref[14] = float2{0, 0};
  backward_odata_ref[15] = float2{0, 0};
  backward_odata_ref[16] = float2{0, 0};
  backward_odata_ref[17] = float2{0, 0};

  cudaFree(data_d);
  cufftDestroy(plan_bwd);

  std::vector<int> bwd_indices = {0, 2, 4,
                                9, 11, 13};
  if (!compare(backward_odata_ref, backward_odata_h, bwd_indices)) {
    std::cout << "backward_odata_h:" << std::endl;
    print_values(backward_odata_h, bwd_indices);
    std::cout << "backward_odata_ref:" << std::endl;
    print_values(backward_odata_ref, bwd_indices);
    return false;
  }
  return true;
}


#ifdef DEBUG_FFT
int main() {
#define FUNC c2c_many_1d_inplace_advanced
  bool res = FUNC();
  cudaDeviceSynchronize();
  if (!res) {
    std::cout << "Fail" << std::endl;
    return -1;
  }
  std::cout << "Pass" << std::endl;
  return 0;
}
#endif

