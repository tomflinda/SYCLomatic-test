// ===--- r2cc2r_many_3d_inplace_basic.dp.cpp -----------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===---------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <oneapi/mkl.hpp>
#include <dpct/fft_utils.hpp>
#include "common.h"
#include <cstring>
#include <iostream>

bool r2cc2r_many_3d_inplace_basic() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::fft::fft_engine *plan_fwd;
  plan_fwd = dpct::fft::fft_engine::create();
  float forward_idata_h[64];
  forward_idata_h[0]  = 0;
  forward_idata_h[1]  = 1;
  forward_idata_h[2]  = 2;
  forward_idata_h[4]  = 3;
  forward_idata_h[5]  = 4;
  forward_idata_h[6]  = 5;
  forward_idata_h[8]  = 6;
  forward_idata_h[9]  = 7;
  forward_idata_h[10] = 8;
  forward_idata_h[12] = 9;
  forward_idata_h[13] = 10;
  forward_idata_h[14] = 11;
  forward_idata_h[16] = 12;
  forward_idata_h[17] = 13;
  forward_idata_h[18] = 14;
  forward_idata_h[20] = 15;
  forward_idata_h[21] = 16;
  forward_idata_h[22] = 17;
  forward_idata_h[24] = 18;
  forward_idata_h[25] = 19;
  forward_idata_h[26] = 20;
  forward_idata_h[28] = 21;
  forward_idata_h[29] = 22;
  forward_idata_h[30] = 23;

  forward_idata_h[32]  = 0;
  forward_idata_h[33]  = 1;
  forward_idata_h[34]  = 2;
  forward_idata_h[36]  = 3;
  forward_idata_h[37]  = 4;
  forward_idata_h[38]  = 5;
  forward_idata_h[40]  = 6;
  forward_idata_h[41]  = 7;
  forward_idata_h[42] = 8;
  forward_idata_h[44] = 9;
  forward_idata_h[45] = 10;
  forward_idata_h[46] = 11;
  forward_idata_h[48] = 12;
  forward_idata_h[49] = 13;
  forward_idata_h[50] = 14;
  forward_idata_h[52] = 15;
  forward_idata_h[53] = 16;
  forward_idata_h[54] = 17;
  forward_idata_h[56] = 18;
  forward_idata_h[57] = 19;
  forward_idata_h[58] = 20;
  forward_idata_h[60] = 21;
  forward_idata_h[61] = 22;
  forward_idata_h[62] = 23;

  float* data_d;
  data_d = sycl::malloc_device<float>(64, q_ct1);
  q_ct1.memcpy(data_d, forward_idata_h, sizeof(float) * 64).wait();

  int n[3] = {4, 2, 3};
  size_t workSize;
  plan_fwd->commit(&q_ct1, 3, n, nullptr, 0, 0, nullptr, 0, 0,
                   dpct::fft::fft_type::real_float_to_complex_float, 2,
                   nullptr);
  plan_fwd->compute<float, sycl::float2>(data_d, (sycl::float2 *)data_d,
                                         dpct::fft::fft_direction::forward);
  dev_ct1.queues_wait_and_throw();
  sycl::float2 forward_odata_h[32];
  q_ct1.memcpy(forward_odata_h, data_d, sizeof(float) * 64).wait();

  sycl::float2 forward_odata_ref[32];
  forward_odata_ref[0] = sycl::float2{276, 0};
  forward_odata_ref[1] = sycl::float2{-12, 6.9282};
  forward_odata_ref[2] = sycl::float2{-36, 0};
  forward_odata_ref[3] = sycl::float2{0, 0};
  forward_odata_ref[4] = sycl::float2{-72, 72};
  forward_odata_ref[5] = sycl::float2{0, 0};
  forward_odata_ref[6] = sycl::float2{0, 0};
  forward_odata_ref[7] = sycl::float2{0, 0};
  forward_odata_ref[8] = sycl::float2{-72, 0};
  forward_odata_ref[9] = sycl::float2{0, 0};
  forward_odata_ref[10] = sycl::float2{0, 0};
  forward_odata_ref[11] = sycl::float2{0, 0};
  forward_odata_ref[12] = sycl::float2{-72, -72};
  forward_odata_ref[13] = sycl::float2{0, 0};
  forward_odata_ref[14] = sycl::float2{0, 0};
  forward_odata_ref[15] = sycl::float2{0, 0};
  forward_odata_ref[16] = sycl::float2{276, 0};
  forward_odata_ref[17] = sycl::float2{-12, 6.9282};
  forward_odata_ref[18] = sycl::float2{-36, 0};
  forward_odata_ref[19] = sycl::float2{0, 0};
  forward_odata_ref[20] = sycl::float2{-72, 72};
  forward_odata_ref[21] = sycl::float2{0, 0};
  forward_odata_ref[22] = sycl::float2{0, 0};
  forward_odata_ref[23] = sycl::float2{0, 0};
  forward_odata_ref[24] = sycl::float2{-72, 0};
  forward_odata_ref[25] = sycl::float2{0, 0};
  forward_odata_ref[26] = sycl::float2{0, 0};
  forward_odata_ref[27] = sycl::float2{0, 0};
  forward_odata_ref[28] = sycl::float2{-72, -72};
  forward_odata_ref[29] = sycl::float2{0, 0};
  forward_odata_ref[30] = sycl::float2{0, 0};
  forward_odata_ref[31] = sycl::float2{0, 0};

  dpct::fft::fft_engine::destroy(plan_fwd);

  if (!compare(forward_odata_ref, forward_odata_h, 32)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, 32);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, 32);

    sycl::free(data_d, q_ct1);

    return false;
  }

  dpct::fft::fft_engine *plan_bwd;
  plan_bwd = dpct::fft::fft_engine::create();
  plan_bwd->commit(&q_ct1, 3, n, nullptr, 0, 0, nullptr, 0, 0,
                   dpct::fft::fft_type::complex_float_to_real_float, 2,
                   nullptr);
  plan_bwd->compute<sycl::float2, float>((sycl::float2 *)data_d, data_d,
                                         dpct::fft::fft_direction::backward);
  dev_ct1.queues_wait_and_throw();
  float backward_odata_h[64];
  q_ct1.memcpy(backward_odata_h, data_d, sizeof(float) * 64).wait();

  float backward_odata_ref[64];
  backward_odata_ref[0]  = 0;
  backward_odata_ref[1]  = 24;
  backward_odata_ref[2]  = 48;
  backward_odata_ref[3]  = 6.9282;
  backward_odata_ref[4]  = 72;
  backward_odata_ref[5]  = 96;
  backward_odata_ref[6]  = 120;
  backward_odata_ref[7]  = 6.9282;
  backward_odata_ref[8]  = 144;
  backward_odata_ref[9]  = 168;
  backward_odata_ref[10] = 192;
  backward_odata_ref[11] = 6.9282;
  backward_odata_ref[12] = 216;
  backward_odata_ref[13] = 240;
  backward_odata_ref[14] = 264;
  backward_odata_ref[15] = 6.9282;
  backward_odata_ref[16] = 288;
  backward_odata_ref[17] = 312;
  backward_odata_ref[18] = 336;
  backward_odata_ref[19] = 6.9282;
  backward_odata_ref[20] = 360;
  backward_odata_ref[21] = 384;
  backward_odata_ref[22] = 408;
  backward_odata_ref[23] = 6.9282;
  backward_odata_ref[24] = 432;
  backward_odata_ref[25] = 456;
  backward_odata_ref[26] = 480;
  backward_odata_ref[27] = 6.9282;
  backward_odata_ref[28] = 504;
  backward_odata_ref[29] = 528;
  backward_odata_ref[30] = 552;
  backward_odata_ref[31] = 6.9282;
  backward_odata_ref[32] = 0;
  backward_odata_ref[33] = 24;
  backward_odata_ref[34] = 48;
  backward_odata_ref[35] = 6.9282;
  backward_odata_ref[36] = 72;
  backward_odata_ref[37] = 96;
  backward_odata_ref[38] = 120;
  backward_odata_ref[39] = 6.9282;
  backward_odata_ref[40] = 144;
  backward_odata_ref[41] = 168;
  backward_odata_ref[42] = 192;
  backward_odata_ref[43] = 6.9282;
  backward_odata_ref[44] = 216;
  backward_odata_ref[45] = 240;
  backward_odata_ref[46] = 264;
  backward_odata_ref[47] = 6.9282;
  backward_odata_ref[48] = 288;
  backward_odata_ref[49] = 312;
  backward_odata_ref[50] = 336;
  backward_odata_ref[51] = 6.9282;
  backward_odata_ref[52] = 360;
  backward_odata_ref[53] = 384;
  backward_odata_ref[54] = 408;
  backward_odata_ref[55] = 6.9282;
  backward_odata_ref[56] = 432;
  backward_odata_ref[57] = 456;
  backward_odata_ref[58] = 480;
  backward_odata_ref[59] = 6.9282;
  backward_odata_ref[60] = 504;
  backward_odata_ref[61] = 528;
  backward_odata_ref[62] = 552;
  backward_odata_ref[63] = 6.9282;

  sycl::free(data_d, q_ct1);
  dpct::fft::fft_engine::destroy(plan_bwd);

  std::vector<int> indices = {0, 1, 2,
                              4, 5, 6,
                              8, 9, 10,
                              12, 13, 14,
                              16 ,17, 18,
                              20, 21, 22,
                              24, 25, 26,
                              28, 29, 30,
                              32, 33, 34,
                              36, 37, 38,
                              40, 41, 42,
                              44, 45, 46,
                              48, 49, 50,
                              52, 53, 54,
                              56, 57, 58,
                              60, 61, 62};
  if (!compare(backward_odata_ref, backward_odata_h, indices)) {
    std::cout << "backward_odata_h:" << std::endl;
    print_values(backward_odata_h, indices);
    std::cout << "backward_odata_ref:" << std::endl;
    print_values(backward_odata_ref, indices);
    return false;
  }
  return true;
}


#ifdef DEBUG_FFT
int main() {
#define FUNC r2cc2r_many_3d_inplace_basic
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

