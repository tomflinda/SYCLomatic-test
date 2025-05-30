// ===--- c2c_2d_outofplace.dp.cpp ----------------------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===---------------------------------------------------------------------===//

#define DPCT_USM_LEVEL_NONE
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <oneapi/mkl.hpp>
#include <dpct/fft_utils.hpp>
#include "common.h"
#include <cstring>
#include <iostream>

bool c2c_2d_outofplace() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::fft::fft_engine *plan_fwd;
  sycl::float2 forward_idata_h[2][5];
  set_value((float*)forward_idata_h, 20);

  sycl::float2 *forward_idata_d;
  sycl::float2 *forward_odata_d;
  sycl::float2 *backward_odata_d;
  forward_idata_d =
      (sycl::float2 *)dpct::dpct_malloc(sizeof(sycl::float2) * 10);
  forward_odata_d =
      (sycl::float2 *)dpct::dpct_malloc(sizeof(sycl::float2) * 10);
  backward_odata_d =
      (sycl::float2 *)dpct::dpct_malloc(sizeof(sycl::float2) * 10);
  dpct::dpct_memcpy(forward_idata_d, forward_idata_h, sizeof(sycl::float2) * 10,
                    dpct::host_to_device);

  plan_fwd = dpct::fft::fft_engine::create(
      &q_ct1, 2, 5, dpct::fft::fft_type::complex_float_to_complex_float);
  plan_fwd->compute<sycl::float2, sycl::float2>(
      forward_idata_d, forward_odata_d, dpct::fft::fft_direction::forward);
  dev_ct1.queues_wait_and_throw();
  sycl::float2 forward_odata_h[10];
  dpct::dpct_memcpy(forward_odata_h, forward_odata_d, sizeof(sycl::float2) * 10,
                    dpct::device_to_host);

  sycl::float2 forward_odata_ref[10];
  forward_odata_ref[0] = sycl::float2{90, 100};
  forward_odata_ref[1] = sycl::float2{-23.7638, 3.76382};
  forward_odata_ref[2] = sycl::float2{-13.2492, -6.7508};
  forward_odata_ref[3] = sycl::float2{-6.7508, -13.2492};
  forward_odata_ref[4] = sycl::float2{3.76382, -23.7638};
  forward_odata_ref[5] = sycl::float2{-50, -50};
  forward_odata_ref[6] = sycl::float2{0, 0};
  forward_odata_ref[7] = sycl::float2{0, 0};
  forward_odata_ref[8] = sycl::float2{0, 0};
  forward_odata_ref[9] = sycl::float2{0, 0};

  dpct::fft::fft_engine::destroy(plan_fwd);

  if (!compare(forward_odata_ref, forward_odata_h, 10)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, 10);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, 10);

    dpct::dpct_free(forward_idata_d);
    dpct::dpct_free(forward_odata_d);
    dpct::dpct_free(backward_odata_d);

    return false;
  }

  dpct::fft::fft_engine *plan_bwd;
  plan_bwd = dpct::fft::fft_engine::create(
      &q_ct1, 2, 5, dpct::fft::fft_type::complex_float_to_complex_float);
  plan_bwd->compute<sycl::float2, sycl::float2>(
      forward_odata_d, backward_odata_d, dpct::fft::fft_direction::backward);
  dev_ct1.queues_wait_and_throw();
  sycl::float2 backward_odata_h[10];
  dpct::dpct_memcpy(backward_odata_h, backward_odata_d,
                    sizeof(sycl::float2) * 10, dpct::device_to_host);

  sycl::float2 backward_odata_ref[10];
  backward_odata_ref[0] = sycl::float2{0, 10};
  backward_odata_ref[1] = sycl::float2{20, 30};
  backward_odata_ref[2] = sycl::float2{40, 50};
  backward_odata_ref[3] = sycl::float2{60, 70};
  backward_odata_ref[4] = sycl::float2{80, 90};
  backward_odata_ref[5] = sycl::float2{100, 110};
  backward_odata_ref[6] = sycl::float2{120, 130};
  backward_odata_ref[7] = sycl::float2{140, 150};
  backward_odata_ref[8] = sycl::float2{160, 170};
  backward_odata_ref[9] = sycl::float2{180, 190};

  dpct::dpct_free(forward_idata_d);
  dpct::dpct_free(forward_odata_d);
  dpct::dpct_free(backward_odata_d);

  dpct::fft::fft_engine::destroy(plan_bwd);

  if (!compare(backward_odata_ref, backward_odata_h, 10)) {
    std::cout << "backward_odata_h:" << std::endl;
    print_values(backward_odata_h, 10);
    std::cout << "backward_odata_ref:" << std::endl;
    print_values(backward_odata_ref, 10);
    return false;
  }
  return true;
}


#ifdef DEBUG_FFT
int main() {
#define FUNC c2c_2d_outofplace
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

