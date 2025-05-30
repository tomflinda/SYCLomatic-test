// ===--- d2zz2d_many_2d_inplace_basic.dp.cpp -----------------*- C++ -*---===//
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

bool d2zz2d_many_2d_inplace_basic() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::fft::fft_engine *plan_fwd;
  plan_fwd = dpct::fft::fft_engine::create();
  double forward_idata_h[16];
  set_value(forward_idata_h, 2, 3, 4);
  set_value(forward_idata_h + 8, 2, 3, 4);

  double* data_d;
  data_d = (double *)dpct::dpct_malloc(sizeof(double) * 16);
  dpct::dpct_memcpy(data_d, forward_idata_h, sizeof(double) * 16,
                    dpct::host_to_device);

  int n[2] = {2, 3};
  size_t workSize;
  plan_fwd->commit(&q_ct1, 2, n, nullptr, 0, 0, nullptr, 0, 0,
                   dpct::fft::fft_type::real_double_to_complex_double, 2,
                   nullptr);
  plan_fwd->compute<double, sycl::double2>(data_d, (sycl::double2 *)data_d,
                                           dpct::fft::fft_direction::forward);
  dev_ct1.queues_wait_and_throw();
  sycl::double2 forward_odata_h[8];
  dpct::dpct_memcpy(forward_odata_h, data_d, sizeof(double) * 16,
                    dpct::device_to_host);

  sycl::double2 forward_odata_ref[8];
  forward_odata_ref[0] = sycl::double2{15, 0};
  forward_odata_ref[1] = sycl::double2{-3, 1.73205};
  forward_odata_ref[2] = sycl::double2{-9, 0};
  forward_odata_ref[3] = sycl::double2{0, 0};
  forward_odata_ref[4] = sycl::double2{15, 0};
  forward_odata_ref[5] = sycl::double2{-3, 1.73205};
  forward_odata_ref[6] = sycl::double2{-9, 0};
  forward_odata_ref[7] = sycl::double2{0, 0};

  dpct::fft::fft_engine::destroy(plan_fwd);

  if (!compare(forward_odata_ref, forward_odata_h, 8)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, 8);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, 8);

    dpct::dpct_free(data_d);

    return false;
  }

  dpct::fft::fft_engine *plan_bwd;
  plan_bwd = dpct::fft::fft_engine::create();
  plan_bwd->commit(&q_ct1, 2, n, nullptr, 0, 0, nullptr, 0, 0,
                   dpct::fft::fft_type::complex_double_to_real_double, 2,
                   nullptr);
  plan_bwd->compute<sycl::double2, double>((sycl::double2 *)data_d, data_d,
                                           dpct::fft::fft_direction::backward);
  dev_ct1.queues_wait_and_throw();
  double backward_odata_h[16];
  dpct::dpct_memcpy(backward_odata_h, data_d, sizeof(double) * 16,
                    dpct::device_to_host);

  double backward_odata_ref[16];
  backward_odata_ref[0] =  0;
  backward_odata_ref[1] =  6;
  backward_odata_ref[2] =  12;
  backward_odata_ref[3] =  1.73205;
  backward_odata_ref[4] =  18;
  backward_odata_ref[5] =  24;
  backward_odata_ref[6] =  30;
  backward_odata_ref[7] =  1.73205;
  backward_odata_ref[8] =  0;
  backward_odata_ref[9] =  6;
  backward_odata_ref[10] = 12;
  backward_odata_ref[11] = 1.73205;
  backward_odata_ref[12] = 18;
  backward_odata_ref[13] = 24;
  backward_odata_ref[14] = 30;
  backward_odata_ref[15] = 1.73205;

  dpct::dpct_free(data_d);
  dpct::fft::fft_engine::destroy(plan_bwd);

  std::vector<int> indices = {0, 1, 2,
                              4, 5, 6,
                              8, 9, 10,
                              12, 13, 14};
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
#define FUNC d2zz2d_many_2d_inplace_basic
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

