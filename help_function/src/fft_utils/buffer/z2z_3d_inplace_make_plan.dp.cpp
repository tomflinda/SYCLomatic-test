// ===--- z2z_3d_inplace_make_plan.dp.cpp ---------------------*- C++ -*---===//
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

bool z2z_3d_inplace_make_plan() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::fft::fft_engine *plan_fwd;
  plan_fwd = dpct::fft::fft_engine::create();
  sycl::double2 forward_idata_h[2][3][5];
  set_value((double*)forward_idata_h, 60);

  sycl::double2 *data_d;
  data_d = (sycl::double2 *)dpct::dpct_malloc(sizeof(sycl::double2) * 30);
  dpct::dpct_memcpy(data_d, forward_idata_h, sizeof(sycl::double2) * 30,
                    dpct::host_to_device);

  size_t workSize;
  plan_fwd->commit(&q_ct1, 2, 3, 5,
                   dpct::fft::fft_type::complex_double_to_complex_double,
                   nullptr);
  plan_fwd->compute<sycl::double2, sycl::double2>(
      data_d, data_d, dpct::fft::fft_direction::forward);
  dev_ct1.queues_wait_and_throw();
  sycl::double2 forward_odata_h[30];
  dpct::dpct_memcpy(forward_odata_h, data_d, sizeof(sycl::double2) * 30,
                    dpct::device_to_host);

  sycl::double2 forward_odata_ref[30];
  forward_odata_ref[0] = sycl::double2{870, 900};
  forward_odata_ref[1] = sycl::double2{-71.2915, 11.2914};
  forward_odata_ref[2] = sycl::double2{-39.7476, -20.2524};
  forward_odata_ref[3] = sycl::double2{-20.2524, -39.7476};
  forward_odata_ref[4] = sycl::double2{11.2915, -71.2915};
  forward_odata_ref[5] = sycl::double2{-236.603, -63.3975};
  forward_odata_ref[6] = sycl::double2{0, 0};
  forward_odata_ref[7] = sycl::double2{0, 0};
  forward_odata_ref[8] = sycl::double2{0, 0};
  forward_odata_ref[9] = sycl::double2{0, 0};
  forward_odata_ref[10] = sycl::double2{-63.3975, -236.603};
  forward_odata_ref[11] = sycl::double2{0, 0};
  forward_odata_ref[12] = sycl::double2{0, 0};
  forward_odata_ref[13] = sycl::double2{0, 0};
  forward_odata_ref[14] = sycl::double2{0, 0};
  forward_odata_ref[15] = sycl::double2{-450, -450};
  forward_odata_ref[16] = sycl::double2{0, 0};
  forward_odata_ref[17] = sycl::double2{0, 0};
  forward_odata_ref[18] = sycl::double2{0, 0};
  forward_odata_ref[19] = sycl::double2{0, 0};
  forward_odata_ref[20] = sycl::double2{0, 0};
  forward_odata_ref[21] = sycl::double2{0, 0};
  forward_odata_ref[22] = sycl::double2{0, 0};
  forward_odata_ref[23] = sycl::double2{0, 0};
  forward_odata_ref[24] = sycl::double2{0, 0};
  forward_odata_ref[25] = sycl::double2{0, 0};
  forward_odata_ref[26] = sycl::double2{0, 0};
  forward_odata_ref[27] = sycl::double2{0, 0};
  forward_odata_ref[28] = sycl::double2{0, 0};
  forward_odata_ref[29] = sycl::double2{0, 0};

  dpct::fft::fft_engine::destroy(plan_fwd);

  if (!compare(forward_odata_ref, forward_odata_h, 30)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, 30);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, 30);

    dpct::dpct_free(data_d);

    return false;
  }

  dpct::fft::fft_engine *plan_bwd;
  plan_bwd = dpct::fft::fft_engine::create();
  plan_bwd->commit(&q_ct1, 2, 3, 5,
                   dpct::fft::fft_type::complex_double_to_complex_double,
                   nullptr);
  plan_bwd->compute<sycl::double2, sycl::double2>(
      data_d, data_d, dpct::fft::fft_direction::backward);
  dev_ct1.queues_wait_and_throw();
  sycl::double2 backward_odata_h[30];
  dpct::dpct_memcpy(backward_odata_h, data_d, sizeof(sycl::double2) * 30,
                    dpct::device_to_host);

  sycl::double2 backward_odata_ref[30];
  backward_odata_ref[0] = sycl::double2{0, 30};
  backward_odata_ref[1] = sycl::double2{60, 90};
  backward_odata_ref[2] = sycl::double2{120, 150};
  backward_odata_ref[3] = sycl::double2{180, 210};
  backward_odata_ref[4] = sycl::double2{240, 270};
  backward_odata_ref[5] = sycl::double2{300, 330};
  backward_odata_ref[6] = sycl::double2{360, 390};
  backward_odata_ref[7] = sycl::double2{420, 450};
  backward_odata_ref[8] = sycl::double2{480, 510};
  backward_odata_ref[9] = sycl::double2{540, 570};
  backward_odata_ref[10] = sycl::double2{600, 630};
  backward_odata_ref[11] = sycl::double2{660, 690};
  backward_odata_ref[12] = sycl::double2{720, 750};
  backward_odata_ref[13] = sycl::double2{780, 810};
  backward_odata_ref[14] = sycl::double2{840, 870};
  backward_odata_ref[15] = sycl::double2{900, 930};
  backward_odata_ref[16] = sycl::double2{960, 990};
  backward_odata_ref[17] = sycl::double2{1020, 1050};
  backward_odata_ref[18] = sycl::double2{1080, 1110};
  backward_odata_ref[19] = sycl::double2{1140, 1170};
  backward_odata_ref[20] = sycl::double2{1200, 1230};
  backward_odata_ref[21] = sycl::double2{1260, 1290};
  backward_odata_ref[22] = sycl::double2{1320, 1350};
  backward_odata_ref[23] = sycl::double2{1380, 1410};
  backward_odata_ref[24] = sycl::double2{1440, 1470};
  backward_odata_ref[25] = sycl::double2{1500, 1530};
  backward_odata_ref[26] = sycl::double2{1560, 1590};
  backward_odata_ref[27] = sycl::double2{1620, 1650};
  backward_odata_ref[28] = sycl::double2{1680, 1710};
  backward_odata_ref[29] = sycl::double2{1740, 1770};

  dpct::dpct_free(data_d);
  dpct::fft::fft_engine::destroy(plan_bwd);

  if (!compare(backward_odata_ref, backward_odata_h, 30)) {
    std::cout << "backward_odata_h:" << std::endl;
    print_values(backward_odata_h, 30);
    std::cout << "backward_odata_ref:" << std::endl;
    print_values(backward_odata_ref, 30);
    return false;
  }
  return true;
}



#ifdef DEBUG_FFT
int main() {
#define FUNC z2z_3d_inplace_make_plan
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

