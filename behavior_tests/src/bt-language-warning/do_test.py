# ====------ do_test.py---------- *- Python -* ----===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
# ===----------------------------------------------------------------------===#
import subprocess
import platform
import os
import sys

from test_utils import *

def setup_test():
    change_dir(test_config.current_test)
    return True

def migrate_test():
    call_subprocess(test_config.CT_TOOL + " test.cpp --out-root=out --cuda-include-path=" + test_config.include_path)
    return is_sub_string("NOTE: test.cpp is treated as a CUDA file by default. Use the --extra-arg=-xc++ option to treat test.cpp as a C++ file if needed.", test_config.command_output)

def build_test():
    return True

def run_test():
    return True