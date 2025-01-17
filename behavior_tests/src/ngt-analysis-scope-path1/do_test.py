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
import shutil

from test_utils import *


def setup_test():
    change_dir(test_config.current_test)
    return True


def migrate_test():
    # clean previous migration output
    if (os.path.exists("out")):
        shutil.rmtree("out")
    migrate_cmd = test_config.CT_TOOL + " --cuda-include-path=" + test_config.include_path + " " + os.path.join(
        "cuda",
        "call_device_func_outside.cu") + " --in-root=" + os.path.realpath(
            ".") + " --out-root=" + os.path.realpath("out")
    for analysis_scope in (os.path.join(".", "cuda"),
                           os.path.join("non_exist_dir", "abc")):
        call_subprocess(migrate_cmd + " --analysis-scope-path=" +
                        os.path.realpath(analysis_scope))
        if not is_sub_string(
                "Error: The path for --analysis-scope-path is not the same as or a parent directory of --in-root",
                test_config.command_output):
            return False
    return True


def build_test():
    return True


def run_test():
    return True