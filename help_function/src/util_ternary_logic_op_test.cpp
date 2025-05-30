// ====------ util_ternary_logic_op_test.cpp ------------ *- C/C++ -* ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===--------------------------------------------------------------------===//


#include <cmath>
#include <cstdint>
#include <cstdio>
#include <dpct/dpct.hpp>
#include <map>
#include <sycl/sycl.hpp>

// clang-format off
void reference_of_lop3(uint32_t &R, uint32_t A, uint32_t B, uint32_t C, uint32_t D) {
  switch (D) {
  case 0: R = 0; break;
  case 1: R = (~A & ~B & ~C); break;
  case 2: R = (~A & ~B & C); break;
  case 3: R = (~A & ~B & ~C) | (~A & ~B & C); break;
  case 4: R = (~A & B & ~C); break;
  case 5: R = (~A & ~B & ~C) | (~A & B & ~C); break;
  case 6: R = (~A & ~B & C) | (~A & B & ~C); break;
  case 7: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C); break;
  case 8: R = (~A & B & C); break;
  case 9: R = (~A & ~B & ~C) | (~A & B & C); break;
  case 10: R = (~A & ~B & C) | (~A & B & C); break;
  case 11: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C); break;
  case 12: R = (~A & B & ~C) | (~A & B & C); break;
  case 13: R = (~A & ~B & ~C) | (~A & B & ~C) | (~A & B & C); break;
  case 14: R = (~A & ~B & C) | (~A & B & ~C) | (~A & B & C); break;
  case 15: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (~A & B & C); break;
  case 16: R = (A & ~B & ~C); break;
  case 17: R = (~A & ~B & ~C) | (A & ~B & ~C); break;
  case 18: R = (~A & ~B & C) | (A & ~B & ~C); break;
  case 19: R = (~A & ~B & ~C) | (~A & ~B & C) | (A & ~B & ~C); break;
  case 20: R = (~A & B & ~C) | (A & ~B & ~C); break;
  case 21: R = (~A & ~B & ~C) | (~A & B & ~C) | (A & ~B & ~C); break;
  case 22: R = (~A & ~B & C) | (~A & B & ~C) | (A & ~B & ~C); break;
  case 23: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (A & ~B & ~C); break;
  case 24: R = (~A & B & C) | (A & ~B & ~C); break;
  case 25: R = (~A & ~B & ~C) | (~A & B & C) | (A & ~B & ~C); break;
  case 26: R = (A & B | C) ^ A; break;
  case 27: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C) | (A & ~B & ~C); break;
  case 28: R = (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C); break;
  case 29: R = (~A & ~B & ~C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C); break;
  case 30: R = A ^ (B | C); break;
  case 31: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C); break;
  case 32: R = (A & ~B & C); break;
  case 33: R = (~A & ~B & ~C) | (A & ~B & C); break;
  case 34: R = (~A & ~B & C) | (A & ~B & C); break;
  case 35: R = (~A & ~B & ~C) | (~A & ~B & C) | (A & ~B & C); break;
  case 36: R = (~A & B & ~C) | (A & ~B & C); break;
  case 37: R = (~A & ~B & ~C) | (~A & B & ~C) | (A & ~B & C); break;
  case 38: R = (~A & ~B & C) | (~A & B & ~C) | (A & ~B & C); break;
  case 39: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (A & ~B & C); break;
  case 40: R = (~A & B & C) | (A & ~B & C); break;
  case 41: R = (~A & ~B & ~C) | (~A & B & C) | (A & ~B & C); break;
  case 42: R = (~A & ~B & C) | (~A & B & C) | (A & ~B & C); break;
  case 43: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C) | (A & ~B & C); break;
  case 44: R = (~A & B & ~C) | (~A & B & C) | (A & ~B & C); break;
  case 45: R = ~A ^ (~B & C); break;
  case 46: R = (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & C); break;
  case 47: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & C); break;
  case 48: R = (A & ~B & ~C) | (A & ~B & C); break;
  case 49: R = (~A & ~B & ~C) | (A & ~B & ~C) | (A & ~B & C); break;
  case 50: R = (~A & ~B & C) | (A & ~B & ~C) | (A & ~B & C); break;
  case 51: R = (~A & ~B & ~C) | (~A & ~B & C) | (A & ~B & ~C) | (A & ~B & C); break;
  case 52: R = (~A & B & ~C) | (A & ~B & ~C) | (A & ~B & C); break;
  case 53: R = (~A & ~B & ~C) | (~A & B & ~C) | (A & ~B & ~C) | (A & ~B & C); break;
  case 54: R = (~A & ~B & C) | (~A & B & ~C) | (A & ~B & ~C) | (A & ~B & C); break;
  case 55: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (A & ~B & ~C) | (A & ~B & C); break;
  case 56: R = (~A & B & C) | (A & ~B & ~C) | (A & ~B & C); break;
  case 57: R = (~A & ~B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C); break;
  case 58: R = (~A & ~B & C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C); break;
  case 59: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C); break;
  case 60: R = (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C); break;
  case 61: R = (~A & ~B & ~C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C); break;
  case 62: R = (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C); break;
  case 63: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C); break;
  case 64: R = A & B & ~C; break;
  case 65: R = (~A & ~B & ~C) | (A & B & ~C); break;
  case 66: R = (~A & ~B & C) | (A & B & ~C); break;
  case 67: R = (~A & ~B & ~C) | (~A & ~B & C) | (A & B & ~C); break;
  case 68: R = (~A & B & ~C) | (A & B & ~C); break;
  case 69: R = (~A & ~B & ~C) | (~A & B & ~C) | (A & B & ~C); break;
  case 70: R = (~A & ~B & C) | (~A & B & ~C) | (A & B & ~C); break;
  case 71: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (A & B & ~C); break;
  case 72: R = (~A & B & C) | (A & B & ~C); break;
  case 73: R = (~A & ~B & ~C) | (~A & B & C) | (A & B & ~C); break;
  case 74: R = (~A & ~B & C) | (~A & B & C) | (A & B & ~C); break;
  case 75: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C) | (A & B & ~C); break;
  case 76: R = (~A & B & ~C) | (~A & B & C) | (A & B & ~C); break;
  case 77: R = (~A & ~B & ~C) | (~A & B & ~C) | (~A & B & C) | (A & B & ~C); break;
  case 78: R = (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & B & ~C); break;
  case 79: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & B & ~C); break;
  case 80: R = (A & ~B & ~C) | (A & B & ~C); break;
  case 81: R = (~A & ~B & ~C) | (A & ~B & ~C) | (A & B & ~C); break;
  case 82: R = (~A & ~B & C) | (A & ~B & ~C) | (A & B & ~C); break;
  case 83: R = (~A & ~B & ~C) | (~A & ~B & C) | (A & ~B & ~C) | (A & B & ~C); break;
  case 84: R = (~A & B & ~C) | (A & ~B & ~C) | (A & B & ~C); break;
  case 85: R = (~A & ~B & ~C) | (~A & B & ~C) | (A & ~B & ~C) | (A & B & ~C); break;
  case 86: R = (~A & ~B & C) | (~A & B & ~C) | (A & ~B & ~C) | (A & B & ~C); break;
  case 87: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (A & ~B & ~C) | (A & B & ~C); break;
  case 88: R = (~A & B & C) | (A & ~B & ~C) | (A & B & ~C); break;
  case 89: R = (~A & ~B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & B & ~C); break;
  case 90: R = (~A & ~B & C) | (~A & B & C) | (A & ~B & ~C) | (A & B & ~C); break;
  case 91: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C) | (A & ~B & ~C) | (A & B & ~C); break;
  case 92: R = (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & B & ~C); break;
  case 93: R = (~A & ~B & ~C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & B & ~C); break;
  case 94: R = (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & B & ~C); break;
  case 95: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & B & ~C); break;
  case 96: R = (A & ~B & C) | (A & B & ~C); break;
  case 97: R = (~A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 98: R = (~A & ~B & C) | (A & ~B & C) | (A & B & ~C); break;
  case 99: R = (~A & ~B & ~C) | (~A & ~B & C) | (A & ~B & C) | (A & B & ~C); break;
  case 100: R = (~A & B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 101: R = (~A & ~B & ~C) | (~A & B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 102: R = (~A & ~B & C) | (~A & B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 103: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 104: R = (~A & B & C) | (A & ~B & C) | (A & B & ~C); break;
  case 105: R = (~A & ~B & ~C) | (~A & B & C) | (A & ~B & C) | (A & B & ~C); break;
  case 106: R = (~A & ~B & C) | (~A & B & C) | (A & ~B & C) | (A & B & ~C); break;
  case 107: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C) | (A & ~B & C) | (A & B & ~C); break;
  case 108: R = (~A & B & ~C) | (~A & B & C) | (A & ~B & C) | (A & B & ~C); break;
  case 109: R = (~A & ~B & ~C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & C) | (A & B & ~C); break;
  case 110: R = (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & C) | (A & B & ~C); break;
  case 111: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & C) | (A & B & ~C); break;
  case 112: R = (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 113: R = (~A & ~B & ~C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 114: R = (~A & ~B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 115: R = (~A & ~B & ~C) | (~A & ~B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 116: R = (~A & B & ~C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 117: R = (~A & ~B & ~C) | (~A & B & ~C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 118: R = (~A & ~B & C) | (~A & B & ~C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 119: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 120: R = A ^ (B & C); break;
  case 121: R = (~A & ~B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 122: R = (~A & ~B & C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 123: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 124: R = (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 125: R = (~A & ~B & ~C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 126: R = (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 127: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C); break;
  case 128: R = A & B & C; break;
  case 129: R = (~A & ~B & ~C) | (A & B & C); break;
  case 130: R = (~A & ~B & C) | (A & B & C); break;
  case 131: R = (~A & ~B & ~C) | (~A & ~B & C) | (A & B & C); break;
  case 132: R = (~A & B & ~C) | (A & B & C); break;
  case 133: R = (~A & ~B & ~C) | (~A & B & ~C) | (A & B & C); break;
  case 134: R = (~A & ~B & C) | (~A & B & ~C) | (A & B & C); break;
  case 135: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (A & B & C); break;
  case 136: R = (~A & B & C) | (A & B & C); break;
  case 137: R = (~A & ~B & ~C) | (~A & B & C) | (A & B & C); break;
  case 138: R = (~A & ~B & C) | (~A & B & C) | (A & B & C); break;
  case 139: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C) | (A & B & C); break;
  case 140: R = (~A & B & ~C) | (~A & B & C) | (A & B & C); break;
  case 141: R = (~A & ~B & ~C) | (~A & B & ~C) | (~A & B & C) | (A & B & C); break;
  case 142: R = (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & B & C); break;
  case 143: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & B & C); break;
  case 144: R = (A & ~B & ~C) | (A & B & C); break;
  case 145: R = (~A & ~B & ~C) | (A & ~B & ~C) | (A & B & C); break;
  case 146: R = (~A & ~B & C) | (A & ~B & ~C) | (A & B & C); break;
  case 147: R = (~A & ~B & ~C) | (~A & ~B & C) | (A & ~B & ~C) | (A & B & C); break;
  case 148: R = (~A & B & ~C) | (A & ~B & ~C) | (A & B & C); break;
  case 149: R = (~A & ~B & ~C) | (~A & B & ~C) | (A & ~B & ~C) | (A & B & C); break;
  case 150: R = A ^ B ^ C; break;
  case 151: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (A & ~B & ~C) | (A & B & C); break;
  case 152: R = (~A & B & C) | (A & ~B & ~C) | (A & B & C); break;
  case 153: R = (~A & ~B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & B & C); break;
  case 154: R = (~A & ~B & C) | (~A & B & C) | (A & ~B & ~C) | (A & B & C); break;
  case 155: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C) | (A & ~B & ~C) | (A & B & C); break;
  case 156: R = (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & B & C); break;
  case 157: R = (~A & ~B & ~C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & B & C); break;
  case 158: R = (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & B & C); break;
  case 159: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & B & C); break;
  case 160: R = (A & ~B & C) | (A & B & C); break;
  case 161: R = (~A & ~B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 162: R = (~A & ~B & C) | (A & ~B & C) | (A & B & C); break;
  case 163: R = (~A & ~B & ~C) | (~A & ~B & C) | (A & ~B & C) | (A & B & C); break;
  case 164: R = (~A & B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 165: R = (~A & ~B & ~C) | (~A & B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 166: R = (~A & ~B & C) | (~A & B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 167: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 168: R = (~A & B & C) | (A & ~B & C) | (A & B & C); break;
  case 169: R = (~A & ~B & ~C) | (~A & B & C) | (A & ~B & C) | (A & B & C); break;
  case 170: R = (~A & ~B & C) | (~A & B & C) | (A & ~B & C) | (A & B & C); break;
  case 171: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C) | (A & ~B & C) | (A & B & C); break;
  case 172: R = (~A & B & ~C) | (~A & B & C) | (A & ~B & C) | (A & B & C); break;
  case 173: R = (~A & ~B & ~C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & C) | (A & B & C); break;
  case 174: R = (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & C) | (A & B & C); break;
  case 175: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & C) | (A & B & C); break;
  case 176: R = (A & ~B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 177: R = (~A & ~B & ~C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 178: R = (~A & ~B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 179: R = (~A & ~B & ~C) | (~A & ~B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 180: R = A ^ (B & ~C); break;
  case 181: R = (~A & ~B & ~C) | (~A & B & ~C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 182: R = (~A & ~B & C) | (~A & B & ~C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 183: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 184: R = (A ^ (B & (C ^ A))); break;
  case 185: R = (~A & ~B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 186: R = (~A & ~B & C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 187: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 188: R = (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 189: R = (~A & ~B & ~C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 190: R = (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 191: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & C); break;
  case 192: R = (A & B & ~C) | (A & B & C); break;
  case 193: R = (~A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 194: R = (~A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 195: R = (~A & ~B & ~C) | (~A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 196: R = (~A & B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 197: R = (~A & ~B & ~C) | (~A & B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 198: R = (~A & ~B & C) | (~A & B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 199: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 200: R = (~A & B & C) | (A & B & ~C) | (A & B & C); break;
  case 201: R = (~A & ~B & ~C) | (~A & B & C) | (A & B & ~C) | (A & B & C); break;
  case 202: R = (~A & ~B & C) | (~A & B & C) | (A & B & ~C) | (A & B & C); break;
  case 203: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C) | (A & B & ~C) | (A & B & C); break;
  case 204: R = (~A & B & ~C) | (~A & B & C) | (A & B & ~C) | (A & B & C); break;
  case 205: R = (~A & ~B & ~C) | (~A & B & ~C) | (~A & B & C) | (A & B & ~C) | (A & B & C); break;
  case 206: R = (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & B & ~C) | (A & B & C); break;
  case 207: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & B & ~C) | (A & B & C); break;
  case 208: R = (A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 209: R = (~A & ~B & ~C) | (A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 210: R = A ^ (~B & C); break;
  case 211: R = (~A & ~B & ~C) | (~A & ~B & C) | (A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 212: R = (~A & B & ~C) | (A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 213: R = (~A & ~B & ~C) | (~A & B & ~C) | (A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 214: R = (~A & ~B & C) | (~A & B & ~C) | (A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 215: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 216: R = (~A & B & C) | (A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 217: R = (~A & ~B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 218: R = (~A & ~B & C) | (~A & B & C) | (A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 219: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C) | (A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 220: R = (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 221: R = (~A & ~B & ~C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 222: R = (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 223: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & B & ~C) | (A & B & C); break;
  case 224: R = (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 225: R = (~A & ~B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 226: R = (~A & ~B & C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 227: R = (~A & ~B & ~C) | (~A & ~B & C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 228: R = (~A & B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 229: R = (~A & ~B & ~C) | (~A & B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 230: R = (~A & ~B & C) | (~A & B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 231: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 232: R = ((A & (B | C)) | (B & C)); break;
  case 233: R = (~A & ~B & ~C) | (~A & B & C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 234: R = (A & B) | C; break;
  case 235: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 236: R = (~A & B & ~C) | (~A & B & C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 237: R = (~A & ~B & ~C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 238: R = (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 239: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 240: R = (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 241: R = (~A & ~B & ~C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 242: R = (~A & ~B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 243: R = (~A & ~B & ~C) | (~A & ~B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 244: R = (~A & B & ~C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 245: R = (~A & ~B & ~C) | (~A & B & ~C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 246: R = (~A & ~B & C) | (~A & B & ~C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 247: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & ~C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 248: R = (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 249: R = (~A & ~B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 250: R = (~A & ~B & C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 251: R = (~A & ~B & ~C) | (~A & ~B & C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 252: R = (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 253: R = (~A & ~B & ~C) | (~A & B & ~C) | (~A & B & C) | (A & ~B & ~C) | (A & ~B & C) | (A & B & ~C) | (A & B & C); break;
  case 254: R = A | B | C; break;
  case 255: R = uint32_t(-1); break;
  default: break;
  }
}

void asm_lop3(uint32_t &R, uint32_t A, uint32_t B, uint32_t C, uint32_t D) {
  switch (D) {
  case 0: R = dpct::ternary_logic_op(A, B, C, 0x0); break;
  case 1: R = dpct::ternary_logic_op(A, B, C, 0x1); break;
  case 2: R = dpct::ternary_logic_op(A, B, C, 0x2); break;
  case 3: R = dpct::ternary_logic_op(A, B, C, 0x3); break;
  case 4: R = dpct::ternary_logic_op(A, B, C, 0x4); break;
  case 5: R = dpct::ternary_logic_op(A, B, C, 0x5); break;
  case 6: R = dpct::ternary_logic_op(A, B, C, 0x6); break;
  case 7: R = dpct::ternary_logic_op(A, B, C, 0x7); break;
  case 8: R = dpct::ternary_logic_op(A, B, C, 0x8); break;
  case 9: R = dpct::ternary_logic_op(A, B, C, 0x9); break;
  case 10: R = dpct::ternary_logic_op(A, B, C, 0xA); break;
  case 11: R = dpct::ternary_logic_op(A, B, C, 0xB); break;
  case 12: R = dpct::ternary_logic_op(A, B, C, 0xC); break;
  case 13: R = dpct::ternary_logic_op(A, B, C, 0xD); break;
  case 14: R = dpct::ternary_logic_op(A, B, C, 0xE); break;
  case 15: R = dpct::ternary_logic_op(A, B, C, 0xF); break;
  case 16: R = dpct::ternary_logic_op(A, B, C, 0x10); break;
  case 17: R = dpct::ternary_logic_op(A, B, C, 0x11); break;
  case 18: R = dpct::ternary_logic_op(A, B, C, 0x12); break;
  case 19: R = dpct::ternary_logic_op(A, B, C, 0x13); break;
  case 20: R = dpct::ternary_logic_op(A, B, C, 0x14); break;
  case 21: R = dpct::ternary_logic_op(A, B, C, 0x15); break;
  case 22: R = dpct::ternary_logic_op(A, B, C, 0x16); break;
  case 23: R = dpct::ternary_logic_op(A, B, C, 0x17); break;
  case 24: R = dpct::ternary_logic_op(A, B, C, 0x18); break;
  case 25: R = dpct::ternary_logic_op(A, B, C, 0x19); break;
  case 26: R = dpct::ternary_logic_op(A, B, C, 0x1A); break;
  case 27: R = dpct::ternary_logic_op(A, B, C, 0x1B); break;
  case 28: R = dpct::ternary_logic_op(A, B, C, 0x1C); break;
  case 29: R = dpct::ternary_logic_op(A, B, C, 0x1D); break;
  case 30: R = dpct::ternary_logic_op(A, B, C, 0x1E); break;
  case 31: R = dpct::ternary_logic_op(A, B, C, 0x1F); break;
  case 32: R = dpct::ternary_logic_op(A, B, C, 0x20); break;
  case 33: R = dpct::ternary_logic_op(A, B, C, 0x21); break;
  case 34: R = dpct::ternary_logic_op(A, B, C, 0x22); break;
  case 35: R = dpct::ternary_logic_op(A, B, C, 0x23); break;
  case 36: R = dpct::ternary_logic_op(A, B, C, 0x24); break;
  case 37: R = dpct::ternary_logic_op(A, B, C, 0x25); break;
  case 38: R = dpct::ternary_logic_op(A, B, C, 0x26); break;
  case 39: R = dpct::ternary_logic_op(A, B, C, 0x27); break;
  case 40: R = dpct::ternary_logic_op(A, B, C, 0x28); break;
  case 41: R = dpct::ternary_logic_op(A, B, C, 0x29); break;
  case 42: R = dpct::ternary_logic_op(A, B, C, 0x2A); break;
  case 43: R = dpct::ternary_logic_op(A, B, C, 0x2B); break;
  case 44: R = dpct::ternary_logic_op(A, B, C, 0x2C); break;
  case 45: R = dpct::ternary_logic_op(A, B, C, 0x2D); break;
  case 46: R = dpct::ternary_logic_op(A, B, C, 0x2E); break;
  case 47: R = dpct::ternary_logic_op(A, B, C, 0x2F); break;
  case 48: R = dpct::ternary_logic_op(A, B, C, 0x30); break;
  case 49: R = dpct::ternary_logic_op(A, B, C, 0x31); break;
  case 50: R = dpct::ternary_logic_op(A, B, C, 0x32); break;
  case 51: R = dpct::ternary_logic_op(A, B, C, 0x33); break;
  case 52: R = dpct::ternary_logic_op(A, B, C, 0x34); break;
  case 53: R = dpct::ternary_logic_op(A, B, C, 0x35); break;
  case 54: R = dpct::ternary_logic_op(A, B, C, 0x36); break;
  case 55: R = dpct::ternary_logic_op(A, B, C, 0x37); break;
  case 56: R = dpct::ternary_logic_op(A, B, C, 0x38); break;
  case 57: R = dpct::ternary_logic_op(A, B, C, 0x39); break;
  case 58: R = dpct::ternary_logic_op(A, B, C, 0x3A); break;
  case 59: R = dpct::ternary_logic_op(A, B, C, 0x3B); break;
  case 60: R = dpct::ternary_logic_op(A, B, C, 0x3C); break;
  case 61: R = dpct::ternary_logic_op(A, B, C, 0x3D); break;
  case 62: R = dpct::ternary_logic_op(A, B, C, 0x3E); break;
  case 63: R = dpct::ternary_logic_op(A, B, C, 0x3F); break;
  case 64: R = dpct::ternary_logic_op(A, B, C, 0x40); break;
  case 65: R = dpct::ternary_logic_op(A, B, C, 0x41); break;
  case 66: R = dpct::ternary_logic_op(A, B, C, 0x42); break;
  case 67: R = dpct::ternary_logic_op(A, B, C, 0x43); break;
  case 68: R = dpct::ternary_logic_op(A, B, C, 0x44); break;
  case 69: R = dpct::ternary_logic_op(A, B, C, 0x45); break;
  case 70: R = dpct::ternary_logic_op(A, B, C, 0x46); break;
  case 71: R = dpct::ternary_logic_op(A, B, C, 0x47); break;
  case 72: R = dpct::ternary_logic_op(A, B, C, 0x48); break;
  case 73: R = dpct::ternary_logic_op(A, B, C, 0x49); break;
  case 74: R = dpct::ternary_logic_op(A, B, C, 0x4A); break;
  case 75: R = dpct::ternary_logic_op(A, B, C, 0x4B); break;
  case 76: R = dpct::ternary_logic_op(A, B, C, 0x4C); break;
  case 77: R = dpct::ternary_logic_op(A, B, C, 0x4D); break;
  case 78: R = dpct::ternary_logic_op(A, B, C, 0x4E); break;
  case 79: R = dpct::ternary_logic_op(A, B, C, 0x4F); break;
  case 80: R = dpct::ternary_logic_op(A, B, C, 0x50); break;
  case 81: R = dpct::ternary_logic_op(A, B, C, 0x51); break;
  case 82: R = dpct::ternary_logic_op(A, B, C, 0x52); break;
  case 83: R = dpct::ternary_logic_op(A, B, C, 0x53); break;
  case 84: R = dpct::ternary_logic_op(A, B, C, 0x54); break;
  case 85: R = dpct::ternary_logic_op(A, B, C, 0x55); break;
  case 86: R = dpct::ternary_logic_op(A, B, C, 0x56); break;
  case 87: R = dpct::ternary_logic_op(A, B, C, 0x57); break;
  case 88: R = dpct::ternary_logic_op(A, B, C, 0x58); break;
  case 89: R = dpct::ternary_logic_op(A, B, C, 0x59); break;
  case 90: R = dpct::ternary_logic_op(A, B, C, 0x5A); break;
  case 91: R = dpct::ternary_logic_op(A, B, C, 0x5B); break;
  case 92: R = dpct::ternary_logic_op(A, B, C, 0x5C); break;
  case 93: R = dpct::ternary_logic_op(A, B, C, 0x5D); break;
  case 94: R = dpct::ternary_logic_op(A, B, C, 0x5E); break;
  case 95: R = dpct::ternary_logic_op(A, B, C, 0x5F); break;
  case 96: R = dpct::ternary_logic_op(A, B, C, 0x60); break;
  case 97: R = dpct::ternary_logic_op(A, B, C, 0x61); break;
  case 98: R = dpct::ternary_logic_op(A, B, C, 0x62); break;
  case 99: R = dpct::ternary_logic_op(A, B, C, 0x63); break;
  case 100: R = dpct::ternary_logic_op(A, B, C, 0x64); break;
  case 101: R = dpct::ternary_logic_op(A, B, C, 0x65); break;
  case 102: R = dpct::ternary_logic_op(A, B, C, 0x66); break;
  case 103: R = dpct::ternary_logic_op(A, B, C, 0x67); break;
  case 104: R = dpct::ternary_logic_op(A, B, C, 0x68); break;
  case 105: R = dpct::ternary_logic_op(A, B, C, 0x69); break;
  case 106: R = dpct::ternary_logic_op(A, B, C, 0x6A); break;
  case 107: R = dpct::ternary_logic_op(A, B, C, 0x6B); break;
  case 108: R = dpct::ternary_logic_op(A, B, C, 0x6C); break;
  case 109: R = dpct::ternary_logic_op(A, B, C, 0x6D); break;
  case 110: R = dpct::ternary_logic_op(A, B, C, 0x6E); break;
  case 111: R = dpct::ternary_logic_op(A, B, C, 0x6F); break;
  case 112: R = dpct::ternary_logic_op(A, B, C, 0x70); break;
  case 113: R = dpct::ternary_logic_op(A, B, C, 0x71); break;
  case 114: R = dpct::ternary_logic_op(A, B, C, 0x72); break;
  case 115: R = dpct::ternary_logic_op(A, B, C, 0x73); break;
  case 116: R = dpct::ternary_logic_op(A, B, C, 0x74); break;
  case 117: R = dpct::ternary_logic_op(A, B, C, 0x75); break;
  case 118: R = dpct::ternary_logic_op(A, B, C, 0x76); break;
  case 119: R = dpct::ternary_logic_op(A, B, C, 0x77); break;
  case 120: R = dpct::ternary_logic_op(A, B, C, 0x78); break;
  case 121: R = dpct::ternary_logic_op(A, B, C, 0x79); break;
  case 122: R = dpct::ternary_logic_op(A, B, C, 0x7A); break;
  case 123: R = dpct::ternary_logic_op(A, B, C, 0x7B); break;
  case 124: R = dpct::ternary_logic_op(A, B, C, 0x7C); break;
  case 125: R = dpct::ternary_logic_op(A, B, C, 0x7D); break;
  case 126: R = dpct::ternary_logic_op(A, B, C, 0x7E); break;
  case 127: R = dpct::ternary_logic_op(A, B, C, 0x7F); break;
  case 128: R = dpct::ternary_logic_op(A, B, C, 0x80); break;
  case 129: R = dpct::ternary_logic_op(A, B, C, 0x81); break;
  case 130: R = dpct::ternary_logic_op(A, B, C, 0x82); break;
  case 131: R = dpct::ternary_logic_op(A, B, C, 0x83); break;
  case 132: R = dpct::ternary_logic_op(A, B, C, 0x84); break;
  case 133: R = dpct::ternary_logic_op(A, B, C, 0x85); break;
  case 134: R = dpct::ternary_logic_op(A, B, C, 0x86); break;
  case 135: R = dpct::ternary_logic_op(A, B, C, 0x87); break;
  case 136: R = dpct::ternary_logic_op(A, B, C, 0x88); break;
  case 137: R = dpct::ternary_logic_op(A, B, C, 0x89); break;
  case 138: R = dpct::ternary_logic_op(A, B, C, 0x8A); break;
  case 139: R = dpct::ternary_logic_op(A, B, C, 0x8B); break;
  case 140: R = dpct::ternary_logic_op(A, B, C, 0x8C); break;
  case 141: R = dpct::ternary_logic_op(A, B, C, 0x8D); break;
  case 142: R = dpct::ternary_logic_op(A, B, C, 0x8E); break;
  case 143: R = dpct::ternary_logic_op(A, B, C, 0x8F); break;
  case 144: R = dpct::ternary_logic_op(A, B, C, 0x90); break;
  case 145: R = dpct::ternary_logic_op(A, B, C, 0x91); break;
  case 146: R = dpct::ternary_logic_op(A, B, C, 0x92); break;
  case 147: R = dpct::ternary_logic_op(A, B, C, 0x93); break;
  case 148: R = dpct::ternary_logic_op(A, B, C, 0x94); break;
  case 149: R = dpct::ternary_logic_op(A, B, C, 0x95); break;
  case 150: R = dpct::ternary_logic_op(A, B, C, 0x96); break;
  case 151: R = dpct::ternary_logic_op(A, B, C, 0x97); break;
  case 152: R = dpct::ternary_logic_op(A, B, C, 0x98); break;
  case 153: R = dpct::ternary_logic_op(A, B, C, 0x99); break;
  case 154: R = dpct::ternary_logic_op(A, B, C, 0x9A); break;
  case 155: R = dpct::ternary_logic_op(A, B, C, 0x9B); break;
  case 156: R = dpct::ternary_logic_op(A, B, C, 0x9C); break;
  case 157: R = dpct::ternary_logic_op(A, B, C, 0x9D); break;
  case 158: R = dpct::ternary_logic_op(A, B, C, 0x9E); break;
  case 159: R = dpct::ternary_logic_op(A, B, C, 0x9F); break;
  case 160: R = dpct::ternary_logic_op(A, B, C, 0xA0); break;
  case 161: R = dpct::ternary_logic_op(A, B, C, 0xA1); break;
  case 162: R = dpct::ternary_logic_op(A, B, C, 0xA2); break;
  case 163: R = dpct::ternary_logic_op(A, B, C, 0xA3); break;
  case 164: R = dpct::ternary_logic_op(A, B, C, 0xA4); break;
  case 165: R = dpct::ternary_logic_op(A, B, C, 0xA5); break;
  case 166: R = dpct::ternary_logic_op(A, B, C, 0xA6); break;
  case 167: R = dpct::ternary_logic_op(A, B, C, 0xA7); break;
  case 168: R = dpct::ternary_logic_op(A, B, C, 0xA8); break;
  case 169: R = dpct::ternary_logic_op(A, B, C, 0xA9); break;
  case 170: R = dpct::ternary_logic_op(A, B, C, 0xAA); break;
  case 171: R = dpct::ternary_logic_op(A, B, C, 0xAB); break;
  case 172: R = dpct::ternary_logic_op(A, B, C, 0xAC); break;
  case 173: R = dpct::ternary_logic_op(A, B, C, 0xAD); break;
  case 174: R = dpct::ternary_logic_op(A, B, C, 0xAE); break;
  case 175: R = dpct::ternary_logic_op(A, B, C, 0xAF); break;
  case 176: R = dpct::ternary_logic_op(A, B, C, 0xB0); break;
  case 177: R = dpct::ternary_logic_op(A, B, C, 0xB1); break;
  case 178: R = dpct::ternary_logic_op(A, B, C, 0xB2); break;
  case 179: R = dpct::ternary_logic_op(A, B, C, 0xB3); break;
  case 180: R = dpct::ternary_logic_op(A, B, C, 0xB4); break;
  case 181: R = dpct::ternary_logic_op(A, B, C, 0xB5); break;
  case 182: R = dpct::ternary_logic_op(A, B, C, 0xB6); break;
  case 183: R = dpct::ternary_logic_op(A, B, C, 0xB7); break;
  case 184: R = dpct::ternary_logic_op(A, B, C, 0xB8); break;
  case 185: R = dpct::ternary_logic_op(A, B, C, 0xB9); break;
  case 186: R = dpct::ternary_logic_op(A, B, C, 0xBA); break;
  case 187: R = dpct::ternary_logic_op(A, B, C, 0xBB); break;
  case 188: R = dpct::ternary_logic_op(A, B, C, 0xBC); break;
  case 189: R = dpct::ternary_logic_op(A, B, C, 0xBD); break;
  case 190: R = dpct::ternary_logic_op(A, B, C, 0xBE); break;
  case 191: R = dpct::ternary_logic_op(A, B, C, 0xBF); break;
  case 192: R = dpct::ternary_logic_op(A, B, C, 0xC0); break;
  case 193: R = dpct::ternary_logic_op(A, B, C, 0xC1); break;
  case 194: R = dpct::ternary_logic_op(A, B, C, 0xC2); break;
  case 195: R = dpct::ternary_logic_op(A, B, C, 0xC3); break;
  case 196: R = dpct::ternary_logic_op(A, B, C, 0xC4); break;
  case 197: R = dpct::ternary_logic_op(A, B, C, 0xC5); break;
  case 198: R = dpct::ternary_logic_op(A, B, C, 0xC6); break;
  case 199: R = dpct::ternary_logic_op(A, B, C, 0xC7); break;
  case 200: R = dpct::ternary_logic_op(A, B, C, 0xC8); break;
  case 201: R = dpct::ternary_logic_op(A, B, C, 0xC9); break;
  case 202: R = dpct::ternary_logic_op(A, B, C, 0xCA); break;
  case 203: R = dpct::ternary_logic_op(A, B, C, 0xCB); break;
  case 204: R = dpct::ternary_logic_op(A, B, C, 0xCC); break;
  case 205: R = dpct::ternary_logic_op(A, B, C, 0xCD); break;
  case 206: R = dpct::ternary_logic_op(A, B, C, 0xCE); break;
  case 207: R = dpct::ternary_logic_op(A, B, C, 0xCF); break;
  case 208: R = dpct::ternary_logic_op(A, B, C, 0xD0); break;
  case 209: R = dpct::ternary_logic_op(A, B, C, 0xD1); break;
  case 210: R = dpct::ternary_logic_op(A, B, C, 0xD2); break;
  case 211: R = dpct::ternary_logic_op(A, B, C, 0xD3); break;
  case 212: R = dpct::ternary_logic_op(A, B, C, 0xD4); break;
  case 213: R = dpct::ternary_logic_op(A, B, C, 0xD5); break;
  case 214: R = dpct::ternary_logic_op(A, B, C, 0xD6); break;
  case 215: R = dpct::ternary_logic_op(A, B, C, 0xD7); break;
  case 216: R = dpct::ternary_logic_op(A, B, C, 0xD8); break;
  case 217: R = dpct::ternary_logic_op(A, B, C, 0xD9); break;
  case 218: R = dpct::ternary_logic_op(A, B, C, 0xDA); break;
  case 219: R = dpct::ternary_logic_op(A, B, C, 0xDB); break;
  case 220: R = dpct::ternary_logic_op(A, B, C, 0xDC); break;
  case 221: R = dpct::ternary_logic_op(A, B, C, 0xDD); break;
  case 222: R = dpct::ternary_logic_op(A, B, C, 0xDE); break;
  case 223: R = dpct::ternary_logic_op(A, B, C, 0xDF); break;
  case 224: R = dpct::ternary_logic_op(A, B, C, 0xE0); break;
  case 225: R = dpct::ternary_logic_op(A, B, C, 0xE1); break;
  case 226: R = dpct::ternary_logic_op(A, B, C, 0xE2); break;
  case 227: R = dpct::ternary_logic_op(A, B, C, 0xE3); break;
  case 228: R = dpct::ternary_logic_op(A, B, C, 0xE4); break;
  case 229: R = dpct::ternary_logic_op(A, B, C, 0xE5); break;
  case 230: R = dpct::ternary_logic_op(A, B, C, 0xE6); break;
  case 231: R = dpct::ternary_logic_op(A, B, C, 0xE7); break;
  case 232: R = dpct::ternary_logic_op(A, B, C, 0xE8); break;
  case 233: R = dpct::ternary_logic_op(A, B, C, 0xE9); break;
  case 234: R = dpct::ternary_logic_op(A, B, C, 0xEA); break;
  case 235: R = dpct::ternary_logic_op(A, B, C, 0xEB); break;
  case 236: R = dpct::ternary_logic_op(A, B, C, 0xEC); break;
  case 237: R = dpct::ternary_logic_op(A, B, C, 0xED); break;
  case 238: R = dpct::ternary_logic_op(A, B, C, 0xEE); break;
  case 239: R = dpct::ternary_logic_op(A, B, C, 0xEF); break;
  case 240: R = dpct::ternary_logic_op(A, B, C, 0xF0); break;
  case 241: R = dpct::ternary_logic_op(A, B, C, 0xF1); break;
  case 242: R = dpct::ternary_logic_op(A, B, C, 0xF2); break;
  case 243: R = dpct::ternary_logic_op(A, B, C, 0xF3); break;
  case 244: R = dpct::ternary_logic_op(A, B, C, 0xF4); break;
  case 245: R = dpct::ternary_logic_op(A, B, C, 0xF5); break;
  case 246: R = dpct::ternary_logic_op(A, B, C, 0xF6); break;
  case 247: R = dpct::ternary_logic_op(A, B, C, 0xF7); break;
  case 248: R = dpct::ternary_logic_op(A, B, C, 0xF8); break;
  case 249: R = dpct::ternary_logic_op(A, B, C, 0xF9); break;
  case 250: R = dpct::ternary_logic_op(A, B, C, 0xFA); break;
  case 251: R = dpct::ternary_logic_op(A, B, C, 0xFB); break;
  case 252: R = dpct::ternary_logic_op(A, B, C, 0xFC); break;
  case 253: R = dpct::ternary_logic_op(A, B, C, 0xFD); break;
  case 254: R = dpct::ternary_logic_op(A, B, C, 0xFE); break;
  case 255: R = dpct::ternary_logic_op(A, B, C, 0xFF); break;
  }
}

// clang-format on

void lop3(int *ec) {
  uint32_t X, Y, A = 1, B = 2, C = 3, D;
  for (D = 0; D < 256; ++D) {
    reference_of_lop3(X, A, B, C, D);
    asm_lop3(Y, A, B, C, D);
    if (X != Y) {
      *ec = D;
      return;
    }
  }
  *ec = 0;
}

template <int lut, typename T> inline T lop3(T a, T b, T c) {
  T res;
  res = dpct::ternary_logic_op(a, b, c, lut);
  return res;
}

void lop3_template(int *ec) {
  uint32_t X, Y, A = 1, B = 2, C = 3;
  reference_of_lop3(X, A, B, C, 0x8);
  Y = lop3<0x8>(A, B, C);
  if (X != Y) {
    *ec = Y;
    return;
  }

  *ec = 0;
}

int main() try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  int ret = 0;
  int *d_ec = nullptr;
  d_ec = sycl::malloc_device<int>(1, q_ct1);

  auto wait_and_check = [&](const char *case_name) {
    dpct::get_current_device().queues_wait_and_throw();
    int ec = 0;
    dpct::get_in_order_queue().memcpy(&ec, d_ec, sizeof(int)).wait();
    if (ec != 0)
      printf("Test %s failed: return code = %d\n", case_name, ec);
    ret = ret || ec;
  };

  q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
      [=](sycl::nd_item<3> item_ct1) { lop3(d_ec); });
  wait_and_check("lop3");

  q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
      [=](sycl::nd_item<3> item_ct1) { lop3_template(d_ec); });
  wait_and_check("lop3_template");

  dpct::dpct_free(d_ec, q_ct1);

  return ret;
} catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
