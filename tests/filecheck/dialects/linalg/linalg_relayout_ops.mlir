// RUN: scair-opt %s | filecheck %s

// Coverage for the data-layout ops, ported from MLIR's LinalgRelayoutOps.td.
//
// Both ops print in the generic form. Their upstream assembly format is built
// around `custom<DynamicIndexList>($inner_tiles, $static_inner_tiles)`, which
// interleaves the dynamic `inner_tiles` operands with the static
// `static_inner_tiles` entries; ScaIR has no custom-directive support, and the
// generic form is accepted by MLIR either way.

%src, %dst = "test.op"() : () -> (tensor<128x256xf32>, tensor<16x8x8x32xf32>)

// linalg.pack, with only the mandatory attributes.
%packed = "linalg.pack"(%src, %dst) <{
  inner_dims_pos = array<i64: 0, 1>,
  static_inner_tiles = array<i64: 8, 32>,
  operandSegmentSizes = array<i32: 1, 1, 0, 0>
}> : (tensor<128x256xf32>, tensor<16x8x8x32xf32>) -> tensor<16x8x8x32xf32>

// ... with an outer dimension permutation.
%dstp = "test.op"() : () -> tensor<8x16x8x32xf32>
%packed_perm = "linalg.pack"(%src, %dstp) <{
  inner_dims_pos = array<i64: 0, 1>,
  static_inner_tiles = array<i64: 8, 32>,
  outer_dims_perm = array<i64: 1, 0>,
  operandSegmentSizes = array<i32: 1, 1, 0, 0>
}> : (tensor<128x256xf32>, tensor<8x16x8x32xf32>) -> tensor<8x16x8x32xf32>

// ... with a padding value, for a source that does not tile evenly.
%pad, %psrc, %pdst = "test.op"() : () -> (f32, tensor<125x255xf32>, tensor<16x8x8x32xf32>)
%packed_pad = "linalg.pack"(%psrc, %pdst, %pad) <{
  inner_dims_pos = array<i64: 0, 1>,
  static_inner_tiles = array<i64: 8, 32>,
  operandSegmentSizes = array<i32: 1, 1, 1, 0>
}> : (tensor<125x255xf32>, tensor<16x8x8x32xf32>, f32) -> tensor<16x8x8x32xf32>

// ... and with a dynamic tile size, passed as an `inner_tiles` operand. The
// matching `static_inner_tiles` entry is the dynamic marker, which upstream is
// ShapedType::kDynamic = INT64_MIN, i.e. -9223372036854775808 -- not -1.
%dyn, %ddst = "test.op"() : () -> (index, tensor<16x?x8x?xf32>)
%packed_dyn = "linalg.pack"(%src, %ddst, %dyn) <{
  inner_dims_pos = array<i64: 0, 1>,
  static_inner_tiles = array<i64: 8, -9223372036854775808>,
  operandSegmentSizes = array<i32: 1, 1, 0, 1>
}> : (tensor<128x256xf32>, tensor<16x?x8x?xf32>, index) -> tensor<16x?x8x?xf32>

// linalg.unpack, the inverse. It has a single variadic operand group, so unlike
// linalg.pack it carries no operandSegmentSizes, exactly as upstream.
%unpacked = "linalg.unpack"(%dst, %src) <{
  inner_dims_pos = array<i64: 0, 1>,
  static_inner_tiles = array<i64: 8, 32>
}> : (tensor<16x8x8x32xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>

%unpacked_perm = "linalg.unpack"(%dstp, %src) <{
  inner_dims_pos = array<i64: 0, 1>,
  static_inner_tiles = array<i64: 8, 32>,
  outer_dims_perm = array<i64: 1, 0>
}> : (tensor<8x16x8x32xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>

%unpacked_dyn = "linalg.unpack"(%ddst, %src, %dyn) <{
  inner_dims_pos = array<i64: 0, 1>,
  static_inner_tiles = array<i64: 8, -9223372036854775808>
}> : (tensor<16x?x8x?xf32>, tensor<128x256xf32>, index) -> tensor<128x256xf32>

// CHECK:      builtin.module {
// CHECK-NEXT:   %0, %1 = "test.op"() : () -> (tensor<128x256xf32>, tensor<16x8x8x32xf32>)
// CHECK-NEXT:   %2 = "linalg.pack"(%0, %1) <{inner_dims_pos = array<i64: 0, 1>, static_inner_tiles = array<i64: 8, 32>}> : (tensor<128x256xf32>, tensor<16x8x8x32xf32>) -> tensor<16x8x8x32xf32>
// CHECK-NEXT:   %3 = "test.op"() : () -> tensor<8x16x8x32xf32>
// CHECK-NEXT:   %4 = "linalg.pack"(%0, %3) <{inner_dims_pos = array<i64: 0, 1>, static_inner_tiles = array<i64: 8, 32>, outer_dims_perm = array<i64: 1, 0>}> : (tensor<128x256xf32>, tensor<8x16x8x32xf32>) -> tensor<8x16x8x32xf32>
// CHECK-NEXT:   %5, %6, %7 = "test.op"() : () -> (f32, tensor<125x255xf32>, tensor<16x8x8x32xf32>)
// CHECK-NEXT:   %8 = "linalg.pack"(%6, %7, %5) <{inner_dims_pos = array<i64: 0, 1>, static_inner_tiles = array<i64: 8, 32>}> : (tensor<125x255xf32>, tensor<16x8x8x32xf32>, f32) -> tensor<16x8x8x32xf32>
// CHECK-NEXT:   %9, %10 = "test.op"() : () -> (index, tensor<16x?x8x?xf32>)
// CHECK-NEXT:   %11 = "linalg.pack"(%0, %10, %9) <{inner_dims_pos = array<i64: 0, 1>, static_inner_tiles = array<i64: 8, -9223372036854775808>}> : (tensor<128x256xf32>, tensor<16x?x8x?xf32>, index) -> tensor<16x?x8x?xf32>
// CHECK-NEXT:   %12 = "linalg.unpack"(%1, %0) <{inner_dims_pos = array<i64: 0, 1>, static_inner_tiles = array<i64: 8, 32>}> : (tensor<16x8x8x32xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
// CHECK-NEXT:   %13 = "linalg.unpack"(%3, %0) <{inner_dims_pos = array<i64: 0, 1>, static_inner_tiles = array<i64: 8, 32>, outer_dims_perm = array<i64: 1, 0>}> : (tensor<8x16x8x32xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
// CHECK-NEXT:   %14 = "linalg.unpack"(%10, %0, %9) <{inner_dims_pos = array<i64: 0, 1>, static_inner_tiles = array<i64: 8, -9223372036854775808>}> : (tensor<16x?x8x?xf32>, tensor<128x256xf32>, index) -> tensor<128x256xf32>
// CHECK-NEXT: }
