// RUN: scair-opt %s | filecheck %s

// Coverage for the named structured ops ported from MLIR's
// LinalgNamedStructuredOps.yaml. Every op is exercised in the generic form,
// which is what ScaIR prints for them and what MLIR accepts on input.
//
// Operand shapes are derived from each op's YAML indexing maps, evaluated over a
// concrete iteration domain with unit strides and dilations, so they are
// self-consistent rather than arbitrary.
//
// Two upstream attributes are deliberately absent:
//
//   * `strides` and `dilations`. Upstream they are `I64ElementsAttr`s, printed
//     as `dense<1> : tensor<2xi64>`. ScaIR's `DenseIntOrFPElementsAttr` always
//     reads integer elements as i32, so *any* i64 dense attribute fails
//     verification with "data element type i32 does not match expected type
//     i64". Both are optional upstream, so the IR here is still well formed.
//     Add them back once core can parse i64 dense attributes.
//
//   * All-equal dense attributes are additionally collapsed to a splat on print
//     (`dense<[1, 1]>` prints as `dense<1>`), which is a second reason the
//     default all-ones values would not round-trip today.
//
// The contraction ops (mmt4d, batch_mmt4d, matvec, vecmat, batch_matvec,
// batch_vecmat and dot) carry a real multiply-accumulate body rather than a
// trivial yield: MLIR's ContractionOpInterface verifier rejects anything else
// with "expected add/mul op in the body". Every other op keeps the trivial body.

%copy_a0, %copy_a1 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>)
%copy = "linalg.copy"(%copy_a0, %copy_a1) <{operandSegmentSizes = array<i32: 1, 1>, cast = #linalg.type_fn<cast_unsigned>}> ({
^bb0(%copy_b0: f32, %copy_b1: f32):
  "linalg.yield"(%copy_b0) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>)

%exp_a0, %exp_a1 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>)
%exp = "linalg.exp"(%exp_a0, %exp_a1) <{operandSegmentSizes = array<i32: 1, 1>}> ({
^bb0(%exp_b0: f32, %exp_b1: f32):
  "linalg.yield"(%exp_b0) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>)

%log_a0, %log_a1 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>)
%log = "linalg.log"(%log_a0, %log_a1) <{operandSegmentSizes = array<i32: 1, 1>}> ({
^bb0(%log_b0: f32, %log_b1: f32):
  "linalg.yield"(%log_b0) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>)

%abs_a0, %abs_a1 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>)
%abs = "linalg.abs"(%abs_a0, %abs_a1) <{operandSegmentSizes = array<i32: 1, 1>}> ({
^bb0(%abs_b0: f32, %abs_b1: f32):
  "linalg.yield"(%abs_b0) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>)

%ceil_a0, %ceil_a1 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>)
%ceil = "linalg.ceil"(%ceil_a0, %ceil_a1) <{operandSegmentSizes = array<i32: 1, 1>}> ({
^bb0(%ceil_b0: f32, %ceil_b1: f32):
  "linalg.yield"(%ceil_b0) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>)

%floor_a0, %floor_a1 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>)
%floor = "linalg.floor"(%floor_a0, %floor_a1) <{operandSegmentSizes = array<i32: 1, 1>}> ({
^bb0(%floor_b0: f32, %floor_b1: f32):
  "linalg.yield"(%floor_b0) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>)

%negf_a0, %negf_a1 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>)
%negf = "linalg.negf"(%negf_a0, %negf_a1) <{operandSegmentSizes = array<i32: 1, 1>}> ({
^bb0(%negf_b0: f32, %negf_b1: f32):
  "linalg.yield"(%negf_b0) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>)

%reciprocal_a0, %reciprocal_a1 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>)
%reciprocal = "linalg.reciprocal"(%reciprocal_a0, %reciprocal_a1) <{operandSegmentSizes = array<i32: 1, 1>}> ({
^bb0(%reciprocal_b0: f32, %reciprocal_b1: f32):
  "linalg.yield"(%reciprocal_b0) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>)

%round_a0, %round_a1 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>)
%round = "linalg.round"(%round_a0, %round_a1) <{operandSegmentSizes = array<i32: 1, 1>}> ({
^bb0(%round_b0: f32, %round_b1: f32):
  "linalg.yield"(%round_b0) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>)

%sqrt_a0, %sqrt_a1 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>)
%sqrt = "linalg.sqrt"(%sqrt_a0, %sqrt_a1) <{operandSegmentSizes = array<i32: 1, 1>}> ({
^bb0(%sqrt_b0: f32, %sqrt_b1: f32):
  "linalg.yield"(%sqrt_b0) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>)

%rsqrt_a0, %rsqrt_a1 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>)
%rsqrt = "linalg.rsqrt"(%rsqrt_a0, %rsqrt_a1) <{operandSegmentSizes = array<i32: 1, 1>}> ({
^bb0(%rsqrt_b0: f32, %rsqrt_b1: f32):
  "linalg.yield"(%rsqrt_b0) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>)

%square_a0, %square_a1 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>)
%square = "linalg.square"(%square_a0, %square_a1) <{operandSegmentSizes = array<i32: 1, 1>}> ({
^bb0(%square_b0: f32, %square_b1: f32):
  "linalg.yield"(%square_b0) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>)

%tanh_a0, %tanh_a1 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>)
%tanh = "linalg.tanh"(%tanh_a0, %tanh_a1) <{operandSegmentSizes = array<i32: 1, 1>}> ({
^bb0(%tanh_b0: f32, %tanh_b1: f32):
  "linalg.yield"(%tanh_b0) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>)

%erf_a0, %erf_a1 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>)
%erf = "linalg.erf"(%erf_a0, %erf_a1) <{operandSegmentSizes = array<i32: 1, 1>}> ({
^bb0(%erf_b0: f32, %erf_b1: f32):
  "linalg.yield"(%erf_b0) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>)

%add_a0, %add_a1, %add_a2 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
%add = "linalg.add"(%add_a0, %add_a1, %add_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%add_b0: f32, %add_b1: f32, %add_b2: f32):
  "linalg.yield"(%add_b0) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>)

%sub_a0, %sub_a1, %sub_a2 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
%sub = "linalg.sub"(%sub_a0, %sub_a1, %sub_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%sub_b0: f32, %sub_b1: f32, %sub_b2: f32):
  "linalg.yield"(%sub_b0) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>)

%mul_a0, %mul_a1, %mul_a2 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
%mul = "linalg.mul"(%mul_a0, %mul_a1, %mul_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%mul_b0: f32, %mul_b1: f32, %mul_b2: f32):
  "linalg.yield"(%mul_b0) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>)

%div_a0, %div_a1, %div_a2 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
%div = "linalg.div"(%div_a0, %div_a1, %div_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%div_b0: f32, %div_b1: f32, %div_b2: f32):
  "linalg.yield"(%div_b0) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>)

%div_unsigned_a0, %div_unsigned_a1, %div_unsigned_a2 = "test.op"() : () -> (tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>)
%div_unsigned = "linalg.div_unsigned"(%div_unsigned_a0, %div_unsigned_a1, %div_unsigned_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%div_unsigned_b0: i32, %div_unsigned_b1: i32, %div_unsigned_b2: i32):
  "linalg.yield"(%div_unsigned_b0) : (i32) -> ()
}) : (tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>) -> (tensor<4x8xi32>)

%max_a0, %max_a1, %max_a2 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
%max = "linalg.max"(%max_a0, %max_a1, %max_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%max_b0: f32, %max_b1: f32, %max_b2: f32):
  "linalg.yield"(%max_b0) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>)

%min_a0, %min_a1, %min_a2 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
%min = "linalg.min"(%min_a0, %min_a1, %min_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%min_b0: f32, %min_b1: f32, %min_b2: f32):
  "linalg.yield"(%min_b0) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>)

%powf_a0, %powf_a1, %powf_a2 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
%powf = "linalg.powf"(%powf_a0, %powf_a1, %powf_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%powf_b0: f32, %powf_b1: f32, %powf_b2: f32):
  "linalg.yield"(%powf_b0) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>)

%select_a0, %select_a1, %select_a2, %select_a3 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
%select = "linalg.select"(%select_a0, %select_a1, %select_a2, %select_a3) <{operandSegmentSizes = array<i32: 3, 1>}> ({
^bb0(%select_b0: f32, %select_b1: f32, %select_b2: f32, %select_b3: f32):
  "linalg.yield"(%select_b0) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>)

%quantized_matmul_a0, %quantized_matmul_a1, %quantized_matmul_a2, %quantized_matmul_a3, %quantized_matmul_a4 = "test.op"() : () -> (tensor<2x4xi32>, tensor<4x3xi32>, i32, i32, tensor<2x3xi32>)
%quantized_matmul = "linalg.quantized_matmul"(%quantized_matmul_a0, %quantized_matmul_a1, %quantized_matmul_a2, %quantized_matmul_a3, %quantized_matmul_a4) <{operandSegmentSizes = array<i32: 4, 1>}> ({
^bb0(%quantized_matmul_b0: i32, %quantized_matmul_b1: i32, %quantized_matmul_b2: i32, %quantized_matmul_b3: i32, %quantized_matmul_b4: i32):
  "linalg.yield"(%quantized_matmul_b0) : (i32) -> ()
}) : (tensor<2x4xi32>, tensor<4x3xi32>, i32, i32, tensor<2x3xi32>) -> (tensor<2x3xi32>)

%mmt4d_a0, %mmt4d_a1, %mmt4d_a2 = "test.op"() : () -> (tensor<2x4x5x7xf32>, tensor<3x4x6x7xf32>, tensor<2x3x5x6xf32>)
%mmt4d = "linalg.mmt4d"(%mmt4d_a0, %mmt4d_a1, %mmt4d_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%mmt4d_b0: f32, %mmt4d_b1: f32, %mmt4d_b2: f32):
  %mmt4d_p0 = "arith.mulf"(%mmt4d_b0, %mmt4d_b1) : (f32, f32) -> f32
  %mmt4d_p1 = "arith.addf"(%mmt4d_b2, %mmt4d_p0) : (f32, f32) -> f32
  "linalg.yield"(%mmt4d_p1) : (f32) -> ()
}) : (tensor<2x4x5x7xf32>, tensor<3x4x6x7xf32>, tensor<2x3x5x6xf32>) -> (tensor<2x3x5x6xf32>)

%batch_mmt4d_a0, %batch_mmt4d_a1, %batch_mmt4d_a2 = "test.op"() : () -> (tensor<2x3x5x6x8xf32>, tensor<2x4x5x7x8xf32>, tensor<2x3x4x6x7xf32>)
%batch_mmt4d = "linalg.batch_mmt4d"(%batch_mmt4d_a0, %batch_mmt4d_a1, %batch_mmt4d_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%batch_mmt4d_b0: f32, %batch_mmt4d_b1: f32, %batch_mmt4d_b2: f32):
  %batch_mmt4d_p0 = "arith.mulf"(%batch_mmt4d_b0, %batch_mmt4d_b1) : (f32, f32) -> f32
  %batch_mmt4d_p1 = "arith.addf"(%batch_mmt4d_b2, %batch_mmt4d_p0) : (f32, f32) -> f32
  "linalg.yield"(%batch_mmt4d_p1) : (f32) -> ()
}) : (tensor<2x3x5x6x8xf32>, tensor<2x4x5x7x8xf32>, tensor<2x3x4x6x7xf32>) -> (tensor<2x3x4x6x7xf32>)

%quantized_batch_matmul_a0, %quantized_batch_matmul_a1, %quantized_batch_matmul_a2, %quantized_batch_matmul_a3, %quantized_batch_matmul_a4 = "test.op"() : () -> (tensor<2x3x5xi32>, tensor<2x5x4xi32>, i32, i32, tensor<2x3x4xi32>)
%quantized_batch_matmul = "linalg.quantized_batch_matmul"(%quantized_batch_matmul_a0, %quantized_batch_matmul_a1, %quantized_batch_matmul_a2, %quantized_batch_matmul_a3, %quantized_batch_matmul_a4) <{operandSegmentSizes = array<i32: 4, 1>}> ({
^bb0(%quantized_batch_matmul_b0: i32, %quantized_batch_matmul_b1: i32, %quantized_batch_matmul_b2: i32, %quantized_batch_matmul_b3: i32, %quantized_batch_matmul_b4: i32):
  "linalg.yield"(%quantized_batch_matmul_b0) : (i32) -> ()
}) : (tensor<2x3x5xi32>, tensor<2x5x4xi32>, i32, i32, tensor<2x3x4xi32>) -> (tensor<2x3x4xi32>)

%matvec_a0, %matvec_a1, %matvec_a2 = "test.op"() : () -> (tensor<2x3xf32>, tensor<3xf32>, tensor<2xf32>)
%matvec = "linalg.matvec"(%matvec_a0, %matvec_a1, %matvec_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%matvec_b0: f32, %matvec_b1: f32, %matvec_b2: f32):
  %matvec_p0 = "arith.mulf"(%matvec_b0, %matvec_b1) : (f32, f32) -> f32
  %matvec_p1 = "arith.addf"(%matvec_b2, %matvec_p0) : (f32, f32) -> f32
  "linalg.yield"(%matvec_p1) : (f32) -> ()
}) : (tensor<2x3xf32>, tensor<3xf32>, tensor<2xf32>) -> (tensor<2xf32>)

%vecmat_a0, %vecmat_a1, %vecmat_a2 = "test.op"() : () -> (tensor<3xf32>, tensor<3x2xf32>, tensor<2xf32>)
%vecmat = "linalg.vecmat"(%vecmat_a0, %vecmat_a1, %vecmat_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%vecmat_b0: f32, %vecmat_b1: f32, %vecmat_b2: f32):
  %vecmat_p0 = "arith.mulf"(%vecmat_b0, %vecmat_b1) : (f32, f32) -> f32
  %vecmat_p1 = "arith.addf"(%vecmat_b2, %vecmat_p0) : (f32, f32) -> f32
  "linalg.yield"(%vecmat_p1) : (f32) -> ()
}) : (tensor<3xf32>, tensor<3x2xf32>, tensor<2xf32>) -> (tensor<2xf32>)

%batch_matvec_a0, %batch_matvec_a1, %batch_matvec_a2 = "test.op"() : () -> (tensor<2x3x4xf32>, tensor<2x4xf32>, tensor<2x3xf32>)
%batch_matvec = "linalg.batch_matvec"(%batch_matvec_a0, %batch_matvec_a1, %batch_matvec_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%batch_matvec_b0: f32, %batch_matvec_b1: f32, %batch_matvec_b2: f32):
  %batch_matvec_p0 = "arith.mulf"(%batch_matvec_b0, %batch_matvec_b1) : (f32, f32) -> f32
  %batch_matvec_p1 = "arith.addf"(%batch_matvec_b2, %batch_matvec_p0) : (f32, f32) -> f32
  "linalg.yield"(%batch_matvec_p1) : (f32) -> ()
}) : (tensor<2x3x4xf32>, tensor<2x4xf32>, tensor<2x3xf32>) -> (tensor<2x3xf32>)

%batch_vecmat_a0, %batch_vecmat_a1, %batch_vecmat_a2 = "test.op"() : () -> (tensor<2x4xf32>, tensor<2x4x3xf32>, tensor<2x3xf32>)
%batch_vecmat = "linalg.batch_vecmat"(%batch_vecmat_a0, %batch_vecmat_a1, %batch_vecmat_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%batch_vecmat_b0: f32, %batch_vecmat_b1: f32, %batch_vecmat_b2: f32):
  %batch_vecmat_p0 = "arith.mulf"(%batch_vecmat_b0, %batch_vecmat_b1) : (f32, f32) -> f32
  %batch_vecmat_p1 = "arith.addf"(%batch_vecmat_b2, %batch_vecmat_p0) : (f32, f32) -> f32
  "linalg.yield"(%batch_vecmat_p1) : (f32) -> ()
}) : (tensor<2x4xf32>, tensor<2x4x3xf32>, tensor<2x3xf32>) -> (tensor<2x3xf32>)

%dot_a0, %dot_a1, %dot_a2 = "test.op"() : () -> (tensor<2xf32>, tensor<2xf32>, tensor<f32>)
%dot = "linalg.dot"(%dot_a0, %dot_a1, %dot_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%dot_b0: f32, %dot_b1: f32, %dot_b2: f32):
  %dot_p0 = "arith.mulf"(%dot_b0, %dot_b1) : (f32, f32) -> f32
  %dot_p1 = "arith.addf"(%dot_b2, %dot_p0) : (f32, f32) -> f32
  "linalg.yield"(%dot_p1) : (f32) -> ()
}) : (tensor<2xf32>, tensor<2xf32>, tensor<f32>) -> (tensor<f32>)

%conv_1d_a0, %conv_1d_a1, %conv_1d_a2 = "test.op"() : () -> (tensor<4xf32>, tensor<3xf32>, tensor<2xf32>)
%conv_1d = "linalg.conv_1d"(%conv_1d_a0, %conv_1d_a1, %conv_1d_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%conv_1d_b0: f32, %conv_1d_b1: f32, %conv_1d_b2: f32):
  "linalg.yield"(%conv_1d_b0) : (f32) -> ()
}) : (tensor<4xf32>, tensor<3xf32>, tensor<2xf32>) -> (tensor<2xf32>)

%conv_2d_a0, %conv_2d_a1, %conv_2d_a2 = "test.op"() : () -> (tensor<5x7xf32>, tensor<4x5xf32>, tensor<2x3xf32>)
%conv_2d = "linalg.conv_2d"(%conv_2d_a0, %conv_2d_a1, %conv_2d_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%conv_2d_b0: f32, %conv_2d_b1: f32, %conv_2d_b2: f32):
  "linalg.yield"(%conv_2d_b0) : (f32) -> ()
}) : (tensor<5x7xf32>, tensor<4x5xf32>, tensor<2x3xf32>) -> (tensor<2x3xf32>)

%conv_3d_a0, %conv_3d_a1, %conv_3d_a2 = "test.op"() : () -> (tensor<6x8x10xf32>, tensor<5x6x7xf32>, tensor<2x3x4xf32>)
%conv_3d = "linalg.conv_3d"(%conv_3d_a0, %conv_3d_a1, %conv_3d_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%conv_3d_b0: f32, %conv_3d_b1: f32, %conv_3d_b2: f32):
  "linalg.yield"(%conv_3d_b0) : (f32) -> ()
}) : (tensor<6x8x10xf32>, tensor<5x6x7xf32>, tensor<2x3x4xf32>) -> (tensor<2x3x4xf32>)

%conv_1d_nwc_wcf_a0, %conv_1d_nwc_wcf_a1, %conv_1d_nwc_wcf_a2 = "test.op"() : () -> (tensor<2x7x6xf32>, tensor<5x6x4xf32>, tensor<2x3x4xf32>)
%conv_1d_nwc_wcf = "linalg.conv_1d_nwc_wcf"(%conv_1d_nwc_wcf_a0, %conv_1d_nwc_wcf_a1, %conv_1d_nwc_wcf_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%conv_1d_nwc_wcf_b0: f32, %conv_1d_nwc_wcf_b1: f32, %conv_1d_nwc_wcf_b2: f32):
  "linalg.yield"(%conv_1d_nwc_wcf_b0) : (f32) -> ()
}) : (tensor<2x7x6xf32>, tensor<5x6x4xf32>, tensor<2x3x4xf32>) -> (tensor<2x3x4xf32>)

%conv_1d_ncw_fcw_a0, %conv_1d_ncw_fcw_a1, %conv_1d_ncw_fcw_a2 = "test.op"() : () -> (tensor<2x5x9xf32>, tensor<3x5x6xf32>, tensor<2x3x4xf32>)
%conv_1d_ncw_fcw = "linalg.conv_1d_ncw_fcw"(%conv_1d_ncw_fcw_a0, %conv_1d_ncw_fcw_a1, %conv_1d_ncw_fcw_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%conv_1d_ncw_fcw_b0: f32, %conv_1d_ncw_fcw_b1: f32, %conv_1d_ncw_fcw_b2: f32):
  "linalg.yield"(%conv_1d_ncw_fcw_b0) : (f32) -> ()
}) : (tensor<2x5x9xf32>, tensor<3x5x6xf32>, tensor<2x3x4xf32>) -> (tensor<2x3x4xf32>)

%conv_2d_nhwc_hwcf_a0, %conv_2d_nhwc_hwcf_a1, %conv_2d_nhwc_hwcf_a2 = "test.op"() : () -> (tensor<2x8x10x8xf32>, tensor<6x7x8x5xf32>, tensor<2x3x4x5xf32>)
%conv_2d_nhwc_hwcf = "linalg.conv_2d_nhwc_hwcf"(%conv_2d_nhwc_hwcf_a0, %conv_2d_nhwc_hwcf_a1, %conv_2d_nhwc_hwcf_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%conv_2d_nhwc_hwcf_b0: f32, %conv_2d_nhwc_hwcf_b1: f32, %conv_2d_nhwc_hwcf_b2: f32):
  "linalg.yield"(%conv_2d_nhwc_hwcf_b0) : (f32) -> ()
}) : (tensor<2x8x10x8xf32>, tensor<6x7x8x5xf32>, tensor<2x3x4x5xf32>) -> (tensor<2x3x4x5xf32>)

%conv_2d_nhwc_fhwc_a0, %conv_2d_nhwc_fhwc_a1, %conv_2d_nhwc_fhwc_a2 = "test.op"() : () -> (tensor<2x8x10x8xf32>, tensor<5x6x7x8xf32>, tensor<2x3x4x5xf32>)
%conv_2d_nhwc_fhwc = "linalg.conv_2d_nhwc_fhwc"(%conv_2d_nhwc_fhwc_a0, %conv_2d_nhwc_fhwc_a1, %conv_2d_nhwc_fhwc_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%conv_2d_nhwc_fhwc_b0: f32, %conv_2d_nhwc_fhwc_b1: f32, %conv_2d_nhwc_fhwc_b2: f32):
  "linalg.yield"(%conv_2d_nhwc_fhwc_b0) : (f32) -> ()
}) : (tensor<2x8x10x8xf32>, tensor<5x6x7x8xf32>, tensor<2x3x4x5xf32>) -> (tensor<2x3x4x5xf32>)

%conv_2d_nhwc_hwcf_q_a0, %conv_2d_nhwc_hwcf_q_a1, %conv_2d_nhwc_hwcf_q_a2, %conv_2d_nhwc_hwcf_q_a3, %conv_2d_nhwc_hwcf_q_a4 = "test.op"() : () -> (tensor<2x8x10x8xi32>, tensor<6x7x8x5xi32>, i32, i32, tensor<2x3x4x5xi32>)
%conv_2d_nhwc_hwcf_q = "linalg.conv_2d_nhwc_hwcf_q"(%conv_2d_nhwc_hwcf_q_a0, %conv_2d_nhwc_hwcf_q_a1, %conv_2d_nhwc_hwcf_q_a2, %conv_2d_nhwc_hwcf_q_a3, %conv_2d_nhwc_hwcf_q_a4) <{operandSegmentSizes = array<i32: 4, 1>}> ({
^bb0(%conv_2d_nhwc_hwcf_q_b0: i32, %conv_2d_nhwc_hwcf_q_b1: i32, %conv_2d_nhwc_hwcf_q_b2: i32, %conv_2d_nhwc_hwcf_q_b3: i32, %conv_2d_nhwc_hwcf_q_b4: i32):
  "linalg.yield"(%conv_2d_nhwc_hwcf_q_b0) : (i32) -> ()
}) : (tensor<2x8x10x8xi32>, tensor<6x7x8x5xi32>, i32, i32, tensor<2x3x4x5xi32>) -> (tensor<2x3x4x5xi32>)

%conv_2d_nhwc_fhwc_q_a0, %conv_2d_nhwc_fhwc_q_a1, %conv_2d_nhwc_fhwc_q_a2, %conv_2d_nhwc_fhwc_q_a3, %conv_2d_nhwc_fhwc_q_a4 = "test.op"() : () -> (tensor<2x8x10x8xi32>, tensor<5x6x7x8xi32>, i32, i32, tensor<2x3x4x5xi32>)
%conv_2d_nhwc_fhwc_q = "linalg.conv_2d_nhwc_fhwc_q"(%conv_2d_nhwc_fhwc_q_a0, %conv_2d_nhwc_fhwc_q_a1, %conv_2d_nhwc_fhwc_q_a2, %conv_2d_nhwc_fhwc_q_a3, %conv_2d_nhwc_fhwc_q_a4) <{operandSegmentSizes = array<i32: 4, 1>}> ({
^bb0(%conv_2d_nhwc_fhwc_q_b0: i32, %conv_2d_nhwc_fhwc_q_b1: i32, %conv_2d_nhwc_fhwc_q_b2: i32, %conv_2d_nhwc_fhwc_q_b3: i32, %conv_2d_nhwc_fhwc_q_b4: i32):
  "linalg.yield"(%conv_2d_nhwc_fhwc_q_b0) : (i32) -> ()
}) : (tensor<2x8x10x8xi32>, tensor<5x6x7x8xi32>, i32, i32, tensor<2x3x4x5xi32>) -> (tensor<2x3x4x5xi32>)

%conv_2d_nchw_fchw_q_a0, %conv_2d_nchw_fchw_q_a1, %conv_2d_nchw_fchw_q_a2, %conv_2d_nchw_fchw_q_a3, %conv_2d_nchw_fchw_q_a4 = "test.op"() : () -> (tensor<2x6x10x12xi32>, tensor<3x6x7x8xi32>, i32, i32, tensor<2x3x4x5xi32>)
%conv_2d_nchw_fchw_q = "linalg.conv_2d_nchw_fchw_q"(%conv_2d_nchw_fchw_q_a0, %conv_2d_nchw_fchw_q_a1, %conv_2d_nchw_fchw_q_a2, %conv_2d_nchw_fchw_q_a3, %conv_2d_nchw_fchw_q_a4) <{operandSegmentSizes = array<i32: 4, 1>}> ({
^bb0(%conv_2d_nchw_fchw_q_b0: i32, %conv_2d_nchw_fchw_q_b1: i32, %conv_2d_nchw_fchw_q_b2: i32, %conv_2d_nchw_fchw_q_b3: i32, %conv_2d_nchw_fchw_q_b4: i32):
  "linalg.yield"(%conv_2d_nchw_fchw_q_b0) : (i32) -> ()
}) : (tensor<2x6x10x12xi32>, tensor<3x6x7x8xi32>, i32, i32, tensor<2x3x4x5xi32>) -> (tensor<2x3x4x5xi32>)

%conv_2d_nchw_fchw_a0, %conv_2d_nchw_fchw_a1, %conv_2d_nchw_fchw_a2 = "test.op"() : () -> (tensor<2x6x10x12xf32>, tensor<3x6x7x8xf32>, tensor<2x3x4x5xf32>)
%conv_2d_nchw_fchw = "linalg.conv_2d_nchw_fchw"(%conv_2d_nchw_fchw_a0, %conv_2d_nchw_fchw_a1, %conv_2d_nchw_fchw_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%conv_2d_nchw_fchw_b0: f32, %conv_2d_nchw_fchw_b1: f32, %conv_2d_nchw_fchw_b2: f32):
  "linalg.yield"(%conv_2d_nchw_fchw_b0) : (f32) -> ()
}) : (tensor<2x6x10x12xf32>, tensor<3x6x7x8xf32>, tensor<2x3x4x5xf32>) -> (tensor<2x3x4x5xf32>)

%conv_2d_ngchw_fgchw_a0, %conv_2d_ngchw_fgchw_a1, %conv_2d_ngchw_fgchw_a2 = "test.op"() : () -> (tensor<2x3x7x12x14xf32>, tensor<4x3x7x8x9xf32>, tensor<2x3x4x5x6xf32>)
%conv_2d_ngchw_fgchw = "linalg.conv_2d_ngchw_fgchw"(%conv_2d_ngchw_fgchw_a0, %conv_2d_ngchw_fgchw_a1, %conv_2d_ngchw_fgchw_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%conv_2d_ngchw_fgchw_b0: f32, %conv_2d_ngchw_fgchw_b1: f32, %conv_2d_ngchw_fgchw_b2: f32):
  "linalg.yield"(%conv_2d_ngchw_fgchw_b0) : (f32) -> ()
}) : (tensor<2x3x7x12x14xf32>, tensor<4x3x7x8x9xf32>, tensor<2x3x4x5x6xf32>) -> (tensor<2x3x4x5x6xf32>)

%conv_2d_ngchw_gfchw_a0, %conv_2d_ngchw_gfchw_a1, %conv_2d_ngchw_gfchw_a2 = "test.op"() : () -> (tensor<2x3x7x12x14xf32>, tensor<3x4x7x8x9xf32>, tensor<2x3x4x5x6xf32>)
%conv_2d_ngchw_gfchw = "linalg.conv_2d_ngchw_gfchw"(%conv_2d_ngchw_gfchw_a0, %conv_2d_ngchw_gfchw_a1, %conv_2d_ngchw_gfchw_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%conv_2d_ngchw_gfchw_b0: f32, %conv_2d_ngchw_gfchw_b1: f32, %conv_2d_ngchw_gfchw_b2: f32):
  "linalg.yield"(%conv_2d_ngchw_gfchw_b0) : (f32) -> ()
}) : (tensor<2x3x7x12x14xf32>, tensor<3x4x7x8x9xf32>, tensor<2x3x4x5x6xf32>) -> (tensor<2x3x4x5x6xf32>)

%conv_2d_nhwgc_gfhwc_a0, %conv_2d_nhwgc_gfhwc_a1, %conv_2d_nhwgc_gfhwc_a2 = "test.op"() : () -> (tensor<2x9x11x5x9xf32>, tensor<5x6x7x8x9xf32>, tensor<2x3x4x5x6xf32>)
%conv_2d_nhwgc_gfhwc = "linalg.conv_2d_nhwgc_gfhwc"(%conv_2d_nhwgc_gfhwc_a0, %conv_2d_nhwgc_gfhwc_a1, %conv_2d_nhwgc_gfhwc_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%conv_2d_nhwgc_gfhwc_b0: f32, %conv_2d_nhwgc_gfhwc_b1: f32, %conv_2d_nhwgc_gfhwc_b2: f32):
  "linalg.yield"(%conv_2d_nhwgc_gfhwc_b0) : (f32) -> ()
}) : (tensor<2x9x11x5x9xf32>, tensor<5x6x7x8x9xf32>, tensor<2x3x4x5x6xf32>) -> (tensor<2x3x4x5x6xf32>)

%conv_2d_nhwgc_gfhwc_q_a0, %conv_2d_nhwgc_gfhwc_q_a1, %conv_2d_nhwgc_gfhwc_q_a2, %conv_2d_nhwgc_gfhwc_q_a3, %conv_2d_nhwgc_gfhwc_q_a4 = "test.op"() : () -> (tensor<2x9x11x5x9xi32>, tensor<5x6x7x8x9xi32>, i32, i32, tensor<2x3x4x5x6xi32>)
%conv_2d_nhwgc_gfhwc_q = "linalg.conv_2d_nhwgc_gfhwc_q"(%conv_2d_nhwgc_gfhwc_q_a0, %conv_2d_nhwgc_gfhwc_q_a1, %conv_2d_nhwgc_gfhwc_q_a2, %conv_2d_nhwgc_gfhwc_q_a3, %conv_2d_nhwgc_gfhwc_q_a4) <{operandSegmentSizes = array<i32: 4, 1>}> ({
^bb0(%conv_2d_nhwgc_gfhwc_q_b0: i32, %conv_2d_nhwgc_gfhwc_q_b1: i32, %conv_2d_nhwgc_gfhwc_q_b2: i32, %conv_2d_nhwgc_gfhwc_q_b3: i32, %conv_2d_nhwgc_gfhwc_q_b4: i32):
  "linalg.yield"(%conv_2d_nhwgc_gfhwc_q_b0) : (i32) -> ()
}) : (tensor<2x9x11x5x9xi32>, tensor<5x6x7x8x9xi32>, i32, i32, tensor<2x3x4x5x6xi32>) -> (tensor<2x3x4x5x6xi32>)

%conv_2d_ngchw_gfchw_q_a0, %conv_2d_ngchw_gfchw_q_a1, %conv_2d_ngchw_gfchw_q_a2, %conv_2d_ngchw_gfchw_q_a3, %conv_2d_ngchw_gfchw_q_a4 = "test.op"() : () -> (tensor<2x3x7x12x14xi32>, tensor<3x4x7x8x9xi32>, i32, i32, tensor<2x3x4x5x6xi32>)
%conv_2d_ngchw_gfchw_q = "linalg.conv_2d_ngchw_gfchw_q"(%conv_2d_ngchw_gfchw_q_a0, %conv_2d_ngchw_gfchw_q_a1, %conv_2d_ngchw_gfchw_q_a2, %conv_2d_ngchw_gfchw_q_a3, %conv_2d_ngchw_gfchw_q_a4) <{operandSegmentSizes = array<i32: 4, 1>}> ({
^bb0(%conv_2d_ngchw_gfchw_q_b0: i32, %conv_2d_ngchw_gfchw_q_b1: i32, %conv_2d_ngchw_gfchw_q_b2: i32, %conv_2d_ngchw_gfchw_q_b3: i32, %conv_2d_ngchw_gfchw_q_b4: i32):
  "linalg.yield"(%conv_2d_ngchw_gfchw_q_b0) : (i32) -> ()
}) : (tensor<2x3x7x12x14xi32>, tensor<3x4x7x8x9xi32>, i32, i32, tensor<2x3x4x5x6xi32>) -> (tensor<2x3x4x5x6xi32>)

%conv_3d_ndhwc_dhwcf_a0, %conv_3d_ndhwc_dhwcf_a1, %conv_3d_ndhwc_dhwcf_a2 = "test.op"() : () -> (tensor<2x9x11x13x10xf32>, tensor<7x8x9x10x6xf32>, tensor<2x3x4x5x6xf32>)
%conv_3d_ndhwc_dhwcf = "linalg.conv_3d_ndhwc_dhwcf"(%conv_3d_ndhwc_dhwcf_a0, %conv_3d_ndhwc_dhwcf_a1, %conv_3d_ndhwc_dhwcf_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%conv_3d_ndhwc_dhwcf_b0: f32, %conv_3d_ndhwc_dhwcf_b1: f32, %conv_3d_ndhwc_dhwcf_b2: f32):
  "linalg.yield"(%conv_3d_ndhwc_dhwcf_b0) : (f32) -> ()
}) : (tensor<2x9x11x13x10xf32>, tensor<7x8x9x10x6xf32>, tensor<2x3x4x5x6xf32>) -> (tensor<2x3x4x5x6xf32>)

%conv_3d_ndhwc_dhwcf_q_a0, %conv_3d_ndhwc_dhwcf_q_a1, %conv_3d_ndhwc_dhwcf_q_a2, %conv_3d_ndhwc_dhwcf_q_a3, %conv_3d_ndhwc_dhwcf_q_a4 = "test.op"() : () -> (tensor<2x9x11x13x10xi32>, tensor<7x8x9x10x6xi32>, i32, i32, tensor<2x3x4x5x6xi32>)
%conv_3d_ndhwc_dhwcf_q = "linalg.conv_3d_ndhwc_dhwcf_q"(%conv_3d_ndhwc_dhwcf_q_a0, %conv_3d_ndhwc_dhwcf_q_a1, %conv_3d_ndhwc_dhwcf_q_a2, %conv_3d_ndhwc_dhwcf_q_a3, %conv_3d_ndhwc_dhwcf_q_a4) <{operandSegmentSizes = array<i32: 4, 1>}> ({
^bb0(%conv_3d_ndhwc_dhwcf_q_b0: i32, %conv_3d_ndhwc_dhwcf_q_b1: i32, %conv_3d_ndhwc_dhwcf_q_b2: i32, %conv_3d_ndhwc_dhwcf_q_b3: i32, %conv_3d_ndhwc_dhwcf_q_b4: i32):
  "linalg.yield"(%conv_3d_ndhwc_dhwcf_q_b0) : (i32) -> ()
}) : (tensor<2x9x11x13x10xi32>, tensor<7x8x9x10x6xi32>, i32, i32, tensor<2x3x4x5x6xi32>) -> (tensor<2x3x4x5x6xi32>)

%conv_3d_ncdhw_fcdhw_a0, %conv_3d_ncdhw_fcdhw_a1, %conv_3d_ncdhw_fcdhw_a2 = "test.op"() : () -> (tensor<2x7x11x13x15xf32>, tensor<3x7x8x9x10xf32>, tensor<2x3x4x5x6xf32>)
%conv_3d_ncdhw_fcdhw = "linalg.conv_3d_ncdhw_fcdhw"(%conv_3d_ncdhw_fcdhw_a0, %conv_3d_ncdhw_fcdhw_a1, %conv_3d_ncdhw_fcdhw_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%conv_3d_ncdhw_fcdhw_b0: f32, %conv_3d_ncdhw_fcdhw_b1: f32, %conv_3d_ncdhw_fcdhw_b2: f32):
  "linalg.yield"(%conv_3d_ncdhw_fcdhw_b0) : (f32) -> ()
}) : (tensor<2x7x11x13x15xf32>, tensor<3x7x8x9x10xf32>, tensor<2x3x4x5x6xf32>) -> (tensor<2x3x4x5x6xf32>)

%depthwise_conv_1d_nwc_wc_a0, %depthwise_conv_1d_nwc_wc_a1, %depthwise_conv_1d_nwc_wc_a2 = "test.op"() : () -> (tensor<2x7x4xf32>, tensor<5x4xf32>, tensor<2x3x4xf32>)
%depthwise_conv_1d_nwc_wc = "linalg.depthwise_conv_1d_nwc_wc"(%depthwise_conv_1d_nwc_wc_a0, %depthwise_conv_1d_nwc_wc_a1, %depthwise_conv_1d_nwc_wc_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%depthwise_conv_1d_nwc_wc_b0: f32, %depthwise_conv_1d_nwc_wc_b1: f32, %depthwise_conv_1d_nwc_wc_b2: f32):
  "linalg.yield"(%depthwise_conv_1d_nwc_wc_b0) : (f32) -> ()
}) : (tensor<2x7x4xf32>, tensor<5x4xf32>, tensor<2x3x4xf32>) -> (tensor<2x3x4xf32>)

%depthwise_conv_1d_ncw_cw_a0, %depthwise_conv_1d_ncw_cw_a1, %depthwise_conv_1d_ncw_cw_a2 = "test.op"() : () -> (tensor<2x4x7xf32>, tensor<4x5xf32>, tensor<2x4x3xf32>)
%depthwise_conv_1d_ncw_cw = "linalg.depthwise_conv_1d_ncw_cw"(%depthwise_conv_1d_ncw_cw_a0, %depthwise_conv_1d_ncw_cw_a1, %depthwise_conv_1d_ncw_cw_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%depthwise_conv_1d_ncw_cw_b0: f32, %depthwise_conv_1d_ncw_cw_b1: f32, %depthwise_conv_1d_ncw_cw_b2: f32):
  "linalg.yield"(%depthwise_conv_1d_ncw_cw_b0) : (f32) -> ()
}) : (tensor<2x4x7xf32>, tensor<4x5xf32>, tensor<2x4x3xf32>) -> (tensor<2x4x3xf32>)

%depthwise_conv_1d_nwc_wcm_a0, %depthwise_conv_1d_nwc_wcm_a1, %depthwise_conv_1d_nwc_wcm_a2 = "test.op"() : () -> (tensor<2x8x4xf32>, tensor<6x4x5xf32>, tensor<2x3x4x5xf32>)
%depthwise_conv_1d_nwc_wcm = "linalg.depthwise_conv_1d_nwc_wcm"(%depthwise_conv_1d_nwc_wcm_a0, %depthwise_conv_1d_nwc_wcm_a1, %depthwise_conv_1d_nwc_wcm_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%depthwise_conv_1d_nwc_wcm_b0: f32, %depthwise_conv_1d_nwc_wcm_b1: f32, %depthwise_conv_1d_nwc_wcm_b2: f32):
  "linalg.yield"(%depthwise_conv_1d_nwc_wcm_b0) : (f32) -> ()
}) : (tensor<2x8x4xf32>, tensor<6x4x5xf32>, tensor<2x3x4x5xf32>) -> (tensor<2x3x4x5xf32>)

%depthwise_conv_2d_nhwc_hwc_a0, %depthwise_conv_2d_nhwc_hwc_a1, %depthwise_conv_2d_nhwc_hwc_a2 = "test.op"() : () -> (tensor<2x8x10x5xf32>, tensor<6x7x5xf32>, tensor<2x3x4x5xf32>)
%depthwise_conv_2d_nhwc_hwc = "linalg.depthwise_conv_2d_nhwc_hwc"(%depthwise_conv_2d_nhwc_hwc_a0, %depthwise_conv_2d_nhwc_hwc_a1, %depthwise_conv_2d_nhwc_hwc_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%depthwise_conv_2d_nhwc_hwc_b0: f32, %depthwise_conv_2d_nhwc_hwc_b1: f32, %depthwise_conv_2d_nhwc_hwc_b2: f32):
  "linalg.yield"(%depthwise_conv_2d_nhwc_hwc_b0) : (f32) -> ()
}) : (tensor<2x8x10x5xf32>, tensor<6x7x5xf32>, tensor<2x3x4x5xf32>) -> (tensor<2x3x4x5xf32>)

%depthwise_conv_2d_nchw_chw_a0, %depthwise_conv_2d_nchw_chw_a1, %depthwise_conv_2d_nchw_chw_a2 = "test.op"() : () -> (tensor<2x5x8x10xf32>, tensor<5x6x7xf32>, tensor<2x5x3x4xf32>)
%depthwise_conv_2d_nchw_chw = "linalg.depthwise_conv_2d_nchw_chw"(%depthwise_conv_2d_nchw_chw_a0, %depthwise_conv_2d_nchw_chw_a1, %depthwise_conv_2d_nchw_chw_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%depthwise_conv_2d_nchw_chw_b0: f32, %depthwise_conv_2d_nchw_chw_b1: f32, %depthwise_conv_2d_nchw_chw_b2: f32):
  "linalg.yield"(%depthwise_conv_2d_nchw_chw_b0) : (f32) -> ()
}) : (tensor<2x5x8x10xf32>, tensor<5x6x7xf32>, tensor<2x5x3x4xf32>) -> (tensor<2x5x3x4xf32>)

%depthwise_conv_2d_nhwc_hwc_q_a0, %depthwise_conv_2d_nhwc_hwc_q_a1, %depthwise_conv_2d_nhwc_hwc_q_a2, %depthwise_conv_2d_nhwc_hwc_q_a3, %depthwise_conv_2d_nhwc_hwc_q_a4 = "test.op"() : () -> (tensor<2x8x10x5xi32>, tensor<6x7x5xi32>, i32, i32, tensor<2x3x4x5xi32>)
%depthwise_conv_2d_nhwc_hwc_q = "linalg.depthwise_conv_2d_nhwc_hwc_q"(%depthwise_conv_2d_nhwc_hwc_q_a0, %depthwise_conv_2d_nhwc_hwc_q_a1, %depthwise_conv_2d_nhwc_hwc_q_a2, %depthwise_conv_2d_nhwc_hwc_q_a3, %depthwise_conv_2d_nhwc_hwc_q_a4) <{operandSegmentSizes = array<i32: 4, 1>}> ({
^bb0(%depthwise_conv_2d_nhwc_hwc_q_b0: i32, %depthwise_conv_2d_nhwc_hwc_q_b1: i32, %depthwise_conv_2d_nhwc_hwc_q_b2: i32, %depthwise_conv_2d_nhwc_hwc_q_b3: i32, %depthwise_conv_2d_nhwc_hwc_q_b4: i32):
  "linalg.yield"(%depthwise_conv_2d_nhwc_hwc_q_b0) : (i32) -> ()
}) : (tensor<2x8x10x5xi32>, tensor<6x7x5xi32>, i32, i32, tensor<2x3x4x5xi32>) -> (tensor<2x3x4x5xi32>)

%depthwise_conv_2d_nhwc_hwcm_a0, %depthwise_conv_2d_nhwc_hwcm_a1, %depthwise_conv_2d_nhwc_hwcm_a2 = "test.op"() : () -> (tensor<2x9x11x5xf32>, tensor<7x8x5x6xf32>, tensor<2x3x4x5x6xf32>)
%depthwise_conv_2d_nhwc_hwcm = "linalg.depthwise_conv_2d_nhwc_hwcm"(%depthwise_conv_2d_nhwc_hwcm_a0, %depthwise_conv_2d_nhwc_hwcm_a1, %depthwise_conv_2d_nhwc_hwcm_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%depthwise_conv_2d_nhwc_hwcm_b0: f32, %depthwise_conv_2d_nhwc_hwcm_b1: f32, %depthwise_conv_2d_nhwc_hwcm_b2: f32):
  "linalg.yield"(%depthwise_conv_2d_nhwc_hwcm_b0) : (f32) -> ()
}) : (tensor<2x9x11x5xf32>, tensor<7x8x5x6xf32>, tensor<2x3x4x5x6xf32>) -> (tensor<2x3x4x5x6xf32>)

%depthwise_conv_2d_nhwc_hwcm_q_a0, %depthwise_conv_2d_nhwc_hwcm_q_a1, %depthwise_conv_2d_nhwc_hwcm_q_a2, %depthwise_conv_2d_nhwc_hwcm_q_a3, %depthwise_conv_2d_nhwc_hwcm_q_a4 = "test.op"() : () -> (tensor<2x9x11x5xi32>, tensor<7x8x5x6xi32>, i32, i32, tensor<2x3x4x5x6xi32>)
%depthwise_conv_2d_nhwc_hwcm_q = "linalg.depthwise_conv_2d_nhwc_hwcm_q"(%depthwise_conv_2d_nhwc_hwcm_q_a0, %depthwise_conv_2d_nhwc_hwcm_q_a1, %depthwise_conv_2d_nhwc_hwcm_q_a2, %depthwise_conv_2d_nhwc_hwcm_q_a3, %depthwise_conv_2d_nhwc_hwcm_q_a4) <{operandSegmentSizes = array<i32: 4, 1>}> ({
^bb0(%depthwise_conv_2d_nhwc_hwcm_q_b0: i32, %depthwise_conv_2d_nhwc_hwcm_q_b1: i32, %depthwise_conv_2d_nhwc_hwcm_q_b2: i32, %depthwise_conv_2d_nhwc_hwcm_q_b3: i32, %depthwise_conv_2d_nhwc_hwcm_q_b4: i32):
  "linalg.yield"(%depthwise_conv_2d_nhwc_hwcm_q_b0) : (i32) -> ()
}) : (tensor<2x9x11x5xi32>, tensor<7x8x5x6xi32>, i32, i32, tensor<2x3x4x5x6xi32>) -> (tensor<2x3x4x5x6xi32>)

%depthwise_conv_3d_ndhwc_dhwc_a0, %depthwise_conv_3d_ndhwc_dhwc_a1, %depthwise_conv_3d_ndhwc_dhwc_a2 = "test.op"() : () -> (tensor<2x8x10x12x9xf32>, tensor<6x7x8x9xf32>, tensor<2x3x4x5x9xf32>)
%depthwise_conv_3d_ndhwc_dhwc = "linalg.depthwise_conv_3d_ndhwc_dhwc"(%depthwise_conv_3d_ndhwc_dhwc_a0, %depthwise_conv_3d_ndhwc_dhwc_a1, %depthwise_conv_3d_ndhwc_dhwc_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%depthwise_conv_3d_ndhwc_dhwc_b0: f32, %depthwise_conv_3d_ndhwc_dhwc_b1: f32, %depthwise_conv_3d_ndhwc_dhwc_b2: f32):
  "linalg.yield"(%depthwise_conv_3d_ndhwc_dhwc_b0) : (f32) -> ()
}) : (tensor<2x8x10x12x9xf32>, tensor<6x7x8x9xf32>, tensor<2x3x4x5x9xf32>) -> (tensor<2x3x4x5x9xf32>)

%depthwise_conv_3d_ncdhw_cdhw_a0, %depthwise_conv_3d_ncdhw_cdhw_a1, %depthwise_conv_3d_ncdhw_cdhw_a2 = "test.op"() : () -> (tensor<2x9x8x10x12xf32>, tensor<9x6x7x8xf32>, tensor<2x9x3x4x5xf32>)
%depthwise_conv_3d_ncdhw_cdhw = "linalg.depthwise_conv_3d_ncdhw_cdhw"(%depthwise_conv_3d_ncdhw_cdhw_a0, %depthwise_conv_3d_ncdhw_cdhw_a1, %depthwise_conv_3d_ncdhw_cdhw_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%depthwise_conv_3d_ncdhw_cdhw_b0: f32, %depthwise_conv_3d_ncdhw_cdhw_b1: f32, %depthwise_conv_3d_ncdhw_cdhw_b2: f32):
  "linalg.yield"(%depthwise_conv_3d_ncdhw_cdhw_b0) : (f32) -> ()
}) : (tensor<2x9x8x10x12xf32>, tensor<9x6x7x8xf32>, tensor<2x9x3x4x5xf32>) -> (tensor<2x9x3x4x5xf32>)

%depthwise_conv_3d_ndhwc_dhwcm_a0, %depthwise_conv_3d_ndhwc_dhwcm_a1, %depthwise_conv_3d_ndhwc_dhwcm_a2 = "test.op"() : () -> (tensor<2x9x11x13x10xf32>, tensor<7x8x9x10x6xf32>, tensor<2x3x4x5x10x6xf32>)
%depthwise_conv_3d_ndhwc_dhwcm = "linalg.depthwise_conv_3d_ndhwc_dhwcm"(%depthwise_conv_3d_ndhwc_dhwcm_a0, %depthwise_conv_3d_ndhwc_dhwcm_a1, %depthwise_conv_3d_ndhwc_dhwcm_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%depthwise_conv_3d_ndhwc_dhwcm_b0: f32, %depthwise_conv_3d_ndhwc_dhwcm_b1: f32, %depthwise_conv_3d_ndhwc_dhwcm_b2: f32):
  "linalg.yield"(%depthwise_conv_3d_ndhwc_dhwcm_b0) : (f32) -> ()
}) : (tensor<2x9x11x13x10xf32>, tensor<7x8x9x10x6xf32>, tensor<2x3x4x5x10x6xf32>) -> (tensor<2x3x4x5x10x6xf32>)

%pooling_nhwc_sum_a0, %pooling_nhwc_sum_a1, %pooling_nhwc_sum_a2 = "test.op"() : () -> (tensor<2x8x10x5xf32>, tensor<6x7xf32>, tensor<2x3x4x5xf32>)
%pooling_nhwc_sum = "linalg.pooling_nhwc_sum"(%pooling_nhwc_sum_a0, %pooling_nhwc_sum_a1, %pooling_nhwc_sum_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%pooling_nhwc_sum_b0: f32, %pooling_nhwc_sum_b1: f32, %pooling_nhwc_sum_b2: f32):
  "linalg.yield"(%pooling_nhwc_sum_b0) : (f32) -> ()
}) : (tensor<2x8x10x5xf32>, tensor<6x7xf32>, tensor<2x3x4x5xf32>) -> (tensor<2x3x4x5xf32>)

%pooling_nchw_sum_a0, %pooling_nchw_sum_a1, %pooling_nchw_sum_a2 = "test.op"() : () -> (tensor<2x3x9x11xf32>, tensor<6x7xf32>, tensor<2x3x4x5xf32>)
%pooling_nchw_sum = "linalg.pooling_nchw_sum"(%pooling_nchw_sum_a0, %pooling_nchw_sum_a1, %pooling_nchw_sum_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%pooling_nchw_sum_b0: f32, %pooling_nchw_sum_b1: f32, %pooling_nchw_sum_b2: f32):
  "linalg.yield"(%pooling_nchw_sum_b0) : (f32) -> ()
}) : (tensor<2x3x9x11xf32>, tensor<6x7xf32>, tensor<2x3x4x5xf32>) -> (tensor<2x3x4x5xf32>)

%pooling_nhwc_max_a0, %pooling_nhwc_max_a1, %pooling_nhwc_max_a2 = "test.op"() : () -> (tensor<2x8x10x5xf32>, tensor<6x7xf32>, tensor<2x3x4x5xf32>)
%pooling_nhwc_max = "linalg.pooling_nhwc_max"(%pooling_nhwc_max_a0, %pooling_nhwc_max_a1, %pooling_nhwc_max_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%pooling_nhwc_max_b0: f32, %pooling_nhwc_max_b1: f32, %pooling_nhwc_max_b2: f32):
  "linalg.yield"(%pooling_nhwc_max_b0) : (f32) -> ()
}) : (tensor<2x8x10x5xf32>, tensor<6x7xf32>, tensor<2x3x4x5xf32>) -> (tensor<2x3x4x5xf32>)

%pooling_nhwc_max_unsigned_a0, %pooling_nhwc_max_unsigned_a1, %pooling_nhwc_max_unsigned_a2 = "test.op"() : () -> (tensor<2x8x10x5xi32>, tensor<6x7xi32>, tensor<2x3x4x5xi32>)
%pooling_nhwc_max_unsigned = "linalg.pooling_nhwc_max_unsigned"(%pooling_nhwc_max_unsigned_a0, %pooling_nhwc_max_unsigned_a1, %pooling_nhwc_max_unsigned_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%pooling_nhwc_max_unsigned_b0: i32, %pooling_nhwc_max_unsigned_b1: i32, %pooling_nhwc_max_unsigned_b2: i32):
  "linalg.yield"(%pooling_nhwc_max_unsigned_b0) : (i32) -> ()
}) : (tensor<2x8x10x5xi32>, tensor<6x7xi32>, tensor<2x3x4x5xi32>) -> (tensor<2x3x4x5xi32>)

%pooling_nchw_max_a0, %pooling_nchw_max_a1, %pooling_nchw_max_a2 = "test.op"() : () -> (tensor<2x3x9x11xf32>, tensor<6x7xf32>, tensor<2x3x4x5xf32>)
%pooling_nchw_max = "linalg.pooling_nchw_max"(%pooling_nchw_max_a0, %pooling_nchw_max_a1, %pooling_nchw_max_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%pooling_nchw_max_b0: f32, %pooling_nchw_max_b1: f32, %pooling_nchw_max_b2: f32):
  "linalg.yield"(%pooling_nchw_max_b0) : (f32) -> ()
}) : (tensor<2x3x9x11xf32>, tensor<6x7xf32>, tensor<2x3x4x5xf32>) -> (tensor<2x3x4x5xf32>)

%pooling_nhwc_min_a0, %pooling_nhwc_min_a1, %pooling_nhwc_min_a2 = "test.op"() : () -> (tensor<2x8x10x5xf32>, tensor<6x7xf32>, tensor<2x3x4x5xf32>)
%pooling_nhwc_min = "linalg.pooling_nhwc_min"(%pooling_nhwc_min_a0, %pooling_nhwc_min_a1, %pooling_nhwc_min_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%pooling_nhwc_min_b0: f32, %pooling_nhwc_min_b1: f32, %pooling_nhwc_min_b2: f32):
  "linalg.yield"(%pooling_nhwc_min_b0) : (f32) -> ()
}) : (tensor<2x8x10x5xf32>, tensor<6x7xf32>, tensor<2x3x4x5xf32>) -> (tensor<2x3x4x5xf32>)

%pooling_nhwc_min_unsigned_a0, %pooling_nhwc_min_unsigned_a1, %pooling_nhwc_min_unsigned_a2 = "test.op"() : () -> (tensor<2x8x10x5xi32>, tensor<6x7xi32>, tensor<2x3x4x5xi32>)
%pooling_nhwc_min_unsigned = "linalg.pooling_nhwc_min_unsigned"(%pooling_nhwc_min_unsigned_a0, %pooling_nhwc_min_unsigned_a1, %pooling_nhwc_min_unsigned_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%pooling_nhwc_min_unsigned_b0: i32, %pooling_nhwc_min_unsigned_b1: i32, %pooling_nhwc_min_unsigned_b2: i32):
  "linalg.yield"(%pooling_nhwc_min_unsigned_b0) : (i32) -> ()
}) : (tensor<2x8x10x5xi32>, tensor<6x7xi32>, tensor<2x3x4x5xi32>) -> (tensor<2x3x4x5xi32>)

%pooling_nwc_sum_a0, %pooling_nwc_sum_a1, %pooling_nwc_sum_a2 = "test.op"() : () -> (tensor<2x7x4xf32>, tensor<5xf32>, tensor<2x3x4xf32>)
%pooling_nwc_sum = "linalg.pooling_nwc_sum"(%pooling_nwc_sum_a0, %pooling_nwc_sum_a1, %pooling_nwc_sum_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%pooling_nwc_sum_b0: f32, %pooling_nwc_sum_b1: f32, %pooling_nwc_sum_b2: f32):
  "linalg.yield"(%pooling_nwc_sum_b0) : (f32) -> ()
}) : (tensor<2x7x4xf32>, tensor<5xf32>, tensor<2x3x4xf32>) -> (tensor<2x3x4xf32>)

%pooling_ncw_sum_a0, %pooling_ncw_sum_a1, %pooling_ncw_sum_a2 = "test.op"() : () -> (tensor<2x3x8xf32>, tensor<5xf32>, tensor<2x3x4xf32>)
%pooling_ncw_sum = "linalg.pooling_ncw_sum"(%pooling_ncw_sum_a0, %pooling_ncw_sum_a1, %pooling_ncw_sum_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%pooling_ncw_sum_b0: f32, %pooling_ncw_sum_b1: f32, %pooling_ncw_sum_b2: f32):
  "linalg.yield"(%pooling_ncw_sum_b0) : (f32) -> ()
}) : (tensor<2x3x8xf32>, tensor<5xf32>, tensor<2x3x4xf32>) -> (tensor<2x3x4xf32>)

%pooling_nwc_max_a0, %pooling_nwc_max_a1, %pooling_nwc_max_a2 = "test.op"() : () -> (tensor<2x7x4xf32>, tensor<5xf32>, tensor<2x3x4xf32>)
%pooling_nwc_max = "linalg.pooling_nwc_max"(%pooling_nwc_max_a0, %pooling_nwc_max_a1, %pooling_nwc_max_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%pooling_nwc_max_b0: f32, %pooling_nwc_max_b1: f32, %pooling_nwc_max_b2: f32):
  "linalg.yield"(%pooling_nwc_max_b0) : (f32) -> ()
}) : (tensor<2x7x4xf32>, tensor<5xf32>, tensor<2x3x4xf32>) -> (tensor<2x3x4xf32>)

%pooling_nwc_max_unsigned_a0, %pooling_nwc_max_unsigned_a1, %pooling_nwc_max_unsigned_a2 = "test.op"() : () -> (tensor<2x7x4xi32>, tensor<5xi32>, tensor<2x3x4xi32>)
%pooling_nwc_max_unsigned = "linalg.pooling_nwc_max_unsigned"(%pooling_nwc_max_unsigned_a0, %pooling_nwc_max_unsigned_a1, %pooling_nwc_max_unsigned_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%pooling_nwc_max_unsigned_b0: i32, %pooling_nwc_max_unsigned_b1: i32, %pooling_nwc_max_unsigned_b2: i32):
  "linalg.yield"(%pooling_nwc_max_unsigned_b0) : (i32) -> ()
}) : (tensor<2x7x4xi32>, tensor<5xi32>, tensor<2x3x4xi32>) -> (tensor<2x3x4xi32>)

%pooling_ncw_max_a0, %pooling_ncw_max_a1, %pooling_ncw_max_a2 = "test.op"() : () -> (tensor<2x3x8xf32>, tensor<5xf32>, tensor<2x3x4xf32>)
%pooling_ncw_max = "linalg.pooling_ncw_max"(%pooling_ncw_max_a0, %pooling_ncw_max_a1, %pooling_ncw_max_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%pooling_ncw_max_b0: f32, %pooling_ncw_max_b1: f32, %pooling_ncw_max_b2: f32):
  "linalg.yield"(%pooling_ncw_max_b0) : (f32) -> ()
}) : (tensor<2x3x8xf32>, tensor<5xf32>, tensor<2x3x4xf32>) -> (tensor<2x3x4xf32>)

%pooling_nwc_min_a0, %pooling_nwc_min_a1, %pooling_nwc_min_a2 = "test.op"() : () -> (tensor<2x7x4xf32>, tensor<5xf32>, tensor<2x3x4xf32>)
%pooling_nwc_min = "linalg.pooling_nwc_min"(%pooling_nwc_min_a0, %pooling_nwc_min_a1, %pooling_nwc_min_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%pooling_nwc_min_b0: f32, %pooling_nwc_min_b1: f32, %pooling_nwc_min_b2: f32):
  "linalg.yield"(%pooling_nwc_min_b0) : (f32) -> ()
}) : (tensor<2x7x4xf32>, tensor<5xf32>, tensor<2x3x4xf32>) -> (tensor<2x3x4xf32>)

%pooling_nwc_min_unsigned_a0, %pooling_nwc_min_unsigned_a1, %pooling_nwc_min_unsigned_a2 = "test.op"() : () -> (tensor<2x7x4xi32>, tensor<5xi32>, tensor<2x3x4xi32>)
%pooling_nwc_min_unsigned = "linalg.pooling_nwc_min_unsigned"(%pooling_nwc_min_unsigned_a0, %pooling_nwc_min_unsigned_a1, %pooling_nwc_min_unsigned_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%pooling_nwc_min_unsigned_b0: i32, %pooling_nwc_min_unsigned_b1: i32, %pooling_nwc_min_unsigned_b2: i32):
  "linalg.yield"(%pooling_nwc_min_unsigned_b0) : (i32) -> ()
}) : (tensor<2x7x4xi32>, tensor<5xi32>, tensor<2x3x4xi32>) -> (tensor<2x3x4xi32>)

%pooling_ndhwc_sum_a0, %pooling_ndhwc_sum_a1, %pooling_ndhwc_sum_a2 = "test.op"() : () -> (tensor<2x9x11x13x6xf32>, tensor<7x8x9xf32>, tensor<2x3x4x5x6xf32>)
%pooling_ndhwc_sum = "linalg.pooling_ndhwc_sum"(%pooling_ndhwc_sum_a0, %pooling_ndhwc_sum_a1, %pooling_ndhwc_sum_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%pooling_ndhwc_sum_b0: f32, %pooling_ndhwc_sum_b1: f32, %pooling_ndhwc_sum_b2: f32):
  "linalg.yield"(%pooling_ndhwc_sum_b0) : (f32) -> ()
}) : (tensor<2x9x11x13x6xf32>, tensor<7x8x9xf32>, tensor<2x3x4x5x6xf32>) -> (tensor<2x3x4x5x6xf32>)

%pooling_ndhwc_max_a0, %pooling_ndhwc_max_a1, %pooling_ndhwc_max_a2 = "test.op"() : () -> (tensor<2x9x11x13x6xf32>, tensor<7x8x9xf32>, tensor<2x3x4x5x6xf32>)
%pooling_ndhwc_max = "linalg.pooling_ndhwc_max"(%pooling_ndhwc_max_a0, %pooling_ndhwc_max_a1, %pooling_ndhwc_max_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%pooling_ndhwc_max_b0: f32, %pooling_ndhwc_max_b1: f32, %pooling_ndhwc_max_b2: f32):
  "linalg.yield"(%pooling_ndhwc_max_b0) : (f32) -> ()
}) : (tensor<2x9x11x13x6xf32>, tensor<7x8x9xf32>, tensor<2x3x4x5x6xf32>) -> (tensor<2x3x4x5x6xf32>)

%pooling_ndhwc_min_a0, %pooling_ndhwc_min_a1, %pooling_ndhwc_min_a2 = "test.op"() : () -> (tensor<2x9x11x13x6xf32>, tensor<7x8x9xf32>, tensor<2x3x4x5x6xf32>)
%pooling_ndhwc_min = "linalg.pooling_ndhwc_min"(%pooling_ndhwc_min_a0, %pooling_ndhwc_min_a1, %pooling_ndhwc_min_a2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%pooling_ndhwc_min_b0: f32, %pooling_ndhwc_min_b1: f32, %pooling_ndhwc_min_b2: f32):
  "linalg.yield"(%pooling_ndhwc_min_b0) : (f32) -> ()
}) : (tensor<2x9x11x13x6xf32>, tensor<7x8x9xf32>, tensor<2x3x4x5x6xf32>) -> (tensor<2x3x4x5x6xf32>)

%fill_a0, %fill_a1 = "test.op"() : () -> (f32, tensor<4x8xf32>)
%fill = "linalg.fill"(%fill_a0, %fill_a1) <{operandSegmentSizes = array<i32: 1, 1>}> ({
^bb0(%fill_b0: f32, %fill_b1: f32):
  "linalg.yield"(%fill_b0) : (f32) -> ()
}) : (f32, tensor<4x8xf32>) -> (tensor<4x8xf32>)

%fill_rng_2d_a0, %fill_rng_2d_a1, %fill_rng_2d_a2, %fill_rng_2d_a3 = "test.op"() : () -> (f32, f32, f32, tensor<2x3xf32>)
%fill_rng_2d = "linalg.fill_rng_2d"(%fill_rng_2d_a0, %fill_rng_2d_a1, %fill_rng_2d_a2, %fill_rng_2d_a3) <{operandSegmentSizes = array<i32: 3, 1>}> ({
^bb0(%fill_rng_2d_b0: f32, %fill_rng_2d_b1: f32, %fill_rng_2d_b2: f32, %fill_rng_2d_b3: f32):
  "linalg.yield"(%fill_rng_2d_b0) : (f32) -> ()
}) : (f32, f32, f32, tensor<2x3xf32>) -> (tensor<2x3xf32>)

// CHECK:      builtin.module {
// CHECK-NEXT:   %0, %1 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK-NEXT:   %2 = "linalg.copy"(%0, %1) <{operandSegmentSizes = array<i32: 1, 1>, cast = #linalg.type_fn<cast_unsigned>}> ({
// CHECK-NEXT:   ^bb0(%3: f32, %4: f32):
// CHECK-NEXT:     "linalg.yield"(%3) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %3, %4 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK-NEXT:   %5 = "linalg.exp"(%3, %4) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-NEXT:   ^bb0(%6: f32, %7: f32):
// CHECK-NEXT:     "linalg.yield"(%6) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %6, %7 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK-NEXT:   %8 = "linalg.log"(%6, %7) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-NEXT:   ^bb0(%9: f32, %10: f32):
// CHECK-NEXT:     "linalg.yield"(%9) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %9, %10 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK-NEXT:   %11 = "linalg.abs"(%9, %10) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-NEXT:   ^bb0(%12: f32, %13: f32):
// CHECK-NEXT:     "linalg.yield"(%12) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %12, %13 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK-NEXT:   %14 = "linalg.ceil"(%12, %13) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-NEXT:   ^bb0(%15: f32, %16: f32):
// CHECK-NEXT:     "linalg.yield"(%15) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %15, %16 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK-NEXT:   %17 = "linalg.floor"(%15, %16) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-NEXT:   ^bb0(%18: f32, %19: f32):
// CHECK-NEXT:     "linalg.yield"(%18) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %18, %19 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK-NEXT:   %20 = "linalg.negf"(%18, %19) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-NEXT:   ^bb0(%21: f32, %22: f32):
// CHECK-NEXT:     "linalg.yield"(%21) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %21, %22 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK-NEXT:   %23 = "linalg.reciprocal"(%21, %22) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-NEXT:   ^bb0(%24: f32, %25: f32):
// CHECK-NEXT:     "linalg.yield"(%24) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %24, %25 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK-NEXT:   %26 = "linalg.round"(%24, %25) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-NEXT:   ^bb0(%27: f32, %28: f32):
// CHECK-NEXT:     "linalg.yield"(%27) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %27, %28 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK-NEXT:   %29 = "linalg.sqrt"(%27, %28) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-NEXT:   ^bb0(%30: f32, %31: f32):
// CHECK-NEXT:     "linalg.yield"(%30) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %30, %31 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK-NEXT:   %32 = "linalg.rsqrt"(%30, %31) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-NEXT:   ^bb0(%33: f32, %34: f32):
// CHECK-NEXT:     "linalg.yield"(%33) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %33, %34 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK-NEXT:   %35 = "linalg.square"(%33, %34) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-NEXT:   ^bb0(%36: f32, %37: f32):
// CHECK-NEXT:     "linalg.yield"(%36) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %36, %37 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK-NEXT:   %38 = "linalg.tanh"(%36, %37) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-NEXT:   ^bb0(%39: f32, %40: f32):
// CHECK-NEXT:     "linalg.yield"(%39) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %39, %40 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK-NEXT:   %41 = "linalg.erf"(%39, %40) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-NEXT:   ^bb0(%42: f32, %43: f32):
// CHECK-NEXT:     "linalg.yield"(%42) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %42, %43, %44 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK-NEXT:   %45 = "linalg.add"(%42, %43, %44) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%46: f32, %47: f32, %48: f32):
// CHECK-NEXT:     "linalg.yield"(%46) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %46, %47, %48 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK-NEXT:   %49 = "linalg.sub"(%46, %47, %48) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%50: f32, %51: f32, %52: f32):
// CHECK-NEXT:     "linalg.yield"(%50) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %50, %51, %52 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK-NEXT:   %53 = "linalg.mul"(%50, %51, %52) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%54: f32, %55: f32, %56: f32):
// CHECK-NEXT:     "linalg.yield"(%54) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %54, %55, %56 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK-NEXT:   %57 = "linalg.div"(%54, %55, %56) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%58: f32, %59: f32, %60: f32):
// CHECK-NEXT:     "linalg.yield"(%58) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %58, %59, %60 = "test.op"() : () -> (tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>)
// CHECK-NEXT:   %61 = "linalg.div_unsigned"(%58, %59, %60) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%62: i32, %63: i32, %64: i32):
// CHECK-NEXT:     "linalg.yield"(%62) : (i32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi32>
// CHECK-NEXT:   %62, %63, %64 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK-NEXT:   %65 = "linalg.max"(%62, %63, %64) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%66: f32, %67: f32, %68: f32):
// CHECK-NEXT:     "linalg.yield"(%66) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %66, %67, %68 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK-NEXT:   %69 = "linalg.min"(%66, %67, %68) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%70: f32, %71: f32, %72: f32):
// CHECK-NEXT:     "linalg.yield"(%70) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %70, %71, %72 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK-NEXT:   %73 = "linalg.powf"(%70, %71, %72) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%74: f32, %75: f32, %76: f32):
// CHECK-NEXT:     "linalg.yield"(%74) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %74, %75, %76, %77 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK-NEXT:   %78 = "linalg.select"(%74, %75, %76, %77) <{operandSegmentSizes = array<i32: 3, 1>}> ({
// CHECK-NEXT:   ^bb0(%79: f32, %80: f32, %81: f32, %82: f32):
// CHECK-NEXT:     "linalg.yield"(%79) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %79, %80, %81, %82, %83 = "test.op"() : () -> (tensor<2x4xi32>, tensor<4x3xi32>, i32, i32, tensor<2x3xi32>)
// CHECK-NEXT:   %84 = "linalg.quantized_matmul"(%79, %80, %81, %82, %83) <{operandSegmentSizes = array<i32: 4, 1>}> ({
// CHECK-NEXT:   ^bb0(%85: i32, %86: i32, %87: i32, %88: i32, %89: i32):
// CHECK-NEXT:     "linalg.yield"(%85) : (i32) -> ()
// CHECK-NEXT:   }) : (tensor<2x4xi32>, tensor<4x3xi32>, i32, i32, tensor<2x3xi32>) -> tensor<2x3xi32>
// CHECK-NEXT:   %85, %86, %87 = "test.op"() : () -> (tensor<2x4x5x7xf32>, tensor<3x4x6x7xf32>, tensor<2x3x5x6xf32>)
// CHECK-NEXT:   %88 = "linalg.mmt4d"(%85, %86, %87) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%89: f32, %90: f32, %91: f32):
// CHECK-NEXT:     %92 = "arith.mulf"(%89, %90) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:     %93 = "arith.addf"(%91, %92) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:     "linalg.yield"(%93) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x4x5x7xf32>, tensor<3x4x6x7xf32>, tensor<2x3x5x6xf32>) -> tensor<2x3x5x6xf32>
// CHECK-NEXT:   %89, %90, %91 = "test.op"() : () -> (tensor<2x3x5x6x8xf32>, tensor<2x4x5x7x8xf32>, tensor<2x3x4x6x7xf32>)
// CHECK-NEXT:   %92 = "linalg.batch_mmt4d"(%89, %90, %91) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%93: f32, %94: f32, %95: f32):
// CHECK-NEXT:     %96 = "arith.mulf"(%93, %94) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:     %97 = "arith.addf"(%95, %96) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:     "linalg.yield"(%97) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x3x5x6x8xf32>, tensor<2x4x5x7x8xf32>, tensor<2x3x4x6x7xf32>) -> tensor<2x3x4x6x7xf32>
// CHECK-NEXT:   %93, %94, %95, %96, %97 = "test.op"() : () -> (tensor<2x3x5xi32>, tensor<2x5x4xi32>, i32, i32, tensor<2x3x4xi32>)
// CHECK-NEXT:   %98 = "linalg.quantized_batch_matmul"(%93, %94, %95, %96, %97) <{operandSegmentSizes = array<i32: 4, 1>}> ({
// CHECK-NEXT:   ^bb0(%99: i32, %100: i32, %101: i32, %102: i32, %103: i32):
// CHECK-NEXT:     "linalg.yield"(%99) : (i32) -> ()
// CHECK-NEXT:   }) : (tensor<2x3x5xi32>, tensor<2x5x4xi32>, i32, i32, tensor<2x3x4xi32>) -> tensor<2x3x4xi32>
// CHECK-NEXT:   %99, %100, %101 = "test.op"() : () -> (tensor<2x3xf32>, tensor<3xf32>, tensor<2xf32>)
// CHECK-NEXT:   %102 = "linalg.matvec"(%99, %100, %101) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%103: f32, %104: f32, %105: f32):
// CHECK-NEXT:     %106 = "arith.mulf"(%103, %104) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:     %107 = "arith.addf"(%105, %106) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:     "linalg.yield"(%107) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x3xf32>, tensor<3xf32>, tensor<2xf32>) -> tensor<2xf32>
// CHECK-NEXT:   %103, %104, %105 = "test.op"() : () -> (tensor<3xf32>, tensor<3x2xf32>, tensor<2xf32>)
// CHECK-NEXT:   %106 = "linalg.vecmat"(%103, %104, %105) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%107: f32, %108: f32, %109: f32):
// CHECK-NEXT:     %110 = "arith.mulf"(%107, %108) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:     %111 = "arith.addf"(%109, %110) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:     "linalg.yield"(%111) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<3xf32>, tensor<3x2xf32>, tensor<2xf32>) -> tensor<2xf32>
// CHECK-NEXT:   %107, %108, %109 = "test.op"() : () -> (tensor<2x3x4xf32>, tensor<2x4xf32>, tensor<2x3xf32>)
// CHECK-NEXT:   %110 = "linalg.batch_matvec"(%107, %108, %109) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%111: f32, %112: f32, %113: f32):
// CHECK-NEXT:     %114 = "arith.mulf"(%111, %112) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:     %115 = "arith.addf"(%113, %114) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:     "linalg.yield"(%115) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x3x4xf32>, tensor<2x4xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:   %111, %112, %113 = "test.op"() : () -> (tensor<2x4xf32>, tensor<2x4x3xf32>, tensor<2x3xf32>)
// CHECK-NEXT:   %114 = "linalg.batch_vecmat"(%111, %112, %113) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%115: f32, %116: f32, %117: f32):
// CHECK-NEXT:     %118 = "arith.mulf"(%115, %116) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:     %119 = "arith.addf"(%117, %118) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:     "linalg.yield"(%119) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x4xf32>, tensor<2x4x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:   %115, %116, %117 = "test.op"() : () -> (tensor<2xf32>, tensor<2xf32>, tensor<f32>)
// CHECK-NEXT:   %118 = "linalg.dot"(%115, %116, %117) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%119: f32, %120: f32, %121: f32):
// CHECK-NEXT:     %122 = "arith.mulf"(%119, %120) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:     %123 = "arith.addf"(%121, %122) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:     "linalg.yield"(%123) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2xf32>, tensor<2xf32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:   %119, %120, %121 = "test.op"() : () -> (tensor<4xf32>, tensor<3xf32>, tensor<2xf32>)
// CHECK-NEXT:   %122 = "linalg.conv_1d"(%119, %120, %121) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%123: f32, %124: f32, %125: f32):
// CHECK-NEXT:     "linalg.yield"(%123) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4xf32>, tensor<3xf32>, tensor<2xf32>) -> tensor<2xf32>
// CHECK-NEXT:   %123, %124, %125 = "test.op"() : () -> (tensor<5x7xf32>, tensor<4x5xf32>, tensor<2x3xf32>)
// CHECK-NEXT:   %126 = "linalg.conv_2d"(%123, %124, %125) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%127: f32, %128: f32, %129: f32):
// CHECK-NEXT:     "linalg.yield"(%127) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<5x7xf32>, tensor<4x5xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:   %127, %128, %129 = "test.op"() : () -> (tensor<6x8x10xf32>, tensor<5x6x7xf32>, tensor<2x3x4xf32>)
// CHECK-NEXT:   %130 = "linalg.conv_3d"(%127, %128, %129) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%131: f32, %132: f32, %133: f32):
// CHECK-NEXT:     "linalg.yield"(%131) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<6x8x10xf32>, tensor<5x6x7xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
// CHECK-NEXT:   %131, %132, %133 = "test.op"() : () -> (tensor<2x7x6xf32>, tensor<5x6x4xf32>, tensor<2x3x4xf32>)
// CHECK-NEXT:   %134 = "linalg.conv_1d_nwc_wcf"(%131, %132, %133) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%135: f32, %136: f32, %137: f32):
// CHECK-NEXT:     "linalg.yield"(%135) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x7x6xf32>, tensor<5x6x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
// CHECK-NEXT:   %135, %136, %137 = "test.op"() : () -> (tensor<2x5x9xf32>, tensor<3x5x6xf32>, tensor<2x3x4xf32>)
// CHECK-NEXT:   %138 = "linalg.conv_1d_ncw_fcw"(%135, %136, %137) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%139: f32, %140: f32, %141: f32):
// CHECK-NEXT:     "linalg.yield"(%139) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x5x9xf32>, tensor<3x5x6xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
// CHECK-NEXT:   %139, %140, %141 = "test.op"() : () -> (tensor<2x8x10x8xf32>, tensor<6x7x8x5xf32>, tensor<2x3x4x5xf32>)
// CHECK-NEXT:   %142 = "linalg.conv_2d_nhwc_hwcf"(%139, %140, %141) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%143: f32, %144: f32, %145: f32):
// CHECK-NEXT:     "linalg.yield"(%143) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x8x10x8xf32>, tensor<6x7x8x5xf32>, tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
// CHECK-NEXT:   %143, %144, %145 = "test.op"() : () -> (tensor<2x8x10x8xf32>, tensor<5x6x7x8xf32>, tensor<2x3x4x5xf32>)
// CHECK-NEXT:   %146 = "linalg.conv_2d_nhwc_fhwc"(%143, %144, %145) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%147: f32, %148: f32, %149: f32):
// CHECK-NEXT:     "linalg.yield"(%147) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x8x10x8xf32>, tensor<5x6x7x8xf32>, tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
// CHECK-NEXT:   %147, %148, %149, %150, %151 = "test.op"() : () -> (tensor<2x8x10x8xi32>, tensor<6x7x8x5xi32>, i32, i32, tensor<2x3x4x5xi32>)
// CHECK-NEXT:   %152 = "linalg.conv_2d_nhwc_hwcf_q"(%147, %148, %149, %150, %151) <{operandSegmentSizes = array<i32: 4, 1>}> ({
// CHECK-NEXT:   ^bb0(%153: i32, %154: i32, %155: i32, %156: i32, %157: i32):
// CHECK-NEXT:     "linalg.yield"(%153) : (i32) -> ()
// CHECK-NEXT:   }) : (tensor<2x8x10x8xi32>, tensor<6x7x8x5xi32>, i32, i32, tensor<2x3x4x5xi32>) -> tensor<2x3x4x5xi32>
// CHECK-NEXT:   %153, %154, %155, %156, %157 = "test.op"() : () -> (tensor<2x8x10x8xi32>, tensor<5x6x7x8xi32>, i32, i32, tensor<2x3x4x5xi32>)
// CHECK-NEXT:   %158 = "linalg.conv_2d_nhwc_fhwc_q"(%153, %154, %155, %156, %157) <{operandSegmentSizes = array<i32: 4, 1>}> ({
// CHECK-NEXT:   ^bb0(%159: i32, %160: i32, %161: i32, %162: i32, %163: i32):
// CHECK-NEXT:     "linalg.yield"(%159) : (i32) -> ()
// CHECK-NEXT:   }) : (tensor<2x8x10x8xi32>, tensor<5x6x7x8xi32>, i32, i32, tensor<2x3x4x5xi32>) -> tensor<2x3x4x5xi32>
// CHECK-NEXT:   %159, %160, %161, %162, %163 = "test.op"() : () -> (tensor<2x6x10x12xi32>, tensor<3x6x7x8xi32>, i32, i32, tensor<2x3x4x5xi32>)
// CHECK-NEXT:   %164 = "linalg.conv_2d_nchw_fchw_q"(%159, %160, %161, %162, %163) <{operandSegmentSizes = array<i32: 4, 1>}> ({
// CHECK-NEXT:   ^bb0(%165: i32, %166: i32, %167: i32, %168: i32, %169: i32):
// CHECK-NEXT:     "linalg.yield"(%165) : (i32) -> ()
// CHECK-NEXT:   }) : (tensor<2x6x10x12xi32>, tensor<3x6x7x8xi32>, i32, i32, tensor<2x3x4x5xi32>) -> tensor<2x3x4x5xi32>
// CHECK-NEXT:   %165, %166, %167 = "test.op"() : () -> (tensor<2x6x10x12xf32>, tensor<3x6x7x8xf32>, tensor<2x3x4x5xf32>)
// CHECK-NEXT:   %168 = "linalg.conv_2d_nchw_fchw"(%165, %166, %167) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%169: f32, %170: f32, %171: f32):
// CHECK-NEXT:     "linalg.yield"(%169) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x6x10x12xf32>, tensor<3x6x7x8xf32>, tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
// CHECK-NEXT:   %169, %170, %171 = "test.op"() : () -> (tensor<2x3x7x12x14xf32>, tensor<4x3x7x8x9xf32>, tensor<2x3x4x5x6xf32>)
// CHECK-NEXT:   %172 = "linalg.conv_2d_ngchw_fgchw"(%169, %170, %171) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%173: f32, %174: f32, %175: f32):
// CHECK-NEXT:     "linalg.yield"(%173) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x3x7x12x14xf32>, tensor<4x3x7x8x9xf32>, tensor<2x3x4x5x6xf32>) -> tensor<2x3x4x5x6xf32>
// CHECK-NEXT:   %173, %174, %175 = "test.op"() : () -> (tensor<2x3x7x12x14xf32>, tensor<3x4x7x8x9xf32>, tensor<2x3x4x5x6xf32>)
// CHECK-NEXT:   %176 = "linalg.conv_2d_ngchw_gfchw"(%173, %174, %175) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%177: f32, %178: f32, %179: f32):
// CHECK-NEXT:     "linalg.yield"(%177) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x3x7x12x14xf32>, tensor<3x4x7x8x9xf32>, tensor<2x3x4x5x6xf32>) -> tensor<2x3x4x5x6xf32>
// CHECK-NEXT:   %177, %178, %179 = "test.op"() : () -> (tensor<2x9x11x5x9xf32>, tensor<5x6x7x8x9xf32>, tensor<2x3x4x5x6xf32>)
// CHECK-NEXT:   %180 = "linalg.conv_2d_nhwgc_gfhwc"(%177, %178, %179) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%181: f32, %182: f32, %183: f32):
// CHECK-NEXT:     "linalg.yield"(%181) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x9x11x5x9xf32>, tensor<5x6x7x8x9xf32>, tensor<2x3x4x5x6xf32>) -> tensor<2x3x4x5x6xf32>
// CHECK-NEXT:   %181, %182, %183, %184, %185 = "test.op"() : () -> (tensor<2x9x11x5x9xi32>, tensor<5x6x7x8x9xi32>, i32, i32, tensor<2x3x4x5x6xi32>)
// CHECK-NEXT:   %186 = "linalg.conv_2d_nhwgc_gfhwc_q"(%181, %182, %183, %184, %185) <{operandSegmentSizes = array<i32: 4, 1>}> ({
// CHECK-NEXT:   ^bb0(%187: i32, %188: i32, %189: i32, %190: i32, %191: i32):
// CHECK-NEXT:     "linalg.yield"(%187) : (i32) -> ()
// CHECK-NEXT:   }) : (tensor<2x9x11x5x9xi32>, tensor<5x6x7x8x9xi32>, i32, i32, tensor<2x3x4x5x6xi32>) -> tensor<2x3x4x5x6xi32>
// CHECK-NEXT:   %187, %188, %189, %190, %191 = "test.op"() : () -> (tensor<2x3x7x12x14xi32>, tensor<3x4x7x8x9xi32>, i32, i32, tensor<2x3x4x5x6xi32>)
// CHECK-NEXT:   %192 = "linalg.conv_2d_ngchw_gfchw_q"(%187, %188, %189, %190, %191) <{operandSegmentSizes = array<i32: 4, 1>}> ({
// CHECK-NEXT:   ^bb0(%193: i32, %194: i32, %195: i32, %196: i32, %197: i32):
// CHECK-NEXT:     "linalg.yield"(%193) : (i32) -> ()
// CHECK-NEXT:   }) : (tensor<2x3x7x12x14xi32>, tensor<3x4x7x8x9xi32>, i32, i32, tensor<2x3x4x5x6xi32>) -> tensor<2x3x4x5x6xi32>
// CHECK-NEXT:   %193, %194, %195 = "test.op"() : () -> (tensor<2x9x11x13x10xf32>, tensor<7x8x9x10x6xf32>, tensor<2x3x4x5x6xf32>)
// CHECK-NEXT:   %196 = "linalg.conv_3d_ndhwc_dhwcf"(%193, %194, %195) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%197: f32, %198: f32, %199: f32):
// CHECK-NEXT:     "linalg.yield"(%197) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x9x11x13x10xf32>, tensor<7x8x9x10x6xf32>, tensor<2x3x4x5x6xf32>) -> tensor<2x3x4x5x6xf32>
// CHECK-NEXT:   %197, %198, %199, %200, %201 = "test.op"() : () -> (tensor<2x9x11x13x10xi32>, tensor<7x8x9x10x6xi32>, i32, i32, tensor<2x3x4x5x6xi32>)
// CHECK-NEXT:   %202 = "linalg.conv_3d_ndhwc_dhwcf_q"(%197, %198, %199, %200, %201) <{operandSegmentSizes = array<i32: 4, 1>}> ({
// CHECK-NEXT:   ^bb0(%203: i32, %204: i32, %205: i32, %206: i32, %207: i32):
// CHECK-NEXT:     "linalg.yield"(%203) : (i32) -> ()
// CHECK-NEXT:   }) : (tensor<2x9x11x13x10xi32>, tensor<7x8x9x10x6xi32>, i32, i32, tensor<2x3x4x5x6xi32>) -> tensor<2x3x4x5x6xi32>
// CHECK-NEXT:   %203, %204, %205 = "test.op"() : () -> (tensor<2x7x11x13x15xf32>, tensor<3x7x8x9x10xf32>, tensor<2x3x4x5x6xf32>)
// CHECK-NEXT:   %206 = "linalg.conv_3d_ncdhw_fcdhw"(%203, %204, %205) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%207: f32, %208: f32, %209: f32):
// CHECK-NEXT:     "linalg.yield"(%207) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x7x11x13x15xf32>, tensor<3x7x8x9x10xf32>, tensor<2x3x4x5x6xf32>) -> tensor<2x3x4x5x6xf32>
// CHECK-NEXT:   %207, %208, %209 = "test.op"() : () -> (tensor<2x7x4xf32>, tensor<5x4xf32>, tensor<2x3x4xf32>)
// CHECK-NEXT:   %210 = "linalg.depthwise_conv_1d_nwc_wc"(%207, %208, %209) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%211: f32, %212: f32, %213: f32):
// CHECK-NEXT:     "linalg.yield"(%211) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x7x4xf32>, tensor<5x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
// CHECK-NEXT:   %211, %212, %213 = "test.op"() : () -> (tensor<2x4x7xf32>, tensor<4x5xf32>, tensor<2x4x3xf32>)
// CHECK-NEXT:   %214 = "linalg.depthwise_conv_1d_ncw_cw"(%211, %212, %213) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%215: f32, %216: f32, %217: f32):
// CHECK-NEXT:     "linalg.yield"(%215) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x4x7xf32>, tensor<4x5xf32>, tensor<2x4x3xf32>) -> tensor<2x4x3xf32>
// CHECK-NEXT:   %215, %216, %217 = "test.op"() : () -> (tensor<2x8x4xf32>, tensor<6x4x5xf32>, tensor<2x3x4x5xf32>)
// CHECK-NEXT:   %218 = "linalg.depthwise_conv_1d_nwc_wcm"(%215, %216, %217) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%219: f32, %220: f32, %221: f32):
// CHECK-NEXT:     "linalg.yield"(%219) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x8x4xf32>, tensor<6x4x5xf32>, tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
// CHECK-NEXT:   %219, %220, %221 = "test.op"() : () -> (tensor<2x8x10x5xf32>, tensor<6x7x5xf32>, tensor<2x3x4x5xf32>)
// CHECK-NEXT:   %222 = "linalg.depthwise_conv_2d_nhwc_hwc"(%219, %220, %221) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%223: f32, %224: f32, %225: f32):
// CHECK-NEXT:     "linalg.yield"(%223) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x8x10x5xf32>, tensor<6x7x5xf32>, tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
// CHECK-NEXT:   %223, %224, %225 = "test.op"() : () -> (tensor<2x5x8x10xf32>, tensor<5x6x7xf32>, tensor<2x5x3x4xf32>)
// CHECK-NEXT:   %226 = "linalg.depthwise_conv_2d_nchw_chw"(%223, %224, %225) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%227: f32, %228: f32, %229: f32):
// CHECK-NEXT:     "linalg.yield"(%227) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x5x8x10xf32>, tensor<5x6x7xf32>, tensor<2x5x3x4xf32>) -> tensor<2x5x3x4xf32>
// CHECK-NEXT:   %227, %228, %229, %230, %231 = "test.op"() : () -> (tensor<2x8x10x5xi32>, tensor<6x7x5xi32>, i32, i32, tensor<2x3x4x5xi32>)
// CHECK-NEXT:   %232 = "linalg.depthwise_conv_2d_nhwc_hwc_q"(%227, %228, %229, %230, %231) <{operandSegmentSizes = array<i32: 4, 1>}> ({
// CHECK-NEXT:   ^bb0(%233: i32, %234: i32, %235: i32, %236: i32, %237: i32):
// CHECK-NEXT:     "linalg.yield"(%233) : (i32) -> ()
// CHECK-NEXT:   }) : (tensor<2x8x10x5xi32>, tensor<6x7x5xi32>, i32, i32, tensor<2x3x4x5xi32>) -> tensor<2x3x4x5xi32>
// CHECK-NEXT:   %233, %234, %235 = "test.op"() : () -> (tensor<2x9x11x5xf32>, tensor<7x8x5x6xf32>, tensor<2x3x4x5x6xf32>)
// CHECK-NEXT:   %236 = "linalg.depthwise_conv_2d_nhwc_hwcm"(%233, %234, %235) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%237: f32, %238: f32, %239: f32):
// CHECK-NEXT:     "linalg.yield"(%237) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x9x11x5xf32>, tensor<7x8x5x6xf32>, tensor<2x3x4x5x6xf32>) -> tensor<2x3x4x5x6xf32>
// CHECK-NEXT:   %237, %238, %239, %240, %241 = "test.op"() : () -> (tensor<2x9x11x5xi32>, tensor<7x8x5x6xi32>, i32, i32, tensor<2x3x4x5x6xi32>)
// CHECK-NEXT:   %242 = "linalg.depthwise_conv_2d_nhwc_hwcm_q"(%237, %238, %239, %240, %241) <{operandSegmentSizes = array<i32: 4, 1>}> ({
// CHECK-NEXT:   ^bb0(%243: i32, %244: i32, %245: i32, %246: i32, %247: i32):
// CHECK-NEXT:     "linalg.yield"(%243) : (i32) -> ()
// CHECK-NEXT:   }) : (tensor<2x9x11x5xi32>, tensor<7x8x5x6xi32>, i32, i32, tensor<2x3x4x5x6xi32>) -> tensor<2x3x4x5x6xi32>
// CHECK-NEXT:   %243, %244, %245 = "test.op"() : () -> (tensor<2x8x10x12x9xf32>, tensor<6x7x8x9xf32>, tensor<2x3x4x5x9xf32>)
// CHECK-NEXT:   %246 = "linalg.depthwise_conv_3d_ndhwc_dhwc"(%243, %244, %245) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%247: f32, %248: f32, %249: f32):
// CHECK-NEXT:     "linalg.yield"(%247) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x8x10x12x9xf32>, tensor<6x7x8x9xf32>, tensor<2x3x4x5x9xf32>) -> tensor<2x3x4x5x9xf32>
// CHECK-NEXT:   %247, %248, %249 = "test.op"() : () -> (tensor<2x9x8x10x12xf32>, tensor<9x6x7x8xf32>, tensor<2x9x3x4x5xf32>)
// CHECK-NEXT:   %250 = "linalg.depthwise_conv_3d_ncdhw_cdhw"(%247, %248, %249) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%251: f32, %252: f32, %253: f32):
// CHECK-NEXT:     "linalg.yield"(%251) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x9x8x10x12xf32>, tensor<9x6x7x8xf32>, tensor<2x9x3x4x5xf32>) -> tensor<2x9x3x4x5xf32>
// CHECK-NEXT:   %251, %252, %253 = "test.op"() : () -> (tensor<2x9x11x13x10xf32>, tensor<7x8x9x10x6xf32>, tensor<2x3x4x5x10x6xf32>)
// CHECK-NEXT:   %254 = "linalg.depthwise_conv_3d_ndhwc_dhwcm"(%251, %252, %253) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%255: f32, %256: f32, %257: f32):
// CHECK-NEXT:     "linalg.yield"(%255) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x9x11x13x10xf32>, tensor<7x8x9x10x6xf32>, tensor<2x3x4x5x10x6xf32>) -> tensor<2x3x4x5x10x6xf32>
// CHECK-NEXT:   %255, %256, %257 = "test.op"() : () -> (tensor<2x8x10x5xf32>, tensor<6x7xf32>, tensor<2x3x4x5xf32>)
// CHECK-NEXT:   %258 = "linalg.pooling_nhwc_sum"(%255, %256, %257) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%259: f32, %260: f32, %261: f32):
// CHECK-NEXT:     "linalg.yield"(%259) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x8x10x5xf32>, tensor<6x7xf32>, tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
// CHECK-NEXT:   %259, %260, %261 = "test.op"() : () -> (tensor<2x3x9x11xf32>, tensor<6x7xf32>, tensor<2x3x4x5xf32>)
// CHECK-NEXT:   %262 = "linalg.pooling_nchw_sum"(%259, %260, %261) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%263: f32, %264: f32, %265: f32):
// CHECK-NEXT:     "linalg.yield"(%263) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x3x9x11xf32>, tensor<6x7xf32>, tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
// CHECK-NEXT:   %263, %264, %265 = "test.op"() : () -> (tensor<2x8x10x5xf32>, tensor<6x7xf32>, tensor<2x3x4x5xf32>)
// CHECK-NEXT:   %266 = "linalg.pooling_nhwc_max"(%263, %264, %265) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%267: f32, %268: f32, %269: f32):
// CHECK-NEXT:     "linalg.yield"(%267) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x8x10x5xf32>, tensor<6x7xf32>, tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
// CHECK-NEXT:   %267, %268, %269 = "test.op"() : () -> (tensor<2x8x10x5xi32>, tensor<6x7xi32>, tensor<2x3x4x5xi32>)
// CHECK-NEXT:   %270 = "linalg.pooling_nhwc_max_unsigned"(%267, %268, %269) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%271: i32, %272: i32, %273: i32):
// CHECK-NEXT:     "linalg.yield"(%271) : (i32) -> ()
// CHECK-NEXT:   }) : (tensor<2x8x10x5xi32>, tensor<6x7xi32>, tensor<2x3x4x5xi32>) -> tensor<2x3x4x5xi32>
// CHECK-NEXT:   %271, %272, %273 = "test.op"() : () -> (tensor<2x3x9x11xf32>, tensor<6x7xf32>, tensor<2x3x4x5xf32>)
// CHECK-NEXT:   %274 = "linalg.pooling_nchw_max"(%271, %272, %273) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%275: f32, %276: f32, %277: f32):
// CHECK-NEXT:     "linalg.yield"(%275) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x3x9x11xf32>, tensor<6x7xf32>, tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
// CHECK-NEXT:   %275, %276, %277 = "test.op"() : () -> (tensor<2x8x10x5xf32>, tensor<6x7xf32>, tensor<2x3x4x5xf32>)
// CHECK-NEXT:   %278 = "linalg.pooling_nhwc_min"(%275, %276, %277) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%279: f32, %280: f32, %281: f32):
// CHECK-NEXT:     "linalg.yield"(%279) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x8x10x5xf32>, tensor<6x7xf32>, tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
// CHECK-NEXT:   %279, %280, %281 = "test.op"() : () -> (tensor<2x8x10x5xi32>, tensor<6x7xi32>, tensor<2x3x4x5xi32>)
// CHECK-NEXT:   %282 = "linalg.pooling_nhwc_min_unsigned"(%279, %280, %281) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%283: i32, %284: i32, %285: i32):
// CHECK-NEXT:     "linalg.yield"(%283) : (i32) -> ()
// CHECK-NEXT:   }) : (tensor<2x8x10x5xi32>, tensor<6x7xi32>, tensor<2x3x4x5xi32>) -> tensor<2x3x4x5xi32>
// CHECK-NEXT:   %283, %284, %285 = "test.op"() : () -> (tensor<2x7x4xf32>, tensor<5xf32>, tensor<2x3x4xf32>)
// CHECK-NEXT:   %286 = "linalg.pooling_nwc_sum"(%283, %284, %285) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%287: f32, %288: f32, %289: f32):
// CHECK-NEXT:     "linalg.yield"(%287) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x7x4xf32>, tensor<5xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
// CHECK-NEXT:   %287, %288, %289 = "test.op"() : () -> (tensor<2x3x8xf32>, tensor<5xf32>, tensor<2x3x4xf32>)
// CHECK-NEXT:   %290 = "linalg.pooling_ncw_sum"(%287, %288, %289) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%291: f32, %292: f32, %293: f32):
// CHECK-NEXT:     "linalg.yield"(%291) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x3x8xf32>, tensor<5xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
// CHECK-NEXT:   %291, %292, %293 = "test.op"() : () -> (tensor<2x7x4xf32>, tensor<5xf32>, tensor<2x3x4xf32>)
// CHECK-NEXT:   %294 = "linalg.pooling_nwc_max"(%291, %292, %293) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%295: f32, %296: f32, %297: f32):
// CHECK-NEXT:     "linalg.yield"(%295) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x7x4xf32>, tensor<5xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
// CHECK-NEXT:   %295, %296, %297 = "test.op"() : () -> (tensor<2x7x4xi32>, tensor<5xi32>, tensor<2x3x4xi32>)
// CHECK-NEXT:   %298 = "linalg.pooling_nwc_max_unsigned"(%295, %296, %297) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%299: i32, %300: i32, %301: i32):
// CHECK-NEXT:     "linalg.yield"(%299) : (i32) -> ()
// CHECK-NEXT:   }) : (tensor<2x7x4xi32>, tensor<5xi32>, tensor<2x3x4xi32>) -> tensor<2x3x4xi32>
// CHECK-NEXT:   %299, %300, %301 = "test.op"() : () -> (tensor<2x3x8xf32>, tensor<5xf32>, tensor<2x3x4xf32>)
// CHECK-NEXT:   %302 = "linalg.pooling_ncw_max"(%299, %300, %301) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%303: f32, %304: f32, %305: f32):
// CHECK-NEXT:     "linalg.yield"(%303) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x3x8xf32>, tensor<5xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
// CHECK-NEXT:   %303, %304, %305 = "test.op"() : () -> (tensor<2x7x4xf32>, tensor<5xf32>, tensor<2x3x4xf32>)
// CHECK-NEXT:   %306 = "linalg.pooling_nwc_min"(%303, %304, %305) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%307: f32, %308: f32, %309: f32):
// CHECK-NEXT:     "linalg.yield"(%307) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x7x4xf32>, tensor<5xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
// CHECK-NEXT:   %307, %308, %309 = "test.op"() : () -> (tensor<2x7x4xi32>, tensor<5xi32>, tensor<2x3x4xi32>)
// CHECK-NEXT:   %310 = "linalg.pooling_nwc_min_unsigned"(%307, %308, %309) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%311: i32, %312: i32, %313: i32):
// CHECK-NEXT:     "linalg.yield"(%311) : (i32) -> ()
// CHECK-NEXT:   }) : (tensor<2x7x4xi32>, tensor<5xi32>, tensor<2x3x4xi32>) -> tensor<2x3x4xi32>
// CHECK-NEXT:   %311, %312, %313 = "test.op"() : () -> (tensor<2x9x11x13x6xf32>, tensor<7x8x9xf32>, tensor<2x3x4x5x6xf32>)
// CHECK-NEXT:   %314 = "linalg.pooling_ndhwc_sum"(%311, %312, %313) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%315: f32, %316: f32, %317: f32):
// CHECK-NEXT:     "linalg.yield"(%315) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x9x11x13x6xf32>, tensor<7x8x9xf32>, tensor<2x3x4x5x6xf32>) -> tensor<2x3x4x5x6xf32>
// CHECK-NEXT:   %315, %316, %317 = "test.op"() : () -> (tensor<2x9x11x13x6xf32>, tensor<7x8x9xf32>, tensor<2x3x4x5x6xf32>)
// CHECK-NEXT:   %318 = "linalg.pooling_ndhwc_max"(%315, %316, %317) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%319: f32, %320: f32, %321: f32):
// CHECK-NEXT:     "linalg.yield"(%319) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x9x11x13x6xf32>, tensor<7x8x9xf32>, tensor<2x3x4x5x6xf32>) -> tensor<2x3x4x5x6xf32>
// CHECK-NEXT:   %319, %320, %321 = "test.op"() : () -> (tensor<2x9x11x13x6xf32>, tensor<7x8x9xf32>, tensor<2x3x4x5x6xf32>)
// CHECK-NEXT:   %322 = "linalg.pooling_ndhwc_min"(%319, %320, %321) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%323: f32, %324: f32, %325: f32):
// CHECK-NEXT:     "linalg.yield"(%323) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x9x11x13x6xf32>, tensor<7x8x9xf32>, tensor<2x3x4x5x6xf32>) -> tensor<2x3x4x5x6xf32>
// CHECK-NEXT:   %323, %324 = "test.op"() : () -> (f32, tensor<4x8xf32>)
// CHECK-NEXT:   %325 = "linalg.fill"(%323, %324) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-NEXT:   ^bb0(%326: f32, %327: f32):
// CHECK-NEXT:     "linalg.yield"(%326) : (f32) -> ()
// CHECK-NEXT:   }) : (f32, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %326, %327, %328, %329 = "test.op"() : () -> (f32, f32, f32, tensor<2x3xf32>)
// CHECK-NEXT:   %330 = "linalg.fill_rng_2d"(%326, %327, %328, %329) <{operandSegmentSizes = array<i32: 3, 1>}> ({
// CHECK-NEXT:   ^bb0(%331: f32, %332: f32, %333: f32, %334: f32):
// CHECK-NEXT:     "linalg.yield"(%331) : (f32) -> ()
// CHECK-NEXT:   }) : (f32, f32, f32, tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT: }
