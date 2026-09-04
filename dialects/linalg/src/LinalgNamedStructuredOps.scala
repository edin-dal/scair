package scair.dialects.linalg

import scair.clair.*
import scair.dialects.builtin.*
import scair.dialects.builtin.TensorLiteralArray
import scair.ir.*

// ███╗░░██╗ ░█████╗░ ███╗░░░███╗ ███████╗ ██████╗░
// ████╗░██║ ██╔══██╗ ████╗░████║ ██╔════╝ ██╔══██╗
// ██╔██╗██║ ███████║ ██╔████╔██║ █████╗░░ ██║░░██║
// ██║╚████║ ██╔══██║ ██║╚██╔╝██║ ██╔══╝░░ ██║░░██║
// ██║░╚███║ ██║░░██║ ██║░╚═╝░██║ ███████╗ ██████╔╝
// ╚═╝░░╚══╝ ╚═╝░░╚═╝ ╚═╝░░░░░╚═╝ ╚══════╝ ╚═════╝░
//
// ░██████╗ ████████╗ ██████╗░ ██╗░░░██╗ ░█████╗░ ████████╗ ██╗░░░██╗ ██████╗░ ███████╗ ██████╗░
// ██╔════╝ ╚══██╔══╝ ██╔══██╗ ██║░░░██║ ██╔══██╗ ╚══██╔══╝ ██║░░░██║ ██╔══██╗ ██╔════╝ ██╔══██╗
// ╚█████╗░ ░░░██║░░░ ██████╔╝ ██║░░░██║ ██║░░╚═╝ ░░░██║░░░ ██║░░░██║ ██████╔╝ █████╗░░ ██║░░██║
// ░╚═══██╗ ░░░██║░░░ ██╔══██╗ ██║░░░██║ ██║░░██╗ ░░░██║░░░ ██║░░░██║ ██╔══██╗ ██╔══╝░░ ██║░░██║
// ██████╔╝ ░░░██║░░░ ██║░░██║ ╚██████╔╝ ╚█████╔╝ ░░░██║░░░ ╚██████╔╝ ██║░░██║ ███████╗ ██████╔╝
// ╚═════╝░ ░░░╚═╝░░░ ╚═╝░░╚═╝ ░╚═════╝░ ░╚════╝░ ░░░╚═╝░░░ ░╚═════╝░ ╚═╝░░╚═╝ ╚══════╝ ╚═════╝░

/* Ports the named structured ops that MLIR generates from
 * LinalgNamedStructuredOps.yaml (via mlir-linalg-ods-yaml-gen).
 *
 * Every one of them lowers to the same ODS shape: two variadic operand groups
 * (`inputs` carries the input tensors *and* any scalar operands, `outputs`
 * carries the output tensors), variadic tensor results, a region holding the
 * scalar body, and `AttrSizedOperandSegments`. The only per-op variation is the
 * set of attributes: the YAML `index_attr` args become `strides`/`dilations`,
 * and `copy`'s `type_fn_attr` arg becomes `cast`.
 *
 * The YAML's indexing maps, iterator types and scalar assignments describe the
 * op's *semantics*, not its IR shape, so they are deliberately not ported here.
 *
 * Note that none of these ops carries LinalgContractionOpInterface,
 * LinalgConvolutionOpInterface or LinalgFillOpInterface, even where the name
 * suggests one: the YAML records no interface information, so the ODS generator
 * gives every op the same trait list, and MLIR recognises contractions,
 * convolutions and fills structurally instead.
 *
 * Upstream `strides` and `dilations` are `I64ElementsAttr`s, printed as
 * `dense<1> : tensor<2xi64>`; they are `DenseIntOrFPElementsAttr` here. Both are
 * optional with a default of all-ones, which ScaIR cannot express, so an absent
 * attribute carries that default implicitly. */

def denseI64ElementsAttr(dims: Int, data: Int*) = DenseIntOrFPElementsAttr(
  typ = RankedTensorType(
    IntegerType(IntData(64), Signless),
    ArrayAttribute(IntData(dims)),
  ),
  data = ArrayAttribute(data.map(IntData(_))*).asInstanceOf[TensorLiteralArray],
)

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||  ELEMENTWISE UNARY / COPY  ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

case class Copy(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    cast: Option[TypeFn] = Some(TypeFn.cast_signed),
    region: Region,
) extends DerivedOperation["linalg.copy"]
    with LinalgStructuredBase derives OpDefs

case class Exp(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.exp"]
    with LinalgStructuredBase derives OpDefs

case class Log(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.log"]
    with LinalgStructuredBase derives OpDefs

case class Abs(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.abs"]
    with LinalgStructuredBase derives OpDefs

case class Ceil(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.ceil"]
    with LinalgStructuredBase derives OpDefs

case class Floor(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.floor"]
    with LinalgStructuredBase derives OpDefs

case class NegF(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.negf"]
    with LinalgStructuredBase derives OpDefs

case class Reciprocal(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.reciprocal"]
    with LinalgStructuredBase derives OpDefs

case class Round(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.round"]
    with LinalgStructuredBase derives OpDefs

case class Sqrt(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.sqrt"]
    with LinalgStructuredBase derives OpDefs

case class Rsqrt(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.rsqrt"]
    with LinalgStructuredBase derives OpDefs

case class Square(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.square"]
    with LinalgStructuredBase derives OpDefs

case class Tanh(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.tanh"]
    with LinalgStructuredBase derives OpDefs

case class Erf(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.erf"]
    with LinalgStructuredBase derives OpDefs

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||  ELEMENTWISE BINARY  ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

case class Add(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.add"]
    with LinalgStructuredBase derives OpDefs

case class Sub(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.sub"]
    with LinalgStructuredBase derives OpDefs

case class Mul(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.mul"]
    with LinalgStructuredBase derives OpDefs

case class Div(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.div"]
    with LinalgStructuredBase derives OpDefs

case class DivUnsigned(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.div_unsigned"]
    with LinalgStructuredBase derives OpDefs

case class Max(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.max"]
    with LinalgStructuredBase derives OpDefs

case class Min(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.min"]
    with LinalgStructuredBase derives OpDefs

case class PowF(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.powf"]
    with LinalgStructuredBase derives OpDefs

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||  ELEMENTWISE TERNARY  ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

case class Select(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.select"]
    with LinalgStructuredBase derives OpDefs

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||  MATMUL AND FRIENDS  ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

case class QuantizedMatmul(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.quantized_matmul"]
    with LinalgStructuredBase derives OpDefs

case class Mmt4D(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.mmt4d"]
    with LinalgStructuredBase
    with LinalgContractionOpInterface derives OpDefs

case class BatchMmt4D(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.batch_mmt4d"]
    with LinalgStructuredBase
    with LinalgContractionOpInterface derives OpDefs

case class QuantizedBatchMatmul(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.quantized_batch_matmul"]
    with LinalgStructuredBase derives OpDefs

case class Matvec(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.matvec"]
    with LinalgStructuredBase
    with LinalgContractionOpInterface derives OpDefs

case class Vecmat(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.vecmat"]
    with LinalgStructuredBase
    with LinalgContractionOpInterface derives OpDefs

case class BatchMatvec(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.batch_matvec"]
    with LinalgStructuredBase
    with LinalgContractionOpInterface derives OpDefs

case class BatchVecmat(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.batch_vecmat"]
    with LinalgStructuredBase
    with LinalgContractionOpInterface derives OpDefs

case class Dot(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.dot"]
    with LinalgStructuredBase
    with LinalgContractionOpInterface derives OpDefs

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||  CONVOLUTIONS  ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

case class Conv1D(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.conv_1d"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class Conv2D(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.conv_2d"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class Conv3D(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.conv_3d"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class Conv1DNwcWcf(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(1, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(1, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.conv_1d_nwc_wcf"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class Conv1DNcwFcw(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(1, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(1, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.conv_1d_ncw_fcw"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class Conv2DNhwcHwcf(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.conv_2d_nhwc_hwcf"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class Conv2DNhwcFhwc(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.conv_2d_nhwc_fhwc"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class Conv2DNhwcHwcfQ(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.conv_2d_nhwc_hwcf_q"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class Conv2DNhwcFhwcQ(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.conv_2d_nhwc_fhwc_q"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class Conv2DNchwFchwQ(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.conv_2d_nchw_fchw_q"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class Conv2DNchwFchw(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.conv_2d_nchw_fchw"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class Conv2DNgchwFgchw(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.conv_2d_ngchw_fgchw"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class Conv2DNgchwGfchw(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.conv_2d_ngchw_gfchw"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class Conv2DNhwgcGfhwc(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.conv_2d_nhwgc_gfhwc"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class Conv2DNhwgcGfhwcQ(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.conv_2d_nhwgc_gfhwc_q"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class Conv2DNgchwGfchwQ(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.conv_2d_ngchw_gfchw_q"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class Conv3DNdhwcDhwcf(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(3, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(3, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.conv_3d_ndhwc_dhwcf"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class Conv3DNdhwcDhwcfQ(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(3, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(3, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.conv_3d_ndhwc_dhwcf_q"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class Conv3DNcdhwFcdhw(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(3, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(3, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.conv_3d_ncdhw_fcdhw"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||  DEPTHWISE CONVOLUTIONS  ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

case class DepthwiseConv1DNwcWc(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(1, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(1, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.depthwise_conv_1d_nwc_wc"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class DepthwiseConv1DNcwCw(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(1, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(1, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.depthwise_conv_1d_ncw_cw"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class DepthwiseConv1DNwcWcm(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(1, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(1, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.depthwise_conv_1d_nwc_wcm"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class DepthwiseConv2DNhwcHwc(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.depthwise_conv_2d_nhwc_hwc"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class DepthwiseConv2DNchwChw(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.depthwise_conv_2d_nchw_chw"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class DepthwiseConv2DNhwcHwcQ(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.depthwise_conv_2d_nhwc_hwc_q"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class DepthwiseConv2DNhwcHwcm(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.depthwise_conv_2d_nhwc_hwcm"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class DepthwiseConv2DNhwcHwcmQ(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.depthwise_conv_2d_nhwc_hwcm_q"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class DepthwiseConv3DNdhwcDhwc(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(3, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(3, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.depthwise_conv_3d_ndhwc_dhwc"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class DepthwiseConv3DNcdhwCdhw(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(3, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(3, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.depthwise_conv_3d_ncdhw_cdhw"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class DepthwiseConv3DNdhwcDhwcm(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(3, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(3, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.depthwise_conv_3d_ndhwc_dhwcm"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||  POOLING  ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

case class PoolingNhwcSum(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.pooling_nhwc_sum"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class PoolingNchwSum(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.pooling_nchw_sum"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class PoolingNhwcMax(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.pooling_nhwc_max"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class PoolingNhwcMaxUnsigned(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.pooling_nhwc_max_unsigned"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class PoolingNchwMax(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.pooling_nchw_max"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class PoolingNhwcMin(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.pooling_nhwc_min"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class PoolingNhwcMinUnsigned(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(2, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.pooling_nhwc_min_unsigned"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class PoolingNwcSum(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(1, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(1, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.pooling_nwc_sum"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class PoolingNcwSum(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(1, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(1, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.pooling_ncw_sum"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class PoolingNwcMax(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(1, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(1, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.pooling_nwc_max"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class PoolingNwcMaxUnsigned(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(1, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(1, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.pooling_nwc_max_unsigned"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class PoolingNcwMax(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(1, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(1, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.pooling_ncw_max"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class PoolingNwcMin(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(1, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(1, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.pooling_nwc_min"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class PoolingNwcMinUnsigned(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(1, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(1, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.pooling_nwc_min_unsigned"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class PoolingNdhwcSum(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(3, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(3, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.pooling_ndhwc_sum"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class PoolingNdhwcMax(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(3, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(3, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.pooling_ndhwc_max"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

case class PoolingNdhwcMin(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    strides: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(3, 1)
    ),
    dilations: Option[DenseIntOrFPElementsAttr] = Some(
      denseI64ElementsAttr(3, 1)
    ),
    region: Region,
) extends DerivedOperation["linalg.pooling_ndhwc_min"]
    with LinalgStructuredBase
    with LinalgConvolutionOpInterface derives OpDefs

/*≡==--==≡≡≡≡≡≡≡≡==--=≡≡*\
||  FILL  ||
\*≡==---==≡≡≡≡≡≡≡≡==---==≡*/

case class Fill(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.fill"]
    with LinalgStructuredBase
    with LinalgFillOpInterface derives OpDefs

case class FillRng2D(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.fill_rng_2d"]
    with LinalgStructuredBase derives OpDefs
