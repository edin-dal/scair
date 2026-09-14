package scair.dialects.linalg

import scair.enums.I32Enum
import scair.ir.EnumAttr

// ██╗░░░░░ ██╗ ███╗░░██╗ ░█████╗░ ██╗░░░░░ ░██████╗░
// ██║░░░░░ ██║ ████╗░██║ ██╔══██╗ ██║░░░░░ ██╔════╝░
// ██║░░░░░ ██║ ██╔██╗██║ ███████║ ██║░░░░░ ██║░░██╗░
// ██║░░░░░ ██║ ██║╚████║ ██╔══██║ ██║░░░░░ ██║░░╚██╗
// ███████╗ ██║ ██║░╚███║ ██║░░██║ ███████╗ ╚██████╔╝
// ╚══════╝ ╚═╝ ╚═╝░░╚══╝ ╚═╝░░╚═╝ ╚══════╝ ░╚═════╝░
//
// ███████╗ ███╗░░██╗ ██╗░░░██╗ ███╗░░░███╗ ░██████╗
// ██╔════╝ ████╗░██║ ██║░░░██║ ████╗░████║ ██╔════╝
// █████╗░░ ██╔██╗██║ ██║░░░██║ ██╔████╔██║ ╚█████╗░
// ██╔══╝░░ ██║╚████║ ██║░░░██║ ██║╚██╔╝██║ ░╚═══██╗
// ███████╗ ██║░╚███║ ╚██████╔╝ ██║░╚═╝░██║ ██████╔╝
// ╚══════╝ ╚═╝░░╚══╝ ░╚═════╝░ ╚═╝░░░░░╚═╝ ╚═════╝░

// enum Shade(caseName: String) extends EnumAttr("enum.shade", caseName):
//   case Pale extends Shade("pale")
//   case Vivid extends Shade("vivid")
//   case Dark extends Shade("dark")

/*≡==--==≡≡≡≡≡==--=≡≡*\
||     UNARY FN      ||
\*≡==---==≡≡≡==---==≡*/

/** Function attribute enum matching the OpDSL unary functions. Ported from
  * MLIR's `UnaryFn` in `LinalgEnums.td`.
  */
enum UnaryFn(name: String) extends EnumAttr("linalg.unary_fn", name):
  case exp extends UnaryFn("exp")
  case log extends UnaryFn("log")
  case abs extends UnaryFn("abs")
  case ceil extends UnaryFn("ceil")
  case floor extends UnaryFn("floor")
  case negf extends UnaryFn("negf")
  case reciprocal extends UnaryFn("reciprocal")
  case round extends UnaryFn("round")
  case sqrt extends UnaryFn("sqrt")
  case rsqrt extends UnaryFn("rsqrt")
  case square extends UnaryFn("square")
  case tanh extends UnaryFn("tanh")
  case erf extends UnaryFn("erf")

/*≡==--==≡≡≡≡≡==--=≡≡*\
||     BINARY FN     ||
\*≡==---==≡≡≡==---==≡*/

/** Function attribute enum matching the OpDSL binary functions. Ported from
  * MLIR's `BinaryFn` in `LinalgEnums.td`.
  */
enum BinaryFn(name: String) extends EnumAttr("linalg.binary_fn", name):
  case add extends BinaryFn("add")
  case sub extends BinaryFn("sub")
  case mul extends BinaryFn("mul")
  case div extends BinaryFn("div")
  case div_unsigned extends BinaryFn("div_unsigned")
  case max_signed extends BinaryFn("max_signed")
  case min_signed extends BinaryFn("min_signed")
  case max_unsigned extends BinaryFn("max_unsigned")
  case min_unsigned extends BinaryFn("min_unsigned")
  case powf extends BinaryFn("powf")

/*≡==--==≡≡≡≡≡==--=≡≡*\
||    TERNARY FN     ||
\*≡==---==≡≡≡==---==≡*/

/** Function attribute enum matching the OpDSL ternary functions. Ported from
  * MLIR's `TernaryFn` in `LinalgEnums.td`.
  */
enum TernaryFn(name: String) extends EnumAttr("linalg.ternary_fn", name):
  case select extends TernaryFn("select")

/*≡==--==≡≡≡≡≡≡≡==--=≡≡*\
||   ELEMENTWISE KIND  ||
\*≡==---==≡≡≡≡≡==---==≡*/

/** Unified enum for all element-wise op functions. Upstream this is built by
  * concatenating the `UnaryFn`, `BinaryFn` and `TernaryFn` case lists, with the
  * offsets chosen so that the numeric values do not overlap; the concatenation
  * is spelled out here since ScaIR has no equivalent of TableGen's `!foldl`.
  */
enum ElementwiseKind(name: String)
    extends EnumAttr("linalg.elementwise_kind", name):
  // UnaryFn cases, offset 0
  case exp extends ElementwiseKind("exp")
  case log extends ElementwiseKind("log")
  case abs extends ElementwiseKind("abs")
  case ceil extends ElementwiseKind("ceil")
  case floor extends ElementwiseKind("floor")
  case negf extends ElementwiseKind("negf")
  case reciprocal extends ElementwiseKind("reciprocal")
  case round extends ElementwiseKind("round")
  case sqrt extends ElementwiseKind("sqrt")
  case rsqrt extends ElementwiseKind("rsqrt")
  case square extends ElementwiseKind("square")
  case tanh extends ElementwiseKind("tanh")
  case erf extends ElementwiseKind("erf")
  // BinaryFn cases, offset 13
  case add extends ElementwiseKind("add")
  case sub extends ElementwiseKind("sub")
  case mul extends ElementwiseKind("mul")
  case div extends ElementwiseKind("div")
  case div_unsigned extends ElementwiseKind("div_unsigned")
  case max_signed extends ElementwiseKind("max_signed")
  case min_signed extends ElementwiseKind("min_signed")
  case max_unsigned extends ElementwiseKind("max_unsigned")
  case min_unsigned extends ElementwiseKind("min_unsigned")
  case powf extends ElementwiseKind("powf")
  // TernaryFn cases, offset 23
  case select extends ElementwiseKind("select")

/*≡==--==≡≡≡≡≡==--=≡≡*\
||      TYPE FN      ||
\*≡==---==≡≡≡==---==≡*/

/** Function attribute enum matching the OpDSL type conversion functions. Ported
  * from MLIR's `TypeFn` in `LinalgEnums.td`.
  */
enum TypeFn(name: String) extends EnumAttr("linalg.type_fn", name):
  case cast_signed extends TypeFn("cast_signed")
  case cast_unsigned extends TypeFn("cast_unsigned")

/*≡==--==≡≡≡≡≡≡≡==--=≡≡*\
||  WINOGRAD CONV FMR  ||
\*≡==---==≡≡≡≡≡==---==≡*/

/** `F(m, r)` sizing of the minimal filtering algorithm used by the Winograd
  * convolution ops: `m` is the output dimension, `r` the filter dimension, and
  * the input dimension follows as `alpha = m + r - 1`.
  */
enum WinogradConv2DFmr(name: String) extends I32Enum(name):
  case F_2_3 extends WinogradConv2DFmr("F_2_3")
  case F_4_3 extends WinogradConv2DFmr("F_4_3")
  case F_2_5 extends WinogradConv2DFmr("F_2_5")

/*≡==--==≡≡≡≡≡≡≡==--=≡≡*\
||    ITERATOR TYPE    ||
\*≡==---==≡≡≡≡≡==---==≡*/

/** Iterator kind of a structured op's loop nest. Defined upstream in
  * `mlir/Dialect/Utils/StructuredOpsUtils.td` rather than in the Linalg dialect
  * itself, but only ever surfaced through `linalg.generic`'s `iterator_types`,
  * so it is ported here.
  */
enum IteratorType(name: String) extends EnumAttr("linalg.iterator_type", name):
  case parallel extends IteratorType("parallel")
  case reduction extends IteratorType("reduction")
  case window extends IteratorType("window")
