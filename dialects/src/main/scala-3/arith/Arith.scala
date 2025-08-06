package scair.dialects.arith

import fastparse.*
import scair.AttrParser
import scair.clair.codegen.*
import scair.clair.macros.*
import scair.dialects.builtin.*
import scair.ir.*

import scala.collection.immutable.*
import org.w3c.dom.Attr

// TODO: Upstream Arith natively support vector or other containers of it's operands and results type
// i.e., add vectors not just integers.
// Let's keep it progressive here though.

// TODO, think about providing bunch of helpers for this kind of attribute
enum FastMathFlag:
//   case none
  case reassoc
  case nnan
  case ninf
  case nsz
  case arcp
  case contract
  case afn
  // case fast

type FastMathFlags = HashSet[FastMathFlag]

object FastMathFlags:
  inline def none: FastMathFlags = HashSet.empty
  inline def reassoc: FastMathFlags = HashSet(FastMathFlag.reassoc)
  inline def nnan: FastMathFlags = HashSet(FastMathFlag.nnan)
  inline def ninf: FastMathFlags = HashSet(FastMathFlag.ninf)
  inline def nsz: FastMathFlags = HashSet(FastMathFlag.nsz)
  inline def arcp: FastMathFlags = HashSet(FastMathFlag.arcp)
  inline def contract: FastMathFlags = HashSet(FastMathFlag.contract)
  inline def afn: FastMathFlags = HashSet(FastMathFlag.afn)

  inline def fast: FastMathFlags =
    reassoc | nnan | ninf | nsz | arcp | contract | afn

  def apply(flags: FastMathFlag*): FastMathFlags = HashSet(flags*)

object FastMathFlagsAttr extends AttributeCompanion {
  override def name: String = "arith.fastmath"

  override def parse[$: P](p: AttrParser): P[FastMathFlagsAttr] = {

    import scair.Parser.whitespace
    P(
      "<" ~ ("none" | "reassoc" | "nnan" | "ninf" | "nsz" | "arcp" | "contract" | "afn" | "fast").!.rep(
        sep = ","
      ) ~ ">"
    ).map { parsed_flags =>
      if parsed_flags.isEmpty then
        throw new Exception("FastMathFlagsAttr requires at least one flag")
      val flags = parsed_flags
        .map(_ match {
          case "none"     => FastMathFlags.none
          case "reassoc"  => FastMathFlags.reassoc
          case "nnan"     => FastMathFlags.nnan
          case "ninf"     => FastMathFlags.ninf
          case "nsz"      => FastMathFlags.nsz
          case "arcp"     => FastMathFlags.arcp
          case "contract" => FastMathFlags.contract
          case "afn"      => FastMathFlags.afn
          case "fast"     => FastMathFlags.fast
          case f          => throw new Exception(s"Invalid fastmath flag '$f'")
        })
        .reduce(_ | _)
      FastMathFlagsAttr(flags)
    }

  }

}

case class FastMathFlagsAttr(val flags: FastMathFlags)
    extends scair.ir.DataAttribute[FastMathFlags]("arith.fastmath", flags):

  override def custom_print: String =
    val p =
      if flags == FastMathFlags.fast then "fast"
      else if flags == FastMathFlags.none then "none"
      else
        this.flags
          .map {
            case FastMathFlag.reassoc  => "reassoc"
            case FastMathFlag.nnan     => "nnan"
            case FastMathFlag.ninf     => "ninf"
            case FastMathFlag.nsz      => "nsz"
            case FastMathFlag.arcp     => "arcp"
            case FastMathFlag.contract => "contract"
            case FastMathFlag.afn      => "afn"
          }
          .mkString(",")
    s"#arith.fastmath<$p>"

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||  TYPES AND CONSTRAINTS  ||
\*≡==---==≡≡≡≡≡≡≡≡≡==---==≡*/

// TODO: this should constrain the IntegerType to be signless, i.e., i32, i8, etc.
// also, should be called SignlessIntegerOrIndexLike
type SignlessIntegerOrIndexLike = IntegerType | IndexType
// TODO: This should constrain to specifically i1, MLIR's take at a boolean type.
// Not yet supported in the frontend.
type I1 = IntegerType
// TODO: This should constrain to specifically i64.
// Not yet supported in the frontend.
type I64 = IntegerType
// TODO: This should be encapsulated in a specific kind of enum attribute.
// This specific kind (not all enums...) are implemented as i64 attributes
// and only used differently in operation-level custom syntaxes (MLIR legacy, no fancy reason)
// So it's just as fine as long as doing generic syntax goes.
// TODO: Should be integerattr[i64] or smth but frontend doesn't support it yet.
type IntegerPredicate = IntegerAttr
// TODO: again, this should be constrained to be i1 specifically,
//       or vector or tensor of I1 (ie, bools)
type BoolLike = I1 | VectorType | TensorType
// TODO: same old same old
type SignlessIntegerOrFloatLike = IntegerType | FloatType
// TODO: this should be constrained to be a signless integer, or vector or tensor of a signless integer
type SignlessFixedWidthIntegerLike = IntegerType | VectorType | TensorType
// TODO: MemrefType here should be looked at more closely to how it is actually implemented in MLIR
type BitcastType = SignlessIntegerOrFloatLike | MemrefType

// TODO: this is specifically a memref type of anysignless integer or index
type AnySignlessInteger = IntegerType

type IndexCastTypeConstraint = SignlessIntegerOrIndexLike |
  MemRefOf[(AnySignlessInteger, IndexType)]

type BitcastTypeConstraint = SignlessIntegerOrIndexLike |
  MemRefOf[(AnySignlessInteger, FloatType)]

type TypedAttrInterface = Attribute
type IntegerOverflowAttr = Attribute
type RoundingModeAttr = Attribute
type MemRefOf[Tuple] = MemrefType
type CmpIPredicateAttr = Attribute

/*≡=--==≡≡≡≡==--=≡*\
||   INTERFACES   ||
\*≡=---==≡≡==---=≡*/

trait SameOperandsAndResultTypes extends Operation {

  override def trait_verify(): Either[String, Operation] = {
    val params = this.operands.typ ++ this.results.typ
    if (params.isEmpty) Right(this)
    else {
      val first = params.head
      if (params.tail.forall(_ == first)) Right(this)
      else
        Left(
          "All parameters of TypeConstraint must be of the same type in operation " + this.name
        )
    }
  }

}

trait SameOperandsAndResultShape extends Operation {

  override def trait_verify(): Either[String, Operation] = {
    // gets rid of all unranked types already
    val params = (this.operands ++ this.results).map(_.typ).collect {
      case a: ShapedType => a
    }
    if (params.isEmpty) Right(this)
    else {
      val firstDim = params.head.getNumDims
      // check ranks of all parameters
      if (params.map(_.getNumDims == firstDim).reduceLeft(_ && _)) then
        Right(this)
      else
        Left(
          s"All parameters of operation '${this.name}' must have the same rank"
        )
    }
  }

}

trait SameInputOutputTensorDims extends Operation {

  override def trait_verify(): Either[String, Operation] = {
    // gets rid of all unranked types already
    val params = (this.operands ++ this.results).map(_.typ).collect {
      case a: ShapedType => a
    }
    if (params.isEmpty) Right(this)
    else {
      val firstShape = params.head.getShape
      // checks if all parameters have the same shape
      if (params.map(_.getShape == firstShape).reduceLeft(_ && _)) then
        Right(this)
      else
        Left(
          s"All parameters of operation '${this.name}' must have compatible shape"
        )
    }
  }

}

trait AllTypesMatch(values: Attribute*) extends Operation {

  override def trait_verify(): Either[String, Operation] = {
    if (values.isEmpty) Right(this)
    else {
      val first = values.head
      if (values.tail.forall(_ == first)) Right(this)
      else
        Left(
          "All parameters of AllTypesMatch must be of the same type in operation " + this.name
        )
    }
  }

}

trait BooleanConditionOrMatchingShape(condition: Attribute, result: Attribute)
    extends Operation {

  override def trait_verify(): Either[String, Operation] = {
    condition match {
      case IntegerType(IntData(1), Signless) => Right(this)
      case x: ShapedType                     =>
        result match {
          case y: ShapedType =>
            if (x.getShape == y.getShape) then Right(this)
            else
              Left(
                s"Condition must be a I1 boolean, or the result of operation '${this.name}' must have the same shape as the condition, but got ${x.getShape} and ${y.getShape}"
              )
          case _ =>
            Left(
              s"Condition must be a I1 boolean, or a shaped type in operation '${this.name}'"
            )
        }
    }
  }

}

trait SameTypeOperands

trait ConstantLike
trait OpAsmOpInterface
trait InferIntRangeInterface
trait Commutative
trait ArithIntegerOverflowFlagsInterface
trait ConditionallySpeculable
trait Idempotent
trait ArithFastMathInterface
trait CastOpInterface
trait ArithRoundingModeInterface
trait SelectLikeOpInterface

trait ArithOp extends SameOperandsAndResultTypes

trait IntBinaryOpWithOverflowFlags
    extends ArithOp,
      NoMemoryEffect,
      InferIntRangeInterface,
      ArithIntegerOverflowFlagsInterface {
  def lhs: Operand[SignlessIntegerOrIndexLike]
  def rhs: Operand[SignlessIntegerOrIndexLike]
  def overflowFlags: IntegerOverflowAttr
  def result: Result[SignlessIntegerOrIndexLike]
}

trait IntBinaryOp extends ArithOp, InferIntRangeInterface {
  def lhs: Operand[SignlessIntegerOrIndexLike]
  def rhs: Operand[SignlessIntegerOrIndexLike]
  def result: Result[SignlessIntegerOrIndexLike]
}

trait TotalIntBinaryOp extends IntBinaryOp, NoMemoryEffect

trait FloatUnaryOp extends ArithOp, NoMemoryEffect, ArithFastMathInterface {
  def operand: Operand[FloatType]
  def result: Result[FloatType]
  def fastmath: FastMathFlagsAttr
}

trait FloatBinaryOp extends ArithOp, NoMemoryEffect, ArithFastMathInterface {
  def lhs: Operand[FloatType]
  def rhs: Operand[FloatType]
  def result: Result[FloatType]
  def fastmath: FastMathFlagsAttr
}

trait CastOp[From <: Attribute, To <: Attribute]
    extends ArithOp,
      SameOperandsAndResultShape,
      SameInputOutputTensorDims,
      NoMemoryEffect,
      CastOpInterface {
  def in: Operand[From]
  def out: Result[To]
}

trait CompareOp
    extends ArithOp,
      NoMemoryEffect,
      SameTypeOperands,
      SameOperandsAndResultShape {
  def result: Result[BoolLike]
}

trait ITOICast
    extends CastOp[
      SignlessFixedWidthIntegerLike,
      SignlessFixedWidthIntegerLike
    ],
      InferIntRangeInterface

trait FTOFCast
    extends CastOp[
      FloatType,
      FloatType
    ]

trait ITOFCast
    extends CastOp[
      SignlessFixedWidthIntegerLike,
      FloatType
    ]

trait FTOICast
    extends CastOp[
      FloatType,
      SignlessFixedWidthIntegerLike
    ]

/*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
||  OPERATION DEFINTION  ||
\*≡==---==≡≡≡≡≡≡≡==---==≡*/

case class Constant(
    val value: TypedAttrInterface,
    val result: Result[TypeAttribute]
) extends DerivedOperation["arith.constant", Constant],
      ConstantLike,
      OpAsmOpInterface,
      AllTypesMatch(value, result.typ),
      InferIntRangeInterface,
      NoMemoryEffect derives DerivedOperationCompanion

case class AddI(
    val lhs: Operand[SignlessIntegerOrIndexLike],
    val rhs: Operand[SignlessIntegerOrIndexLike],
    val result: Result[SignlessIntegerOrIndexLike],
    val overflowFlags: IntegerOverflowAttr
) extends DerivedOperation["arith.addi", AddI],
      IntBinaryOpWithOverflowFlags,
      Commutative derives DerivedOperationCompanion

case class AddUIExtendedOp(
    val lhs: Operand[SignlessIntegerOrIndexLike],
    val rhs: Operand[SignlessIntegerOrIndexLike],
    val result: Result[SignlessIntegerOrIndexLike],
    val overflow: Result[BoolLike]
) extends DerivedOperation["arith.addui_extended", AddUIExtendedOp],
      SameOperandsAndResultTypes,
      NoMemoryEffect,
      Commutative,
      ArithOp derives DerivedOperationCompanion

case class SubI(
    val lhs: Operand[SignlessIntegerOrIndexLike],
    val rhs: Operand[SignlessIntegerOrIndexLike],
    val result: Result[SignlessIntegerOrIndexLike],
    val overflowFlags: IntegerOverflowAttr
) extends DerivedOperation["arith.subi", SubI],
      IntBinaryOpWithOverflowFlags derives DerivedOperationCompanion

case class MulI(
    val lhs: Operand[SignlessIntegerOrIndexLike],
    val rhs: Operand[SignlessIntegerOrIndexLike],
    val result: Result[SignlessIntegerOrIndexLike],
    val overflowFlags: IntegerOverflowAttr
) extends DerivedOperation["arith.muli", MulI],
      IntBinaryOpWithOverflowFlags,
      OpAsmOpInterface,
      Commutative derives DerivedOperationCompanion

case class MulSIExtended(
    val lhs: Operand[SignlessIntegerOrIndexLike],
    val rhs: Operand[SignlessIntegerOrIndexLike],
    val low: Result[SignlessIntegerOrIndexLike],
    val high: Result[SignlessIntegerOrIndexLike]
) extends DerivedOperation["arith.mulsi_extended", MulSIExtended],
      AllTypesMatch(lhs.typ, rhs.typ, low.typ, high.typ),
      Commutative,
      NoMemoryEffect,
      ArithOp derives DerivedOperationCompanion

case class MulUIExtended(
    val lhs: Operand[SignlessIntegerOrIndexLike],
    val rhs: Operand[SignlessIntegerOrIndexLike],
    val low: Result[SignlessIntegerOrIndexLike],
    val high: Result[SignlessIntegerOrIndexLike]
) extends DerivedOperation["arith.mului_extended", MulUIExtended],
      AllTypesMatch(lhs.typ, rhs.typ, low.typ, high.typ),
      Commutative,
      NoMemoryEffect,
      ArithOp derives DerivedOperationCompanion

case class DivUI(
    val lhs: Operand[SignlessIntegerOrIndexLike],
    val rhs: Operand[SignlessIntegerOrIndexLike],
    val result: Result[SignlessIntegerOrIndexLike]
) extends DerivedOperation["arith.divui", DivUI],
      ConditionallySpeculable,
      IntBinaryOp derives DerivedOperationCompanion

case class DivSI(
    val lhs: Operand[SignlessIntegerOrIndexLike],
    val rhs: Operand[SignlessIntegerOrIndexLike],
    val result: Result[SignlessIntegerOrIndexLike]
) extends DerivedOperation["arith.divsi", DivSI],
      ConditionallySpeculable,
      IntBinaryOp derives DerivedOperationCompanion

case class CeilDivUi(
    val lhs: Operand[SignlessIntegerOrIndexLike],
    val rhs: Operand[SignlessIntegerOrIndexLike],
    val result: Result[SignlessIntegerOrIndexLike]
) extends DerivedOperation["arith.ceildivui", CeilDivUi],
      ConditionallySpeculable,
      IntBinaryOp derives DerivedOperationCompanion

case class CeilDivSi(
    val lhs: Operand[SignlessIntegerOrIndexLike],
    val rhs: Operand[SignlessIntegerOrIndexLike],
    val result: Result[SignlessIntegerOrIndexLike]
) extends DerivedOperation["arith.ceildivsi", CeilDivSi],
      ConditionallySpeculable,
      IntBinaryOp derives DerivedOperationCompanion

case class FloorDivSI(
    val lhs: Operand[SignlessIntegerOrIndexLike],
    val rhs: Operand[SignlessIntegerOrIndexLike],
    val result: Result[SignlessIntegerOrIndexLike]
) extends DerivedOperation["arith.floordivsi", FloorDivSI],
      TotalIntBinaryOp derives DerivedOperationCompanion

case class RemUI(
    val lhs: Operand[SignlessIntegerOrIndexLike],
    val rhs: Operand[SignlessIntegerOrIndexLike],
    val result: Result[SignlessIntegerOrIndexLike]
) extends DerivedOperation["arith.remui", RemUI],
      TotalIntBinaryOp derives DerivedOperationCompanion

case class RemSI(
    lhs: Operand[SignlessIntegerOrIndexLike],
    rhs: Operand[SignlessIntegerOrIndexLike],
    result: Result[SignlessIntegerOrIndexLike]
) extends DerivedOperation["arith.remsi", RemSI],
      TotalIntBinaryOp derives DerivedOperationCompanion

case class AndI(
    val lhs: Operand[SignlessIntegerOrIndexLike],
    val rhs: Operand[SignlessIntegerOrIndexLike],
    val result: Result[SignlessIntegerOrIndexLike]
) extends DerivedOperation["arith.andi", AndI],
      TotalIntBinaryOp,
      Commutative,
      Idempotent derives DerivedOperationCompanion

case class OrI(
    val lhs: Operand[SignlessIntegerOrIndexLike],
    val rhs: Operand[SignlessIntegerOrIndexLike],
    val result: Result[SignlessIntegerOrIndexLike]
) extends DerivedOperation["arith.ori", OrI],
      TotalIntBinaryOp,
      Commutative,
      Idempotent derives DerivedOperationCompanion

case class XOrI(
    val lhs: Operand[SignlessIntegerOrIndexLike],
    val rhs: Operand[SignlessIntegerOrIndexLike],
    val result: Result[SignlessIntegerOrIndexLike]
) extends DerivedOperation["arith.xori", XOrI],
      TotalIntBinaryOp,
      Commutative derives DerivedOperationCompanion

case class ShLI(
    val lhs: Operand[SignlessIntegerOrIndexLike],
    val rhs: Operand[SignlessIntegerOrIndexLike],
    val result: Result[SignlessIntegerOrIndexLike],
    val overflowFlags: IntegerOverflowAttr
) extends DerivedOperation["arith.shli", ShLI],
      IntBinaryOpWithOverflowFlags derives DerivedOperationCompanion

case class ShRUI(
    val lhs: Operand[SignlessIntegerOrIndexLike],
    val rhs: Operand[SignlessIntegerOrIndexLike],
    val result: Result[SignlessIntegerOrIndexLike]
) extends DerivedOperation["arith.shrui", ShRUI],
      TotalIntBinaryOp derives DerivedOperationCompanion

case class ShRSI(
    val lhs: Operand[SignlessIntegerOrIndexLike],
    val rhs: Operand[SignlessIntegerOrIndexLike],
    val result: Result[SignlessIntegerOrIndexLike]
) extends DerivedOperation["arith.shrsi", ShRSI],
      TotalIntBinaryOp derives DerivedOperationCompanion

case class NegF(
    val operand: Operand[FloatType],
    val result: Result[FloatType],
    val fastmath: FastMathFlagsAttr
) extends DerivedOperation["arith.negf", NegF],
      FloatUnaryOp derives DerivedOperationCompanion

case class AddF(
    val lhs: Operand[FloatType],
    val rhs: Operand[FloatType],
    val result: Result[FloatType],
    val fastmath: FastMathFlagsAttr
) extends DerivedOperation["arith.addf", AddF],
      FloatBinaryOp,
      Commutative derives DerivedOperationCompanion

case class SubF(
    val lhs: Operand[FloatType],
    val rhs: Operand[FloatType],
    val result: Result[FloatType],
    val fastmath: FastMathFlagsAttr
) extends DerivedOperation["arith.subf", SubF],
      FloatBinaryOp derives DerivedOperationCompanion

case class MaximumF(
    val lhs: Operand[FloatType],
    val rhs: Operand[FloatType],
    val result: Result[FloatType],
    val fastmath: FastMathFlagsAttr
) extends DerivedOperation["arith.maximumf", MaximumF],
      FloatBinaryOp,
      Commutative derives DerivedOperationCompanion

case class MaxNumF(
    val lhs: Operand[FloatType],
    val rhs: Operand[FloatType],
    val result: Result[FloatType],
    val fastmath: FastMathFlagsAttr
) extends DerivedOperation["arith.maxnumf", MaxNumF],
      FloatBinaryOp,
      Commutative derives DerivedOperationCompanion

case class MaxSI(
    val lhs: Operand[SignlessIntegerOrIndexLike],
    val rhs: Operand[SignlessIntegerOrIndexLike],
    val result: Result[SignlessIntegerOrIndexLike]
) extends DerivedOperation["arith.maxsi", MaxSI],
      TotalIntBinaryOp,
      Commutative derives DerivedOperationCompanion

case class MaxUI(
    val lhs: Operand[SignlessIntegerOrIndexLike],
    val rhs: Operand[SignlessIntegerOrIndexLike],
    val result: Result[SignlessIntegerOrIndexLike]
) extends DerivedOperation["arith.maxui", MaxUI],
      TotalIntBinaryOp,
      Commutative derives DerivedOperationCompanion

case class MinimumF(
    val lhs: Operand[FloatType],
    val rhs: Operand[FloatType],
    val result: Result[FloatType],
    val fastmath: FastMathFlagsAttr
) extends DerivedOperation["arith.minimumf", MinimumF],
      FloatBinaryOp,
      Commutative derives DerivedOperationCompanion

case class MinNumF(
    val lhs: Operand[FloatType],
    val rhs: Operand[FloatType],
    val result: Result[FloatType],
    val fastmath: FastMathFlagsAttr
) extends DerivedOperation["arith.minnumf", MinNumF],
      FloatBinaryOp,
      Commutative derives DerivedOperationCompanion

case class MinSI(
    val lhs: Operand[SignlessIntegerOrIndexLike],
    val rhs: Operand[SignlessIntegerOrIndexLike],
    val result: Result[SignlessIntegerOrIndexLike]
) extends DerivedOperation["arith.minsi", MinSI],
      TotalIntBinaryOp,
      Commutative derives DerivedOperationCompanion

case class MinUI(
    val lhs: Operand[SignlessIntegerOrIndexLike],
    val rhs: Operand[SignlessIntegerOrIndexLike],
    val result: Result[SignlessIntegerOrIndexLike]
) extends DerivedOperation["arith.minui", MinUI],
      TotalIntBinaryOp,
      Commutative derives DerivedOperationCompanion

case class MulF(
    val lhs: Operand[FloatType],
    val rhs: Operand[FloatType],
    val result: Result[FloatType],
    val fastmath: FastMathFlagsAttr
) extends DerivedOperation["arith.mulf", MulF],
      FloatBinaryOp,
      Commutative derives DerivedOperationCompanion

case class DivF(
    val lhs: Operand[FloatType],
    val rhs: Operand[FloatType],
    val result: Result[FloatType],
    val fastmath: FastMathFlagsAttr
) extends DerivedOperation["arith.divf", DivF],
      FloatBinaryOp derives DerivedOperationCompanion

case class RemF(
    val lhs: Operand[FloatType],
    val rhs: Operand[FloatType],
    val result: Result[FloatType],
    val fastmath: FastMathFlagsAttr
) extends DerivedOperation["arith.remf", RemF],
      FloatBinaryOp derives DerivedOperationCompanion

case class ExtUI(
    val in: Operand[SignlessFixedWidthIntegerLike],
    val out: Result[SignlessFixedWidthIntegerLike]
) extends DerivedOperation["arith.extui", ExtUI],
      ITOICast derives DerivedOperationCompanion

case class ExtSI(
    val in: Operand[SignlessFixedWidthIntegerLike],
    val out: Result[SignlessFixedWidthIntegerLike]
) extends DerivedOperation["arith.extsi", ExtSI],
      ITOICast derives DerivedOperationCompanion

case class ExtF(
    val in: Operand[FloatType],
    val out: Result[FloatType],
    val fastmath: Option[FastMathFlagsAttr] = None
) extends DerivedOperation["arith.extf", ExtF],
      ArithFastMathInterface,
      FTOFCast derives DerivedOperationCompanion

case class ScalingExtF(
    val in: Operand[FloatType],
    val scale: Operand[FloatType],
    val out: Result[FloatType],
    val fastmath: Option[FastMathFlagsAttr]
) extends DerivedOperation["arith.scaling_extf", ScalingExtF],
      NoMemoryEffect,
      SameInputOutputTensorDims,
      ArithFastMathInterface,
      CastOpInterface,
      ArithOp derives DerivedOperationCompanion

case class TruncI(
    val in: Operand[SignlessFixedWidthIntegerLike],
    val out: Result[SignlessFixedWidthIntegerLike],
    val overflowFlags: IntegerOverflowAttr
) extends DerivedOperation["arith.trunci", TruncI],
      NoMemoryEffect,
      SameOperandsAndResultShape,
      SameInputOutputTensorDims,
      CastOpInterface,
      InferIntRangeInterface,
      ArithIntegerOverflowFlagsInterface,
      ArithOp derives DerivedOperationCompanion

case class TruncF(
    val in: Operand[FloatType],
    val out: Result[FloatType],
    val roundingmode: Option[RoundingModeAttr],
    val fastmath: Option[FastMathFlagsAttr]
) extends DerivedOperation["arith.truncf", TruncF],
      SameOperandsAndResultShape,
      NoMemoryEffect,
      SameInputOutputTensorDims,
      ArithRoundingModeInterface,
      ArithFastMathInterface,
      CastOpInterface,
      ArithOp derives DerivedOperationCompanion

case class ScalingTruncF(
    val in: Operand[FloatType],
    val scale: Operand[FloatType],
    val out: Result[FloatType],
    val roundingmode: Option[RoundingModeAttr],
    val fastmath: Option[FastMathFlagsAttr]
) extends DerivedOperation["arith.scaling_truncf", ScalingTruncF],
      NoMemoryEffect,
      SameInputOutputTensorDims,
      ArithRoundingModeInterface,
      ArithFastMathInterface,
      CastOpInterface,
      ArithOp derives DerivedOperationCompanion

case class UIToFP(
    val in: Operand[SignlessFixedWidthIntegerLike],
    val out: Result[FloatType]
) extends DerivedOperation["arith.uitofp", UIToFP],
      ITOFCast derives DerivedOperationCompanion

case class SIToFP(
    in: Operand[SignlessFixedWidthIntegerLike],
    out: Result[FloatType]
) extends DerivedOperation["arith.sitofp", SIToFP],
      ITOFCast derives DerivedOperationCompanion

case class FPToUI(
    val in: Operand[FloatType],
    val out: Result[SignlessFixedWidthIntegerLike]
) extends DerivedOperation["arith.fptoui", FPToUI],
      FTOICast derives DerivedOperationCompanion

case class FPToSI(
    val in: Operand[FloatType],
    val out: Result[SignlessFixedWidthIntegerLike]
) extends DerivedOperation["arith.fptosi", FPToSI],
      FTOICast derives DerivedOperationCompanion

case class IndexCast(
    val in: Operand[IndexCastTypeConstraint],
    val out: Result[IndexCastTypeConstraint]
) extends DerivedOperation["arith.index_cast", IndexCast],
      CastOp[IndexCastTypeConstraint, IndexCastTypeConstraint],
      InferIntRangeInterface derives DerivedOperationCompanion

case class IndexCastUI(
    val in: Operand[IndexCastTypeConstraint],
    val out: Result[IndexCastTypeConstraint]
) extends DerivedOperation["arith.index_castui", IndexCastUI],
      CastOp[IndexCastTypeConstraint, IndexCastTypeConstraint],
      InferIntRangeInterface derives DerivedOperationCompanion

case class BitCast(
    val in: Operand[BitcastTypeConstraint],
    val out: Result[BitcastTypeConstraint]
) extends DerivedOperation["arith.bitcast", BitCast],
      CastOp[BitcastTypeConstraint, BitcastTypeConstraint]
    derives DerivedOperationCompanion

case class CmpI(
    val predicate: CmpIPredicateAttr,
    val lhs: Operand[SignlessIntegerOrIndexLike],
    val rhs: Operand[SignlessIntegerOrIndexLike],
    val result: Result[BoolLike]
) extends DerivedOperation["arith.cmpi", CmpI],
      InferIntRangeInterface,
      CompareOp derives DerivedOperationCompanion

case class CmpF(
    val predicate: CmpIPredicateAttr,
    val lhs: Operand[FloatType],
    val rhs: Operand[FloatType],
    val result: Result[BoolLike],
    val fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none)
) extends DerivedOperation["arith.cmpf", CmpF],
      ArithFastMathInterface,
      CompareOp derives DerivedOperationCompanion

case class SelectOp(
    val condition: Operand[BoolLike],
    val trueValue: Operand[TypeAttribute],
    val falseValue: Operand[TypeAttribute],
    val result: Result[TypeAttribute]
) extends DerivedOperation["arith.select", SelectOp],
      BooleanConditionOrMatchingShape(condition.typ, result.typ),
      AllTypesMatch(trueValue.typ, falseValue.typ, result.typ),
      InferIntRangeInterface,
      SelectLikeOpInterface,
      NoMemoryEffect,
      ArithOp derives DerivedOperationCompanion

val ArithDialect =
  summonDialect[
    EmptyTuple,
    (
        AddF,
        AddI,
        AddUIExtendedOp,
        AndI,
        BitCast,
        CeilDivSi,
        CeilDivUi,
        CmpF,
        CmpI,
        Constant,
        DivF,
        DivSI,
        DivUI,
        ExtF,
        ExtSI,
        ExtUI,
        FloorDivSI,
        FPToSI,
        FPToUI,
        IndexCast,
        IndexCastUI,
        MaximumF,
        MaxNumF,
        MaxSI,
        MaxUI,
        MinimumF,
        MinNumF,
        MinSI,
        MinUI,
        MulF,
        MulI,
        MulSIExtended,
        MulUIExtended,
        NegF,
        OrI,
        RemF,
        RemSI,
        RemUI,
        ScalingExtF,
        ScalingTruncF,
        SelectOp,
        ShLI,
        ShRSI,
        ShRUI,
        SIToFP,
        SubF,
        SubI,
        TruncF,
        TruncI,
        UIToFP,
        XOrI,
        IndexCast
    )
  ](Seq(FastMathFlagsAttr))
