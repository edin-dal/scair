package scair.dialects.arith

import fastparse.*
import scair.Printer
import scair.clair.codegen.*
import scair.clair.macros.*
import scair.core.macros.TransparentData
import scair.dialects.arith.canonicalization.given
import scair.dialects.builtin.*
import scair.enums.enumattr.*
import scair.ir.*
import scair.parse.Parser
import scair.utils.*

import scala.collection.immutable.*

//
// ░█████╗░ ██████╗░ ██╗ ████████╗ ██╗░░██╗
// ██╔══██╗ ██╔══██╗ ██║ ╚══██╔══╝ ██║░░██║
// ███████║ ██████╔╝ ██║ ░░░██║░░░ ███████║
// ██╔══██║ ██╔══██╗ ██║ ░░░██║░░░ ██╔══██║
// ██║░░██║ ██║░░██║ ██║ ░░░██║░░░ ██║░░██║
// ╚═╝░░╚═╝ ╚═╝░░╚═╝ ╚═╝ ░░░╚═╝░░░ ╚═╝░░╚═╝
//

// TODO: Upstream Arith natively support vector or other containers of it's operands and results type
// i.e., add vectors not just integers.
// Let's keep it progressive here though.

/*≡==--==≡≡≡≡≡==--=≡≡*\
||       ENUMS       ||
\*≡==---==≡≡≡==---==≡*/

enum CmpIPredicate(name: String) extends I64Enum(name):
  case eq extends CmpIPredicate("eq")
  case ne extends CmpIPredicate("ne")
  case slt extends CmpIPredicate("slt")
  case sle extends CmpIPredicate("sle")
  case sgt extends CmpIPredicate("sgt")
  case sge extends CmpIPredicate("sge")
  case ult extends CmpIPredicate("ult")
  case ule extends CmpIPredicate("ule")
  case ugt extends CmpIPredicate("ugt")
  case uge extends CmpIPredicate("uge")

enum CmpFPredicate(name: String) extends I64Enum(name):
  case AlwaysFalse extends CmpFPredicate("false")
  case OEQ extends CmpFPredicate("oeq")
  case OGT extends CmpFPredicate("ogt")
  case OGE extends CmpFPredicate("oge")
  case OLT extends CmpFPredicate("olt")
  case OLE extends CmpFPredicate("ole")
  case ONE extends CmpFPredicate("one")
  case ORD extends CmpFPredicate("ord")
  case UEQ extends CmpFPredicate("ueq")
  case UGT extends CmpFPredicate("ugt")
  case UGE extends CmpFPredicate("uge")
  case ULT extends CmpFPredicate("ult")
  case ULE extends CmpFPredicate("ule")
  case UNE extends CmpFPredicate("une")
  case UNO extends CmpFPredicate("uno")
  case AlwaysTrue extends CmpFPredicate("true")

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

given Ordering[FastMathFlag] = Ordering.by(_.ordinal)

type FastMathFlags = SortedSet[FastMathFlag]

object FastMathFlags:
  val none: FastMathFlags = SortedSet.empty
  val reassoc: FastMathFlags = SortedSet(FastMathFlag.reassoc)
  val nnan: FastMathFlags = SortedSet(FastMathFlag.nnan)
  val ninf: FastMathFlags = SortedSet(FastMathFlag.ninf)
  val nsz: FastMathFlags = SortedSet(FastMathFlag.nsz)
  val arcp: FastMathFlags = SortedSet(FastMathFlag.arcp)
  val contract: FastMathFlags = SortedSet(FastMathFlag.contract)
  val afn: FastMathFlags = SortedSet(FastMathFlag.afn)

  val fast: FastMathFlags =
    reassoc | nnan | ninf | nsz | arcp | contract | afn

  def apply(flags: FastMathFlag*): FastMathFlags = SortedSet(flags*)

given AttributeCompanion[FastMathFlagsAttr]:
  override def name: String = "arith.fastmath"

  override def parse[$: P](using Parser): P[FastMathFlagsAttr] =
    P(
      "<" ~
        ("none" | "reassoc" | "nnan" | "ninf" | "nsz" | "arcp" | "contract" |
          "afn" | "fast").!.rep(
          sep = ","
        ) ~ ">"
    ).flatMap { parsedFlags =>
      if parsedFlags.isEmpty then
        Fail("FastMathFlagsAttr expects at least one flag")
      else
        val flags = parsedFlags
          .map(_ match
            case "none"     => FastMathFlags.none
            case "reassoc"  => FastMathFlags.reassoc
            case "nnan"     => FastMathFlags.nnan
            case "ninf"     => FastMathFlags.ninf
            case "nsz"      => FastMathFlags.nsz
            case "arcp"     => FastMathFlags.arcp
            case "contract" => FastMathFlags.contract
            case "afn"      => FastMathFlags.afn
            case "fast"     => FastMathFlags.fast
            // Unreachable per above logic; I feel like FastParse could type this better with new types!
            case f => throw new Exception(s"Invalid fastmath flag '$f'"))
          .reduce(_ | _)
        Pass(FastMathFlagsAttr(flags))
    }

case class FastMathFlagsAttr(val flags: FastMathFlags)
    extends scair.ir.DataAttribute[FastMathFlags]("arith.fastmath", flags)
    derives TransparentData:

  override def customPrint(p: Printer) =
    p.print("#arith.fastmath<")
    flags match
      case FastMathFlags.none =>
        p.print("none")
      case FastMathFlags.fast =>
        p.print("fast")
      case _ =>
        p.printListF(
          flags,
          f =>
            p.print(f match
              case FastMathFlag.reassoc  => "reassoc"
              case FastMathFlag.nnan     => "nnan"
              case FastMathFlag.ninf     => "ninf"
              case FastMathFlag.nsz      => "nsz"
              case FastMathFlag.arcp     => "arcp"
              case FastMathFlag.contract => "contract"
              case FastMathFlag.afn      => "afn"),
          sep = ",",
        )
    p.print(">")

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||  TYPES AND CONSTRAINTS  ||
\*≡==---==≡≡≡≡≡≡≡≡≡==---==≡*/

// TODO: this should constrain the IntegerType to be signless, i.e., i32, i8, etc.
// also, should be called SignlessIntegerOrIndexLike
type AnyIntegerType = IntegerType | IndexType
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
type BoolLike = I1 | VectorType | ContainerType
// TODO: same old same old
type SignlessIntegerOrFloatLike = IntegerType | FloatType
// TODO: this should be constrained to be a signless integer, or vector or tensor of a signless integer
type SignlessFixedWidthIntegerLike = IntegerType | VectorType | ContainerType
// TODO: MemrefType here should be looked at more closely to how it is actually implemented in MLIR
type BitcastType = SignlessIntegerOrFloatLike | MemrefType
// TODO: this is specifically a memref type of anysignless integer or index
type IndexCastTypeConstraint = AnyIntegerType | MemrefType

trait SameOperandsAndResultTypes extends Operation:

  override def traitVerify(): OK[Operation] =
    val params = (this.operands ++ this.results)
    if params.isEmpty then OK(this)
    else
      val first = params.head.typ
      if params.tail.forall(_.typ == first) then OK(this)
      else
        Err(
          "All parameters of TypeConstraint must be of the same type in operation " +
            this.name
        )

trait SameOperandsAndResultShape extends Operation:

  override def traitVerify(): OK[Operation] =
    // gets rid of all unranked types already
    val params = (this.operands ++ this.results).map(_.typ).collect {
      case a: ShapedType => a
    }
    if params.isEmpty then OK(this)
    else
      val firstDim = params.head.getNumDims
      // check ranks of all parameters
      if params.map(_.getNumDims == firstDim).reduceLeft(_ && _) then OK(this)
      else
        Err(
          s"All parameters of operation '${this.name}' must have the same rank"
        )

trait SameInputOutputTensorDims extends Operation:

  override def traitVerify(): OK[Operation] =
    // gets rid of all unranked types already
    val params = (this.operands ++ this.results).map(_.typ).collect {
      case a: ShapedType => a
    }
    if params.isEmpty then OK(this)
    else
      val firstShape = params.head.getShape
      // checks if all parameters have the same shape
      if params.map(_.getShape == firstShape).reduceLeft(_ && _) then OK(this)
      else
        Err(
          s"All parameters of operation '${this.name}' must have compatible shape"
        )

trait AllTypesMatch(values: Attribute*) extends Operation:

  override def traitVerify(): OK[Operation] =
    if values.isEmpty then OK(this)
    else
      val first = values.head
      if values.tail.forall(_ == first) then OK(this)
      else
        Err(
          "All parameters of AllTypesMatch must be of the same type in operation " +
            this.name
        )

trait BooleanConditionOrMatchingShape(condition: Attribute, result: Attribute)
    extends Operation:

  override def traitVerify(): OK[Operation] =
    condition match
      case IntegerType(IntData(1), Signless) => OK(this)
      case x: ShapedType                     =>
        result match
          case y: ShapedType =>
            if x.getShape == y.getShape then OK(this)
            else
              Err(
                s"Condition must be a I1 boolean, or the result of operation '${this
                    .name}' must have the same shape as the condition, but got ${x
                    .getShape} and ${y.getShape}"
              )
          case _ =>
            Err(
              s"Condition must be a I1 boolean, or a shaped type in operation '${this
                  .name}'"
            )
      case _ =>
        Err(
          s"Condition must be a I1 boolean, or a shaped type in operation '${this
              .name}'"
        )

/*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
||  OPERATION DEFINTION  ||
\*≡==---==≡≡≡≡≡≡≡==---==≡*/

case class AddF(
    val lhs: Operand[FloatType],
    val rhs: Operand[FloatType],
    val result: Result[FloatType],
    val fastmath: FastMathFlagsAttr,
) extends DerivedOperation["arith.addf", AddF]
    with SameOperandsAndResultTypes
    with NoMemoryEffect
    with Commutative derives DerivedOperationCompanion

// TODO Apparently there's a new overflow flag here, overlooking for now.
case class AddI(
    val lhs: Operand[AnyIntegerType],
    val rhs: Operand[AnyIntegerType],
    val result: Result[AnyIntegerType],
) extends DerivedOperation["arith.addi", AddI]
    with SameOperandsAndResultTypes
    with NoMemoryEffect
    with Commutative derives DerivedOperationCompanion

case class AddUIExtendedOp(
    val lhs: Operand[AnyIntegerType],
    val rhs: Operand[AnyIntegerType],
    val result: Result[AnyIntegerType],
    val overflow: Result[BoolLike],
) extends DerivedOperation[
      "arith.addui_extended",
      AddUIExtendedOp,
    ]
    with SameOperandsAndResultTypes
    with NoMemoryEffect
    with Commutative derives DerivedOperationCompanion

case class AndI(
    val lhs: Operand[AnyIntegerType],
    val rhs: Operand[AnyIntegerType],
    val result: Result[I1],
) extends DerivedOperation["arith.andi", AndI]
    with SameOperandsAndResultTypes
    with NoMemoryEffect
    with Commutative derives DerivedOperationCompanion

case class BitCast(
    val in: Operand[BitcastType],
    val out: Result[BitcastType],
) extends DerivedOperation["arith.bitcast", BitCast]
    with SameOperandsAndResultShape
    with SameInputOutputTensorDims
    with NoMemoryEffect derives DerivedOperationCompanion

case class CeilDivSi(
    val lhs: Operand[AnyIntegerType],
    val rhs: Operand[AnyIntegerType],
    val result: Result[AnyIntegerType],
) extends DerivedOperation["arith.ceildivsi", CeilDivSi]
    with SameOperandsAndResultTypes
    with NoMemoryEffect derives DerivedOperationCompanion

case class CeilDivUi(
    val lhs: Operand[AnyIntegerType],
    val rhs: Operand[AnyIntegerType],
    val result: Result[AnyIntegerType],
) extends DerivedOperation["arith.ceildivui", CeilDivUi]
    with SameOperandsAndResultTypes
    with NoMemoryEffect derives DerivedOperationCompanion

case class CmpF(
    val lhs: Operand[FloatType],
    val rhs: Operand[FloatType],
    val result: Result[I1],
    val predicate: CmpFPredicate,
    val fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
) extends DerivedOperation["arith.cmpf", CmpF]
    with NoMemoryEffect derives DerivedOperationCompanion

case class CmpI(
    val lhs: Operand[AnyIntegerType],
    val rhs: Operand[AnyIntegerType],
    val result: Result[I1],
    val predicate: CmpIPredicate,
) extends DerivedOperation["arith.cmpi", CmpI]
    with NoMemoryEffect derives DerivedOperationCompanion

case class Constant(
    val value: Attribute,
    val result: Result[Attribute],
) extends DerivedOperation["arith.constant", Constant]
    with NoMemoryEffect
    with ConstantLike(value) derives DerivedOperationCompanion

case class DivF(
    val lhs: Operand[FloatType],
    val rhs: Operand[FloatType],
    val result: Result[FloatType],
    val fastmath: FastMathFlagsAttr,
) extends DerivedOperation["arith.divf", DivF]
    with SameOperandsAndResultTypes
    with NoMemoryEffect derives DerivedOperationCompanion

case class DivSI(
    val lhs: Operand[AnyIntegerType],
    val rhs: Operand[AnyIntegerType],
    val result: Result[AnyIntegerType],
) extends DerivedOperation["arith.divsi", DivSI]
    with SameOperandsAndResultTypes
    with NoMemoryEffect derives DerivedOperationCompanion

case class DivUI(
    val lhs: Operand[AnyIntegerType],
    val rhs: Operand[AnyIntegerType],
    val result: Result[AnyIntegerType],
) extends DerivedOperation["arith.divui", DivUI]
    with SameOperandsAndResultTypes
    with NoMemoryEffect derives DerivedOperationCompanion

case class ExtF(
    val in: Operand[FloatType],
    val out: Result[FloatType],
    val fastmath: Option[FastMathFlagsAttr] = None,
) extends DerivedOperation["arith.extf", ExtF]
    with SameOperandsAndResultShape
    with SameInputOutputTensorDims
    with NoMemoryEffect derives DerivedOperationCompanion

case class ExtSI(
    val in: Operand[SignlessFixedWidthIntegerLike],
    val out: Result[SignlessFixedWidthIntegerLike],
) extends DerivedOperation["arith.extsi", ExtSI]
    with SameOperandsAndResultShape
    with SameInputOutputTensorDims
    with NoMemoryEffect derives DerivedOperationCompanion

case class ExtUI(
    val in: Operand[SignlessFixedWidthIntegerLike],
    val out: Result[SignlessFixedWidthIntegerLike],
) extends DerivedOperation["arith.extui", ExtUI]
    with SameOperandsAndResultShape
    with SameInputOutputTensorDims
    with NoMemoryEffect derives DerivedOperationCompanion

case class FloorDivSI(
    val lhs: Operand[AnyIntegerType],
    val rhs: Operand[AnyIntegerType],
    val result: Result[AnyIntegerType],
) extends DerivedOperation["arith.floordivsi", FloorDivSI]
    with SameOperandsAndResultTypes
    with NoMemoryEffect derives DerivedOperationCompanion

case class FPToSI(
    val in: Operand[FloatType],
    val out: Result[SignlessFixedWidthIntegerLike],
) extends DerivedOperation["arith.fptosi", FPToSI]
    with SameOperandsAndResultShape
    with SameInputOutputTensorDims derives DerivedOperationCompanion

case class FPToUI(
    val in: Operand[FloatType],
    val out: Result[SignlessFixedWidthIntegerLike],
) extends DerivedOperation["arith.fptoui", FPToUI]
    with SameOperandsAndResultShape
    with SameInputOutputTensorDims
    with NoMemoryEffect derives DerivedOperationCompanion

case class IndexCast(
    val in: Operand[IndexCastTypeConstraint],
    val result: Result[IndexCastTypeConstraint],
) extends DerivedOperation["arith.index_cast", IndexCast]
    with SameOperandsAndResultShape
    with SameInputOutputTensorDims
    with NoMemoryEffect derives DerivedOperationCompanion

case class IndexCastUI(
    val in: Operand[IndexCastTypeConstraint],
    val result: Result[IndexCastTypeConstraint],
) extends DerivedOperation["arith.index_castui", IndexCastUI]
    with SameOperandsAndResultShape
    with SameInputOutputTensorDims
    with NoMemoryEffect derives DerivedOperationCompanion

case class MaximumF(
    val lhs: Operand[FloatType],
    val rhs: Operand[FloatType],
    val result: Result[FloatType],
    val fastmath: FastMathFlagsAttr,
) extends DerivedOperation["arith.maximumf", MaximumF]
    with SameOperandsAndResultTypes
    with NoMemoryEffect
    with Commutative derives DerivedOperationCompanion

case class MaxNumF(
    val lhs: Operand[FloatType],
    val rhs: Operand[FloatType],
    val result: Result[FloatType],
    val fastmath: FastMathFlagsAttr,
) extends DerivedOperation["arith.maxnumf", MaxNumF]
    with SameOperandsAndResultTypes
    with NoMemoryEffect
    with Commutative derives DerivedOperationCompanion

case class MaxSI(
    val lhs: Operand[AnyIntegerType],
    val rhs: Operand[AnyIntegerType],
    val result: Result[AnyIntegerType],
) extends DerivedOperation["arith.maxsi", MaxSI]
    with SameOperandsAndResultTypes
    with NoMemoryEffect
    with Commutative derives DerivedOperationCompanion

case class MaxUI(
    val lhs: Operand[AnyIntegerType],
    val rhs: Operand[AnyIntegerType],
    val result: Result[AnyIntegerType],
) extends DerivedOperation["arith.maxui", MaxUI]
    with SameOperandsAndResultTypes
    with NoMemoryEffect
    with Commutative derives DerivedOperationCompanion

case class MinimumF(
    val lhs: Operand[FloatType],
    val rhs: Operand[FloatType],
    val result: Result[FloatType],
    val fastmath: FastMathFlagsAttr,
) extends DerivedOperation["arith.minimumf", MinimumF]
    with SameOperandsAndResultTypes
    with NoMemoryEffect
    with Commutative derives DerivedOperationCompanion

case class MinNumF(
    val lhs: Operand[FloatType],
    val rhs: Operand[FloatType],
    val result: Result[FloatType],
    val fastmath: FastMathFlagsAttr,
) extends DerivedOperation["arith.minnumf", MinNumF]
    with SameOperandsAndResultTypes
    with NoMemoryEffect
    with Commutative derives DerivedOperationCompanion

case class MinSI(
    val lhs: Operand[AnyIntegerType],
    val rhs: Operand[AnyIntegerType],
    val result: Result[AnyIntegerType],
) extends DerivedOperation["arith.minsi", MinSI]
    with SameOperandsAndResultTypes
    with NoMemoryEffect
    with Commutative derives DerivedOperationCompanion

case class MinUI(
    val lhs: Operand[AnyIntegerType],
    val rhs: Operand[AnyIntegerType],
    val result: Result[AnyIntegerType],
) extends DerivedOperation["arith.minui", MinUI]
    with SameOperandsAndResultTypes
    with NoMemoryEffect
    with Commutative derives DerivedOperationCompanion

case class MulF(
    val lhs: Operand[FloatType],
    val rhs: Operand[FloatType],
    val result: Result[FloatType],
    val fastmath: FastMathFlagsAttr,
) extends DerivedOperation["arith.mulf", MulF]
    with NoMemoryEffect
    with Commutative derives DerivedOperationCompanion

case class MulI(
    val lhs: Operand[AnyIntegerType],
    val rhs: Operand[AnyIntegerType],
    val result: Result[AnyIntegerType],
) extends DerivedOperation["arith.muli", MulI]
    with NoMemoryEffect
    with Commutative derives DerivedOperationCompanion

case class MulSIExtended(
    val lhs: Operand[AnyIntegerType],
    val rhs: Operand[AnyIntegerType],
    val low: Result[AnyIntegerType],
    val high: Result[AnyIntegerType],
) extends DerivedOperation["arith.mulsi_extended", MulSIExtended]
    with AllTypesMatch(lhs.typ, rhs.typ, low.typ, high.typ)
    with NoMemoryEffect
    with Commutative derives DerivedOperationCompanion

case class MulUIExtended(
    val lhs: Operand[AnyIntegerType],
    val rhs: Operand[AnyIntegerType],
    val low: Result[AnyIntegerType],
    val high: Result[AnyIntegerType],
) extends DerivedOperation["arith.mului_extended", MulUIExtended]
    with AllTypesMatch(lhs.typ, rhs.typ, low.typ, high.typ)
    with NoMemoryEffect
    with Commutative derives DerivedOperationCompanion

case class NegF(
    val operand: Operand[FloatType],
    val result: Result[FloatType],
    val fastmath: FastMathFlagsAttr,
) extends DerivedOperation["arith.negf", NegF]
    with NoMemoryEffect
    with SameOperandsAndResultTypes derives DerivedOperationCompanion

case class OrI(
    val lhs: Operand[AnyIntegerType],
    val rhs: Operand[AnyIntegerType],
    val result: Result[I1],
) extends DerivedOperation["arith.ori", OrI]
    with NoMemoryEffect
    with Commutative derives DerivedOperationCompanion

case class RemF(
    val lhs: Operand[FloatType],
    val rhs: Operand[FloatType],
    val result: Result[FloatType],
    val fastmath: FastMathFlagsAttr,
) extends DerivedOperation["arith.remf", RemF]
    with NoMemoryEffect
    with SameOperandsAndResultTypes derives DerivedOperationCompanion

case class RemSI(
    lhs: Operand[AnyIntegerType],
    rhs: Operand[AnyIntegerType],
    result: Result[AnyIntegerType],
) extends DerivedOperation["arith.remsi", RemSI]
    with NoMemoryEffect
    with SameOperandsAndResultTypes derives DerivedOperationCompanion

case class RemUI(
    lhs: Operand[AnyIntegerType],
    rhs: Operand[AnyIntegerType],
    result: Result[AnyIntegerType],
) extends DerivedOperation["arith.remui", RemUI]
    with NoMemoryEffect
    with SameOperandsAndResultTypes derives DerivedOperationCompanion

case class ScalingExtF(
    val in: Operand[FloatType],
    val scale: Operand[FloatType],
    val out: Result[FloatType],
    val fastmath: Option[FastMathFlagsAttr],
) extends DerivedOperation["arith.scaling_extf", ScalingExtF]
    with NoMemoryEffect
    with SameInputOutputTensorDims derives DerivedOperationCompanion

case class ScalingTruncF(
    val in: Operand[FloatType],
    val scale: Operand[FloatType],
    val out: Result[FloatType],
    // TODO: val roundingmode: Option[RoundingModeAttr],
    val fastmath: Option[FastMathFlagsAttr],
) extends DerivedOperation["arith.scaling_truncf", ScalingTruncF]
    with NoMemoryEffect
    with SameInputOutputTensorDims derives DerivedOperationCompanion

case class SelectOp(
    val condition: Operand[BoolLike],
    val trueValue: Operand[TypeAttribute],
    val falseValue: Operand[TypeAttribute],
    val result: Result[TypeAttribute],
) extends DerivedOperation["arith.select", SelectOp]
    with NoMemoryEffect
    with BooleanConditionOrMatchingShape(condition.typ, result.typ)
    with AllTypesMatch(trueValue.typ, falseValue.typ, result.typ)
    derives DerivedOperationCompanion

case class ShLI(
    val lhs: Operand[AnyIntegerType],
    val rhs: Operand[AnyIntegerType],
    val result: Result[AnyIntegerType],
    // TODO: val overflowFlags: Option[IntegerOverflowFlags] = None
) extends DerivedOperation["arith.shli", ShLI]
    with SameOperandsAndResultTypes derives DerivedOperationCompanion

case class ShRSI(
    val lhs: Operand[AnyIntegerType],
    val rhs: Operand[AnyIntegerType],
    val result: Result[AnyIntegerType],
) extends DerivedOperation["arith.shrsi", ShRSI]
    with NoMemoryEffect
    with SameOperandsAndResultTypes derives DerivedOperationCompanion

case class ShRUI(
    val lhs: Operand[AnyIntegerType],
    val rhs: Operand[AnyIntegerType],
    val result: Result[AnyIntegerType],
) extends DerivedOperation["arith.shrui", ShRUI]
    with NoMemoryEffect
    with SameOperandsAndResultTypes derives DerivedOperationCompanion

case class SIToFP(
    in: Operand[SignlessFixedWidthIntegerLike],
    out: Result[FloatType],
) extends DerivedOperation["arith.sitofp", SIToFP]
    with NoMemoryEffect
    with SameOperandsAndResultShape
    with SameInputOutputTensorDims derives DerivedOperationCompanion

case class SubF(
    val lhs: Operand[FloatType],
    val rhs: Operand[FloatType],
    val result: Result[FloatType],
    val fastmath: Option[FastMathFlagsAttr] = None,
) extends DerivedOperation["arith.subf", SubF]
    with NoMemoryEffect
    with SameOperandsAndResultTypes derives DerivedOperationCompanion

case class SubI(
    lhs: Operand[AnyIntegerType],
    rhs: Operand[AnyIntegerType],
    result: Result[AnyIntegerType],
    // TODO: val overflowFlags: Option[IntegerOverflowFlags] = None
) extends DerivedOperation["arith.subi", SubI]
    with NoMemoryEffect
    with SameOperandsAndResultTypes derives DerivedOperationCompanion

case class TruncF(
    val in: Operand[FloatType],
    val out: Result[FloatType],
    // TODO: val roundingmode: Option[RoundingModeAttr],
    val fastmath: Option[FastMathFlagsAttr],
) extends DerivedOperation["arith.truncf", TruncF]
    with SameOperandsAndResultShape
    with NoMemoryEffect
    with SameInputOutputTensorDims derives DerivedOperationCompanion

case class TruncI(
    val in: Operand[SignlessFixedWidthIntegerLike],
    val out: Result[SignlessFixedWidthIntegerLike],
    // TODO: val overflowFlags: Option[IntegerOverflowFlags] = None
) extends DerivedOperation["arith.trunci", TruncI]
    with NoMemoryEffect
    with SameOperandsAndResultTypes derives DerivedOperationCompanion

case class UIToFP(
    val in: Operand[SignlessFixedWidthIntegerLike],
    val out: Result[FloatType],
) extends DerivedOperation["arith.uitofp", UIToFP]
    with SameOperandsAndResultShape
    with NoMemoryEffect
    with SameInputOutputTensorDims derives DerivedOperationCompanion

case class XOrI(
    lhs: Operand[AnyIntegerType],
    rhs: Operand[AnyIntegerType],
    result: Result[AnyIntegerType],
) extends DerivedOperation["arith.xori", XOrI]
    with NoMemoryEffect
    with SameOperandsAndResultTypes
    with Commutative derives DerivedOperationCompanion

val ArithDialect =
  summonDialect[
    Tuple1[FastMathFlagsAttr],
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
        IndexCast,
    ),
  ]
