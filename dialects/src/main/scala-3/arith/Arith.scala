package scair.dialects.arithgen

import fastparse.*
import fastparse.ParsingRun
import scair.AttrParser
import scair.clair.mirrored.AnyAttribute
import scair.clair.mirrored.DialectFE
import scair.clair.mirrored.Operand
import scair.clair.mirrored.OperationFE
import scair.clair.mirrored.Property
import scair.clair.mirrored.Result
import scair.clair.mirrored.summonDialect
import scair.dialects.builtin.FloatType
import scair.dialects.builtin.IndexType
import scair.dialects.builtin.IntegerType
import scair.ir.Attribute
import scair.ir.AttributeObject
import scair.scairdl.irdef.AttrEscapeHatch
import scair.scairdl.irdef.ScaIRDLDialect

import scala.collection.immutable.*

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
//   case fast

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

  def apply(flags: FastMathFlag*): FastMathFlags = HashSet(flags: _*)

object FastMathFlagsAttr extends AttributeObject {
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
    s"arith.fastmath<$p>"

// TODO: This should be smth like IntegerType | IndexType, but is not yet supported
// in the frontend.
type AnyIntegerType = AnyAttribute
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
type IntegerPredicate = I64

case class AddfOp(
    lhs: Operand[FloatType],
    rhs: Operand[FloatType],
    res: Result[FloatType],
    flags: Property[FastMathFlagsAttr]
) extends OperationFE
case class MulfOp(
    lhs: Operand[FloatType],
    rhs: Operand[FloatType],
    res: Result[FloatType],
    flags: Property[FastMathFlagsAttr]
) extends OperationFE
case class DivfOp(
    lhs: Operand[FloatType],
    rhs: Operand[FloatType],
    res: Result[FloatType],
    flags: Property[FastMathFlagsAttr]
) extends OperationFE
// TODO Apparently there's a new overflow flag here, overlooking for now.
case class AddiOp(
    lhs: Operand[AnyIntegerType],
    rhs: Operand[AnyIntegerType],
    res: Result[AnyIntegerType]
) extends OperationFE
case class SubiOp(
    lhs: Operand[AnyIntegerType],
    rhs: Operand[AnyIntegerType],
    res: Result[AnyIntegerType]
) extends OperationFE
case class MuliOp(
    lhs: Operand[AnyIntegerType],
    rhs: Operand[AnyIntegerType],
    res: Result[AnyIntegerType]
) extends OperationFE
case class DivuiOp(
    lhs: Operand[AnyIntegerType],
    rhs: Operand[AnyIntegerType],
    res: Result[AnyIntegerType]
) extends OperationFE
case class DivsiOp(
    lhs: Operand[AnyIntegerType],
    rhs: Operand[AnyIntegerType],
    res: Result[AnyIntegerType]
) extends OperationFE
case class RemuiOp(
    lhs: Operand[AnyIntegerType],
    rhs: Operand[AnyIntegerType],
    res: Result[AnyIntegerType]
) extends OperationFE
case class RemsiOp(
    lhs: Operand[AnyIntegerType],
    rhs: Operand[AnyIntegerType],
    res: Result[AnyIntegerType]
) extends OperationFE
case class CmpiOp(
    lhs: Operand[AnyIntegerType],
    rhs: Operand[AnyIntegerType],
    res: Result[I1],
    predicate: Property[IntegerPredicate]
) extends OperationFE
case class AndiOp(
    lhs: Operand[AnyIntegerType],
    rhs: Operand[AnyIntegerType],
    res: Result[I1]
) extends OperationFE
case class OriOp(
    lhs: Operand[AnyIntegerType],
    rhs: Operand[AnyIntegerType],
    res: Result[I1]
) extends OperationFE
case class SitofpOp(
    in: Operand[AnyIntegerType],
    res: Result[FloatType]
) extends OperationFE
case class IndexCastOp(
    in: Operand[IndexType.type],
    res: Result[IndexType.type]
) extends OperationFE

object ArithGen
    extends ScaIRDLDialect(
      summonDialect[
        (
            AddfOp,
            MulfOp,
            DivfOp,
            AddiOp,
            SubiOp,
            MuliOp,
            DivuiOp,
            DivsiOp,
            RemuiOp,
            RemsiOp,
            CmpiOp,
            AndiOp,
            OriOp,
            SitofpOp,
            IndexCastOp
        )
      ](
        "Arith",
        Seq(),
        Seq(new AttrEscapeHatch[FastMathFlagsAttr])
      )
    )
