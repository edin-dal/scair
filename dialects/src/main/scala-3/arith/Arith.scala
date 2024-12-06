package scair.dialects.arithgen

import fastparse._
import scala.collection.immutable._
import scala.collection.BitSetOps
import scair.dialects.affine.floordiv
import scair.ir.AttributeObject
import scair.clair.mirrored.Operand
import scair.dialects.builtin.FloatType
import scair.clair.mirrored.OperationFE
import scair.clair.mirrored.AttributeFE
import scair.clair.mirrored.DialectFE
import scair.clair.mirrored.Result
import scair.clair.mirrored.Property
import scair.scairdl.irdef.ScaIRDLDialect
import scair.clair.mirrored.summonDialect
import scair.scairdl.irdef.AttrEscapeHatch
import scair.AttrParser
import scair.ir.Attribute
import fastparse.ParsingRun
import scair.scairdl.irdef.DialectDef.empty
import scair.dialects.builtin.IntegerType
import scair.dialects.builtin.IndexType
import scair.clair.mirrored.AnyAttribute

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

enum Arith extends DialectFE:
  case AddfOp(
      lhs: Operand[FloatType],
      rhs: Operand[FloatType],
      res: Result[FloatType],
      flags: Property[FastMathFlagsAttr]
  ) extends Arith with OperationFE
  case MulfOp(
      lhs: Operand[FloatType],
      rhs: Operand[FloatType],
      res: Result[FloatType],
      flags: Property[FastMathFlagsAttr]
  ) extends Arith with OperationFE
  case DivfOp(
      lhs: Operand[FloatType],
      rhs: Operand[FloatType],
      res: Result[FloatType],
      flags: Property[FastMathFlagsAttr]
  ) extends Arith with OperationFE
  // TODO Apparently there's a new overflow flag here, overlooking for now.
  case AddiOp(
      lhs: Operand[AnyIntegerType],
      rhs: Operand[AnyIntegerType],
      res: Result[AnyIntegerType]
  ) extends Arith with OperationFE
  case SubiOp(
      lhs: Operand[AnyIntegerType],
      rhs: Operand[AnyIntegerType],
      res: Result[AnyIntegerType]
  ) extends Arith with OperationFE
  case MuliOp(
      lhs: Operand[AnyIntegerType],
      rhs: Operand[AnyIntegerType],
      res: Result[AnyIntegerType]
  ) extends Arith with OperationFE
  case DivuiOp(
      lhs: Operand[AnyIntegerType],
      rhs: Operand[AnyIntegerType],
      res: Result[AnyIntegerType]
  ) extends Arith with OperationFE
  case DivsiOp(
      lhs: Operand[AnyIntegerType],
      rhs: Operand[AnyIntegerType],
      res: Result[AnyIntegerType]
  ) extends Arith with OperationFE
  case RemuiOp(
      lhs: Operand[AnyIntegerType],
      rhs: Operand[AnyIntegerType],
      res: Result[AnyIntegerType]
  ) extends Arith with OperationFE
  case RemsiOp(
      lhs: Operand[AnyIntegerType],
      rhs: Operand[AnyIntegerType],
      res: Result[AnyIntegerType]
  ) extends Arith with OperationFE
  case CmpiOp(
      lhs: Operand[AnyIntegerType],
      rhs: Operand[AnyIntegerType],
      res: Result[I1],
      predicate: Property[IntegerPredicate]
  ) extends Arith with OperationFE
  case AndiOp(
      lhs: Operand[AnyIntegerType],
      rhs: Operand[AnyIntegerType],
      res: Result[I1]
  ) extends Arith with OperationFE
  case OriOp(
      lhs: Operand[AnyIntegerType],
      rhs: Operand[AnyIntegerType],
      res: Result[I1]
  ) extends Arith with OperationFE
  case SitofpOp(
      in: Operand[AnyIntegerType],
      res: Result[FloatType]
  ) extends Arith with OperationFE
  case IndexCastOp(
      in: Operand[IndexType.type],
      res: Result[IndexType.type]
  ) extends Arith with OperationFE

object ArithGen
    extends ScaIRDLDialect(
      summonDialect[Arith](
        Seq(),
        Seq(new AttrEscapeHatch[FastMathFlagsAttr])
      )
    )
