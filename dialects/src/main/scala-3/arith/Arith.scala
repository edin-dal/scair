package scair.dialects.arithgen

import fastparse._
import scala.collection.immutable._
import scala.collection.BitSetOps
import scair.dialects.affine.floordiv
import scair.ir.AttributeObject
import scair.clair.mirrored.Operand
import scair.dialects.builtin.FloatType
import scair.clair.mirrored.DialectOperation
import scair.clair.mirrored.DialectAttribute
import scair.clair.mirrored.Result
import scair.clair.mirrored.Property
import scair.scairdl.irdef.ScaIRDLDialect
import scair.clair.mirrored.summonDialect
import scair.scairdl.irdef.AttrEscapeHatch
import scair.AttrParser
import scair.ir.Attribute
import fastparse.ParsingRun
import scair.scairdl.irdef.DialectDef.empty

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

enum ArithAttrs extends DialectAttribute:
  // TODO: Remove when we have the possibility to define none
  case FakeAttr()

enum ArithOps extends DialectOperation:
  case AddfOp(
      lhs: Operand[FloatType],
      rhs: Operand[FloatType],
      res: Result[FloatType],
      flags: Property[FastMathFlagsAttr]
  )

object ArithGen
    extends ScaIRDLDialect(
      summonDialect[ArithOps, ArithAttrs](
        Seq(),
        Seq(new AttrEscapeHatch[FastMathFlagsAttr])
      )
    )
