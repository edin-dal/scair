package scair.dialects.arith

import fastparse.*
import scair.AttrParser
import scair.dialects.builtin.FloatType
import scair.dialects.builtin.IntegerAttr
import scair.dialects.builtin.IntegerType
import scala.collection.immutable.*
import scair.clair.codegen.*
import scair.clair.mirrored.*
import scair.ir.*
import scair.clair.macros._

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
    s"#arith.fastmath<$p>"

// TODO: This should be smth like IntegerType | IndexType, but is not yet supported
// in the frontend.
type AnyIntegerType = Attribute
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

case class Addf(
    lhs: Operand[FloatType],
    rhs: Operand[FloatType],
    result: Result[FloatType],
    fastmath: Property[FastMathFlagsAttr]
    // assembly_format: "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) `,` type($result)"
) extends MLIRName["arith.addf"]
    derives MLIRTrait

case class Mulf(
    lhs: Operand[FloatType],
    rhs: Operand[FloatType],
    result: Result[FloatType],
    fastmath: Property[FastMathFlagsAttr]
    // assembly_format: "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) `,` type($result)"
) extends MLIRName["arith.mulf"]
    derives MLIRTrait

// I'm not sure about the flag here
case class Divf(
    lhs: Operand[FloatType],
    rhs: Operand[FloatType],
    result: Result[FloatType],
    fastmath: Property[FastMathFlagsAttr]
    // assembly_format: "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) `,` type($result)"
) extends MLIRName["arith.divf"]
    derives MLIRTrait

// TODO Apparently there's a new overflow flag here, overlooking for now.
case class Addi(
    lhs: Operand[AnyIntegerType],
    rhs: Operand[AnyIntegerType],
    result: Result[AnyIntegerType]
    // assembly_format: "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) `,` type($result)"
) extends MLIRName["arith.addi"]
    derives MLIRTrait

case class Subi(
    lhs: Operand[AnyIntegerType],
    rhs: Operand[AnyIntegerType],
    result: Result[AnyIntegerType]
    // assembly_format: "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) `,` type($result)"
) extends MLIRName["arith.subi"]
    derives MLIRTrait

case class Muli(
    lhs: Operand[AnyIntegerType],
    rhs: Operand[AnyIntegerType],
    result: Result[AnyIntegerType]
    // assembly_format: "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) `,` type($result)"
) extends MLIRName["arith.muli"]
    derives MLIRTrait

case class Divui(
    lhs: Operand[AnyIntegerType],
    rhs: Operand[AnyIntegerType],
    result: Result[AnyIntegerType]
    // assembly_format: "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) `,` type($result)"
) extends MLIRName["arith.divui"]
    derives MLIRTrait

case class Divsi(
    lhs: Operand[AnyIntegerType],
    rhs: Operand[AnyIntegerType],
    result: Result[AnyIntegerType]
    // assembly_format: "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) `,` type($result)"
) extends MLIRName["arith.divsi"]
    derives MLIRTrait

case class Remui(
    lhs: Operand[AnyIntegerType],
    rhs: Operand[AnyIntegerType],
    result: Result[AnyIntegerType]
    // assembly_format: "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) `,` type($result)"
) extends MLIRName["arith.remui"]
    derives MLIRTrait

case class Remsi(
    lhs: Operand[AnyIntegerType],
    rhs: Operand[AnyIntegerType],
    result: Result[AnyIntegerType]
    // assembly_format: "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) `,` type($result)"
) extends MLIRName["arith.remsi"]
    derives MLIRTrait

case class Cmpi(
    lhs: Operand[AnyIntegerType],
    rhs: Operand[AnyIntegerType],
    result: Result[I1],
    predicate: Property[IntegerPredicate]
    // assembly_format: "$predicate `,` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `,` type($result)"
) extends MLIRName["arith.cmpi"]
    derives MLIRTrait

case class Andi(
    lhs: Operand[AnyIntegerType],
    rhs: Operand[AnyIntegerType],
    result: Result[I1]
    // assembly_format: "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) `,` type($result)"
) extends MLIRName["arith.andi"]
    derives MLIRTrait

case class Ori(
    lhs: Operand[AnyIntegerType],
    rhs: Operand[AnyIntegerType],
    result: Result[I1]
    // assembly_format: "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) `,` type($result)"
) extends MLIRName["arith.ori"]
    derives MLIRTrait

case class Sitofp(
    in: Operand[AnyIntegerType],
    out: Result[FloatType]
    // assembly_format: "$in `:` type($in) `to` type($out)"
) extends MLIRName["arith.sitofp"]
    derives MLIRTrait

case class Index_Cast(
    in: Operand[AnyIntegerType],
    result: Result[AnyIntegerType]
    // assembly_format: "$in `:` type($in) `to` type($out)"
) extends MLIRName["arith.index_cast"]
    derives MLIRTrait

val ArithDialect =
  summonDialect[
    (
        Addf,
        Mulf,
        Divf,
        Addi,
        Subi,
        Muli,
        Divui,
        Divsi,
        Remui,
        Remsi,
        Cmpi,
        Andi,
        Ori,
        Sitofp,
        Index_Cast
    )
  ](
    Seq(FastMathFlagsAttr)
  )
