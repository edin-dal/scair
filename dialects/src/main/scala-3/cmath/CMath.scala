package scair.dialects.cmathgen

import scair.scairdl.constraints._
import scair.clair.mirrored._
import scair.dialects.builtin.{FloatType}
import scair.scairdl.irdef._
import scair.ir.AttributeObject
import scair.ir.DataAttribute

object SampleData extends AttributeObject {
  override def name: String = "sample"
}

case class SampleData(val d: String) extends DataAttribute[String]("sample", d)

enum CMathAttrs extends DialectAttribute:

  case Complex(
      e1: Operand[FloatType]
  )
enum CMathOps extends DialectOperation:

  case Norm(
      in: Operand[AnyAttribute],
      res: Result[FloatType]
  )
  case Mul[Operation](
      lhs: Operand[AnyAttribute],
      rhs: Operand[AnyAttribute],
      res: Result[AnyAttribute]
  )

object CMathGen
    extends ScaIRDLDialect(
      summonDialect[CMathOps, CMathAttrs](
        Seq(),
        Seq(new AttrEscapeHatch[SampleData])
      )
    )
