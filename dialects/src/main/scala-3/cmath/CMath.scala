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

enum CMath extends DialectFE:

  case Complex(
      e1: Operand[FloatType]
  ) extends CMath with AttributeFE

  case Norm(
      in: Operand[AnyAttribute],
      res: Result[FloatType]
  ) extends CMath with OperationFE

  case Mul[Operation](
      lhs: Operand[AnyAttribute],
      rhs: Operand[AnyAttribute],
      res: Result[AnyAttribute]
  ) extends CMath with OperationFE

object CMathGen
    extends ScaIRDLDialect(
      summonDialect[CMath](
        Seq(),
        Seq(new AttrEscapeHatch[SampleData])
      )
    )
