package scair.dialects.cmathgen

import scair.scairdl.constraints._
import scair.clair.mirrored._
import scair.dialects.builtin.{FloatType}
import scair.scairdl.irdef._

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

object CMathGen extends ScaIRDLDialect(summonDialect[CMath]())
