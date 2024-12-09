package scair.dialects.cmathgen

import scair.scairdl.constraints._
import scair.clair.mirrored._
import scair.dialects.builtin.{FloatType}
import scair.scairdl.irdef._

enum CMath extends DialectFE:

  case Complex(
      e1: Operand[FloatType]
  ) extends CMath with TypeAttributeFE

  case Norm(
      in: Operand[Complex],
      res: Result[FloatType]
  ) extends CMath with OperationFE

  case Mul[Operation](
      lhs: Operand[Complex],
      rhs: Operand[Complex],
      res: Result[Complex]
  ) extends CMath with OperationFE

object CMathGen extends ScaIRDLDialect(summonDialect[CMath]())
