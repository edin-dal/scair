package scair.dialects.cmathgen

import scair.clair.mirrored.*
import scair.dialects.builtin.FloatType
import scair.scairdl.constraints.*
import scair.scairdl.irdef.*

case class Complex(
    e1: Operand[FloatType]
) extends TypeAttributeFE

case class Norm(
    in: Operand[Complex],
    res: Result[FloatType]
) extends OperationFE

case class Mul(
    lhs: Operand[Complex],
    rhs: Operand[Complex],
    res: Result[Complex]
) extends OperationFE

object CMathGen
    extends ScaIRDLDialect(summonDialect[(Complex, Norm, Mul)]("CMath"))
