package scair.dialects.cmath

import scair.clair.codegen.*
import scair.clair.macros.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.clair.constraint._

case class Complex(
    val typ: FloatType | IndexType.type
) extends ParametrizedAttribute(
      name = "cmath.complex",
      parameters = Seq(typ)
    )
    with TypeAttribute
    with MLIRName["cmath.complex"]
    derives AttributeTrait

object cstr extends BaseAttr[Complex]
case class Norm(
    in: Operand[Constrained[cstr.type]],
    res: Result[FloatType]
) extends MLIRName["cmath.norm"]
    derives MLIRTrait

case class Mul(
    lhs: Operand[Complex],
    rhs: Operand[Complex],
    res: Result[Complex]
) extends MLIRName["cmath.mul"]
    derives MLIRTrait

val CMathDialect = summonDialect[Tuple1[Complex], (Norm, Mul)](Seq())
