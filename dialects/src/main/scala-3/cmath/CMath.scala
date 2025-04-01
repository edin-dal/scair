package scair.dialects.cmath

import scair.clair.codegen.*
import scair.clair.mirrored.*
import scair.clair.macros._
import scair.dialects.builtin.*
import scair.ir.*
import scair.clair.testAttrMacros.*

case class Complex(
    val typ: FloatType | IndexType.type
) extends ParametrizedAttribute(
      name = "cmath.complex",
      parameters = Seq(typ)
    )
    with TypeAttribute
    with MLIRName["cmath.complex"]
    derives AttributeTrait

case class Norm(
    in: Operand[Complex],
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
