package scair.dialects.cmath

import scair.clair.codegen.*
import scair.clair.macros.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.clair.constraint.*

case class Complex(
    val typ: FloatType | IndexType.type
) extends ParametrizedAttribute(
      name = "cmath.complex",
      parameters = Seq(typ)
    )
    with TypeAttribute
    with MLIRName["cmath.complex"]
    derives AttributeTrait

val f16 = Float16Type

case class Norm(
    in: Operand[Complex Constrained EqualAttr[f16.type]],
    res: Result[FloatType]
) extends DerivedOperation["cmath.norm", Norm]
    derives DerivedOperationCompanion

case class Mul(
    lhs: Operand[Complex],
    rhs: Operand[Complex],
    res: Result[Complex]
) extends DerivedOperation["cmath.mul", Mul]
    derives DerivedOperationCompanion

val CMathDialect = summonDialect[Tuple1[Complex], (Norm, Mul)](Seq())
