package scair.dialects.cmath

import scair.clair.codegen.*
import scair.clair.macros.*
import scair.dialects.builtin.*
import scair.ir.*

case class Complex(
    val typ: FloatType | IndexType.type
) extends DerivedAttribute["cmath.complex", Complex]
    with TypeAttribute
    derives DerivedAttributeCompanion

case class Norm(
    in: Operand[Complex],
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
