package scair.dialects.cmath

import scair.clair.codegen.*
import scair.clair.macros.*
import scair.dialects.builtin.*
import scair.ir.*

case class Complex(
    val typ: FloatType | IndexType
) extends DerivedAttribute["cmath.complex", Complex]
    with TypeAttribute

case class Norm(
    in: Operand[Complex],
    res: Result[FloatType]
) extends DerivedOperation["cmath.norm", Norm]
    with NoMemoryEffect

case class Mul(
    lhs: Operand[Complex],
    rhs: Operand[Complex],
    res: Result[Complex]
) extends DerivedOperation["cmath.mul", Mul]
    with NoMemoryEffect

val CMathDialect = summonDialect[Tuple1[Complex], (Norm, Mul)](Seq())
