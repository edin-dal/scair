package scair.dialects.cmathv2

import scair.clairV2.codegen.*
import scair.clairV2.mirrored.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.clairV2.macros._

case class MulV2(
    lhs: Operand[IntegerType],
    rhs: Operand[IntegerType],
    result: Result[IntegerType],
    randProp: Property[StringData]
) derives MLIRTrait

case class MulSingleVariadic(
    lhs: Operand[IntegerType],
    rhs: Variadic[Operand[IntegerType]],
    result: Variadic[Result[IntegerType]],
    randProp: Property[StringData]
) derives MLIRTrait

case class NormV2(
    norm: Operand[IntegerType],
    result: Result[IntegerType]
) derives MLIRTrait

val CMathV2Dialect = summonDialect[(MulV2, NormV2)]
