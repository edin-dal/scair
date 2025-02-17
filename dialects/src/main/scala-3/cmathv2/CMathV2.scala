package scair.dialects.cmathv2

import scair.clairV2.codegen.*
import scair.clairV2.mirrored.*
import scair.dialects.builtin.*
import scair.ir.*

case class MulV2(
    lhs: Operand[IntegerAttr],
    rhs: Operand[IntegerAttr],
    result: Result[IntegerAttr],
    randProp: Property[StringData]
) extends ADTOperation

case class NormV2(
    norm: Operand[IntegerAttr],
    result: Result[IntegerAttr]
) extends ADTOperation

object CMathV2Gen extends MLIRRealmize(summonMLIROps[(MulV2, NormV2)]("CMathV2"))
