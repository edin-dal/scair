package scair.dialects.cmathv2

import scair.clairV2.codegen.*
import scair.clairV2.mirrored.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.clairV2.macros._

object ComplexV2 extends AttributeObject {
  override def name: String = "cmathv2.complex"
}

case class ComplexV2(
    val typ: Seq[Attribute]
) extends ParametrizedAttribute(
      name = "cmathV2.complex",
      parameters = Seq(typ)
    )
    with TypeAttribute

case class MulV2(
    lhs: Operand[IntegerType],
    rhs: Operand[IntegerType],
    result: Result[IntegerType],
    randProp: Property[StringData]
) extends MLIRName["cmathv2.mulv2"]
    derives MLIRTrait

case class NormV2(
    norm: Operand[ComplexV2],
    result: Result[IntegerType]
) extends MLIRName["cmathv2.normv2"]
    derives MLIRTrait

val CMathV2Dialect = summonDialect[
  (MulV2, NormV2)
](Seq(ComplexV2))
