package scair.dialects.llvm

import scair.clair.codegen.*
import scair.clair.macros.*
import scair.dialects.builtin.*
import scair.ir.*

object Ptr extends AttributeObject {
  override def name: String = "llvm.ptr"
  override def factory = Ptr.apply
}

case class Ptr(
    val typ: Seq[Attribute]
) extends ParametrizedAttribute(
      name = "llvm.ptr",
      parameters = Seq(typ)
    )
    with TypeAttribute

case class Load(
    ptr: Operand[Ptr],
    result: Result[Attribute]
) extends MLIRName["llvm.load"]
    derives MLIRTrait

case class GetElementPtr(
    base: Operand[Ptr],
    dynamicIndices: Variadic[Operand[IntegerType]],
    res: Result[Ptr],
    rawConstantIndices: Property[DenseArrayAttr],
    elem_type: Property[Attribute]
) extends MLIRName["llvm.getelementptr"]
    derives MLIRTrait

val LLVMDialect = summonDialect[(Load, GetElementPtr)](Seq(Ptr))
