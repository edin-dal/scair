package scair.dialects.llvm

import scair.clair.codegen.*
import scair.clair.macros.*
import scair.dialects.builtin.*
import scair.ir.*

case class Ptr()
    extends ParametrizedAttribute(
      name = "llvm.ptr",
      parameters = Seq()
    )
    with TypeAttribute
    with MLIRName["llvm.ptr"] derives AttributeTrait

case class Load(
    ptr: Operand[Ptr],
    result: Result[Attribute]
) extends MLIRName["llvm.load"]
    derives MLIRTrait

case class GetElementPtr(
    base: Operand[Ptr],
    dynamicIndices: Seq[Operand[IntegerType]],
    res: Result[Ptr],
    rawConstantIndices: DenseArrayAttr,
    elem_type: Attribute
) extends MLIRName["llvm.getelementptr"]
    derives MLIRTrait

val LLVMDialect = summonDialect[Tuple1[Ptr], (Load, GetElementPtr)](Seq())
