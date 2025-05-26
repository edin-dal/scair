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
    with MLIRName["llvm.ptr"] derives DerivedAttributeCompanion

case class Load(
    ptr: Operand[Ptr],
    result: Result[Attribute]
) extends DerivedOperation["llvm.load", Load]
    derives DerivedOperationCompanion

case class GetElementPtr(
    base: Operand[Ptr],
    dynamicIndices: Seq[Operand[IntegerType]],
    res: Result[Ptr],
    rawConstantIndices: DenseArrayAttr,
    elem_type: Attribute
) extends DerivedOperation["llvm.getelementptr", GetElementPtr]
    derives DerivedOperationCompanion

val LLVMDialect = summonDialect[Tuple1[Ptr], (Load, GetElementPtr)](Seq())
