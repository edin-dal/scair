package scair.dialects.llvm

import scair.dialects.builtin.DenseArrayAttr
import scair.dialects.builtin.IntegerType
import scair.clair.codegen.*
import scair.clair.mirrored.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.clair.macros._
import scair.clair.testAttrMacros.*

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
    dynamicIndices: Variadic[Operand[IntegerType]],
    res: Result[Ptr],
    rawConstantIndices: Property[DenseArrayAttr],
    elem_type: Property[Attribute]
) extends MLIRName["llvm.getelementptr"]
    derives MLIRTrait

val LLVMDialect = summonDialect[Tuple1[Ptr], (Load, GetElementPtr)](Seq())
