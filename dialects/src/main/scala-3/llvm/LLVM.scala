package scair.dialects.llvmgen

import scair.clair.mirrored.*
import scair.ir.Attribute
import scair.scairdl.constraints.*
import scair.scairdl.irdef.*
import scair.dialects.builtin.IntegerType
import scair.dialects.builtin.DenseArrayAttr

case class Ptr(
) extends AttributeFE
    with TypeAttributeFE
case class Load(
    ptr: Operand[Ptr],
    result: Result[Attribute]
) extends OperationFE
case class GetElementPtr(
    base: Operand[Ptr],
    dynamicIndices: Variadic[Operand[IntegerType]],
    res: Result[Ptr],
    rawConstantIndices: Property[DenseArrayAttr],
    elem_type: Property[Attribute]
) extends OperationFE

object LLVMGen
    extends ScaIRDLDialect(
      summonDialect[(GetElementPtr, Ptr, Load)]("LLVM")
    )
