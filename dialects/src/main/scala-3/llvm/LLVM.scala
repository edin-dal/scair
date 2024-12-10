package scair.dialects.llvmgen

import scair.clair.mirrored.*
import scair.ir.Attribute
import scair.scairdl.constraints.*
import scair.scairdl.irdef.*
import scair.dialects.builtin.IntegerType
import scair.dialects.builtin.DenseIntOrFPElementsAttr

case class PtrType(
) extends AttributeFE
    with TypeAttributeFE
case class LoadOp(
    ptr: Operand[PtrType],
    result: Result[Attribute]
) extends OperationFE
case class GetElementPtrOp(
    base: Operand[PtrType],
    dynamicIndices: Variadic[Operand[IntegerType]],
    res: Result[PtrType],
    rawConstantIndices: Property[DenseIntOrFPElementsAttr],
    elem_type: Property[Attribute]
) extends OperationFE

object LLVMGen
    extends ScaIRDLDialect(
      summonDialect[(GetElementPtrOp, PtrType, LoadOp)]("LLVM")
    )
