package scair.dialects.memrefgen

import scair.clair.mirrored.*
import scair.dialects.builtin.IndexType
import scair.dialects.builtin.MemrefType
import scair.ir.Attribute
import scair.scairdl.constraints.*
import scair.scairdl.irdef.*
import scair.dialects.builtin.IntegerAttr

case class Alloc(
    dynamicSizes: Variadic[Operand[IndexType.type]],
    symbolOperands: Variadic[Operand[IndexType.type]],
    memref: Result[MemrefType],
    alignment: Property[IntegerAttr]
) extends OperationFE

case class Dealloc(
    memref: Operand[MemrefType]
) extends OperationFE
case class Load(
    memref: Operand[MemrefType],
    indices: Variadic[Operand[IndexType.type]],
    result: Result[MemrefType]
) extends OperationFE
case class Store(
    value: Operand[MemrefType],
    memref: Operand[MemrefType],
    indices: Variadic[Operand[IndexType.type]]
) extends OperationFE

object MemrefGen
    extends ScaIRDLDialect(
      summonDialect[(Alloc, Dealloc, Load, Store)]("Memref")
    )
