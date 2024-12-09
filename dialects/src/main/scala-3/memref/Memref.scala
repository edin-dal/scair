package scair.dialects.memrefgen

import scair.scairdl.constraints._
import scair.clair.mirrored._
import scair.scairdl.irdef._
import scair.ir.Attribute
import scair.dialects.builtin.{MemrefType, IndexType}

case class AllocOp(
    dynamicSizes: Operand[IndexType.type],
    symbolOperands: Operand[IndexType.type],
    memref: Result[MemrefType]
) extends OperationFE

case class DeallocOp(
    memref: Operand[MemrefType]
) extends OperationFE

object MemrefGen
    extends ScaIRDLDialect(summonDialect[(AllocOp, DeallocOp)]("Memref"))
