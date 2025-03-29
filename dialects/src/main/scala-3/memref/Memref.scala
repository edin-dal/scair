package scair.dialects.memref

import scair.clair.codegen.*
import scair.clair.macros.*
import scair.dialects.builtin.*
import scair.ir.*

case class Alloc(
    dynamicSizes: Variadic[Operand[IndexType.type]],
    symbolOperands: Variadic[Operand[IndexType.type]],
    memref: Result[MemrefType],
    alignment: Property[IntegerAttr]
) extends MLIRName["memref.alloc"]
    derives MLIRTrait

case class Dealloc(
    memref: Operand[MemrefType]
) extends MLIRName["memref.dealloc"]
    derives MLIRTrait

case class Load(
    memref: Operand[MemrefType],
    indices: Variadic[Operand[IndexType.type]],
    result: Result[Attribute]
) extends MLIRName["memref.load"]
    derives MLIRTrait

case class Store(
    value: Operand[Attribute],
    memref: Operand[MemrefType],
    indices: Variadic[Operand[IndexType.type]]
) extends MLIRName["memref.store"]
    derives MLIRTrait

val MemrefDialect =
  summonDialect[(Alloc, Dealloc, Load, Store)](Seq())
