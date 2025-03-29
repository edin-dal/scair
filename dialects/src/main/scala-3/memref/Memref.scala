package scair.dialects.memref

import scair.dialects.builtin.IndexType
import scair.dialects.builtin.IntegerAttr
import scair.dialects.builtin.MemrefType
import scair.ir.*
import scair.clairV2.codegen.*
import scair.clairV2.mirrored.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.clairV2.macros._

case class Alloc(
    dynamicSizes: Variadic[Operand[IndexType.type]],
    symbolOperands: Variadic[Operand[IndexType.type]],
    memref: Result[MemrefType],
    alignment: IntegerAttr
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
