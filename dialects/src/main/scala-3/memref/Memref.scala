package scair.dialects.memref

import scair.clair.codegen.*
import scair.clair.macros.*
import scair.dialects.builtin.*
import scair.ir.*

case class Alloc(
    dynamicSizes: Seq[Operand[IndexType.type]],
    symbolOperands: Seq[Operand[IndexType.type]],
    memref: Result[MemrefType],
    alignment: IntegerAttr
) extends DerivedOperation["memref.alloc", Alloc]
    derives DerivedOperationCompanion

case class Dealloc(
    memref: Operand[MemrefType]
) extends DerivedOperation["memref.dealloc", Dealloc]
    derives DerivedOperationCompanion

case class Load(
    memref: Operand[MemrefType],
    indices: Seq[Operand[IndexType.type]],
    result: Result[Attribute]
) extends DerivedOperation["memref.load", Load]
    derives DerivedOperationCompanion

case class Store(
    value: Operand[Attribute],
    memref: Operand[MemrefType],
    indices: Seq[Operand[IndexType.type]]
) extends DerivedOperation["memref.store", Store]
    derives DerivedOperationCompanion

val MemrefDialect =
  summonDialect[EmptyTuple, (Alloc, Dealloc, Load, Store)](Seq())
