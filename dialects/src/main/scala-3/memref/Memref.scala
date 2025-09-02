package scair.dialects.memref

import scair.clair.codegen.*
import scair.clair.macros.*
import scair.dialects.builtin.*
import scair.ir.*

case class Alloc(
    dynamicSizes: Seq[Operand[IndexType]],
    symbolOperands: Seq[Operand[IndexType]],
    memref: Result[MemrefType],
    alignment: IntegerAttr
) extends DerivedOperation["memref.alloc", Alloc]

case class Dealloc(
    memref: Operand[MemrefType]
) extends DerivedOperation["memref.dealloc", Dealloc]
    with AssemblyFormat["$memref attr-dict `:` type($memref)"]

case class Dim(
    memref: Operand[MemrefType],
    index: Operand[IndexType],
    result: Result[IndexType]
) extends DerivedOperation["memref.dim", Dim]
    with NoMemoryEffect

case class Load(
    memref: Operand[MemrefType],
    indices: Seq[Operand[IndexType]],
    result: Result[Attribute]
) extends DerivedOperation["memref.load", Load]

case class Store(
    value: Operand[Attribute],
    memref: Operand[MemrefType],
    indices: Seq[Operand[IndexType]]
) extends DerivedOperation["memref.store", Store]

val MemrefDialect =
  summonDialect[EmptyTuple, (Alloc, Dealloc, Load, Store, Dim)](Seq())
