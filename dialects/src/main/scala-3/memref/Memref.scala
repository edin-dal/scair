package scair.dialects.memref

import scair.clair.codegen.*
import scair.clair.macros.*
import scair.dialects.builtin.*
import scair.ir.*

//
// ███╗░░░███╗ ███████╗ ███╗░░░███╗ ██████╗░ ███████╗ ███████╗
// ████╗░████║ ██╔════╝ ████╗░████║ ██╔══██╗ ██╔════╝ ██╔════╝
// ██╔████╔██║ █████╗░░ ██╔████╔██║ ██████╔╝ █████╗░░ █████╗░░
// ██║╚██╔╝██║ ██╔══╝░░ ██║╚██╔╝██║ ██╔══██╗ ██╔══╝░░ ██╔══╝░░
// ██║░╚═╝░██║ ███████╗ ██║░╚═╝░██║ ██║░░██║ ███████╗ ██║░░░░░
// ╚═╝░░░░░╚═╝ ╚══════╝ ╚═╝░░░░░╚═╝ ╚═╝░░╚═╝ ╚══════╝ ╚═╝░░░░░
//

case class Alloc(
    dynamicSizes: Seq[Operand[IndexType]],
    symbolOperands: Seq[Operand[IndexType]],
    memref: Result[MemrefType],
    alignment: IntegerAttr
) extends DerivedOperation.WithCompanion["memref.alloc", Alloc]
    derives DerivedOperationCompanion

case class Dealloc(
    memref: Operand[MemrefType]
) extends DerivedOperation.WithCompanion["memref.dealloc", Dealloc]
    with AssemblyFormat["$memref attr-dict `:` type($memref)"]
    derives DerivedOperationCompanion

case class Dim(
    memref: Operand[MemrefType],
    index: Operand[IndexType],
    result: Result[IndexType]
) extends DerivedOperation.WithCompanion["memref.dim", Dim]
    with NoMemoryEffect derives DerivedOperationCompanion

case class Load(
    memref: Operand[MemrefType],
    indices: Seq[Operand[IndexType]],
    result: Result[Attribute]
) extends DerivedOperation.WithCompanion["memref.load", Load]
    derives DerivedOperationCompanion

case class Store(
    value: Operand[Attribute],
    memref: Operand[MemrefType],
    indices: Seq[Operand[IndexType]]
) extends DerivedOperation.WithCompanion["memref.store", Store]
    derives DerivedOperationCompanion

val MemrefDialect =
  summonDialect[EmptyTuple, (Alloc, Dealloc, Load, Store, Dim)](Seq())
