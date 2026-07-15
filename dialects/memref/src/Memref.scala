package scair.dialects.memref

import scair.clair.*
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
    dynamicSizes: Seq[Operand[IndexType]] = Seq.empty,
    symbolOperands: Seq[Operand[IndexType]] = Seq.empty,
    memref: Result[MemrefType],
    alignment: Option[IntegerAttr] = None,
) extends DerivedOperation["memref.alloc"] derives OpDefs

case class Dealloc(
    memref: Operand[MemrefType]
) extends DerivedOperation["memref.dealloc"]
    with AssemblyFormat["$memref attr-dict `:` type($memref)"] derives OpDefs

case class Dim(
    memref: Operand[MemrefType],
    index: Operand[IndexType],
    result: Result[IndexType],
) extends DerivedOperation["memref.dim"]
    with NoMemoryEffect derives OpDefs

case class Load(
    memref: Operand[MemrefType],
    indices: Seq[Operand[IndexType]] = Seq.empty,
    result: Result[Attribute],
) extends DerivedOperation["memref.load"] derives OpDefs

case class Store(
    value: Operand[Attribute],
    memref: Operand[MemrefType],
    indices: Seq[Operand[IndexType]] = Seq.empty,
) extends DerivedOperation["memref.store"] derives OpDefs

val MemrefDialect =
  summonDialect[EmptyTuple, (Alloc, Dealloc, Load, Store, Dim)]
