package scair.dialects.gpu

import scair.clair.*
import scair.dialects.builtin.*
import scair.ir.*

//
// ░██████╗░ ██████╗░ ██╗░░░██╗
// ██╔════╝░ ██╔══██╗ ██║░░░██║
// ██║░░██╗░ ██████╔╝ ██║░░░██║
// ██║░░╚██╗ ██╔═══╝░ ██║░░░██║
// ╚██████╔╝ ██║░░░░░ ╚██████╔╝
// ░╚═════╝░ ╚═╝░░░░░ ░╚═════╝░
//

enum AddressSpace(spelling: String)
    extends EnumAttr("gpu.address_space", spelling):
  case Global extends AddressSpace("global")
  case Workgroup extends AddressSpace("workgroup")
  case Private extends AddressSpace("private")

case class AsyncTokenType()
    extends DerivedAttribute["gpu.async.token"]
    with TypeAttribute derives AttrDefs

case class Launch(
    asyncDependencies: Seq[Operand[AsyncTokenType]] = Seq.empty,
    gridSizeX: Operand[IndexType],
    gridSizeY: Operand[IndexType],
    gridSizeZ: Operand[IndexType],
    blockSizeX: Operand[IndexType],
    blockSizeY: Operand[IndexType],
    blockSizeZ: Operand[IndexType],
    clusterSizeX: Option[Operand[IndexType]] = None,
    clusterSizeY: Option[Operand[IndexType]] = None,
    clusterSizeZ: Option[Operand[IndexType]] = None,
    dynamicSharedMemorySize: Option[Operand[IntegerType]] = None,
    module: Option[SymbolRefAttr] = None,
    function: Option[SymbolRefAttr] = None,
    asyncToken: Option[Result[AsyncTokenType]] = None,
    body: Region,
) extends DerivedOperation["gpu.launch"]
    with RecursiveMemoryEffects derives OpDefs

case class Terminator()
    extends DerivedOperation["gpu.terminator"]
    with AssemblyFormat["attr-dict"]
    with Pure
    with IsTerminator derives OpDefs

case class Barrier()
    extends DerivedOperation["gpu.barrier"]
    with AssemblyFormat["attr-dict"] derives OpDefs

case class Alloc(
    asyncDependencies: Seq[Operand[AsyncTokenType]] = Seq.empty,
    dynamicSizes: Seq[Operand[IndexType]] = Seq.empty,
    symbolOperands: Seq[Operand[IndexType]] = Seq.empty,
    hostShared: Option[UnitAttr] = None,
    memref: Result[MemrefType],
    asyncToken: Option[Result[AsyncTokenType]] = None,
) extends DerivedOperation["gpu.alloc"] derives OpDefs

case class Dealloc(
    asyncDependencies: Seq[Operand[AsyncTokenType]] = Seq.empty,
    memref: Operand[MemrefType],
    asyncToken: Option[Result[AsyncTokenType]] = None,
) extends DerivedOperation["gpu.dealloc"] derives OpDefs

val GPUDialect = summonDialect[
  (AsyncTokenType, AddressSpace),
  (Launch, Terminator, Barrier, Alloc, Dealloc),
]
