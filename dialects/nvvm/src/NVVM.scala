package scair.dialects.nvvm

import scair.clair.*
import scair.ir.*

// ███╗░░██╗ ██╗░░░██╗ ██╗░░░██╗ ███╗░░░███╗
// ████╗░██║ ██║░░░██║ ██║░░░██║ ████╗░████║
// ██╔██╗██║ ╚██╗░██╔╝ ╚██╗░██╔╝ ██╔████╔██║
// ██║╚████║ ░╚████╔╝░ ░╚████╔╝░ ██║╚██╔╝██║
// ██║░╚███║ ░░╚██╔╝░░ ░░╚██╔╝░░ ██║░╚═╝░██║
// ╚═╝░░╚══╝ ░░░╚═╝░░░ ░░░╚═╝░░░ ╚═╝░░░░░╚═╝

enum NVVMMemorySpace(spelling: String)
    extends EnumAttr("nvvm.memory_space", spelling):
  case Generic extends NVVMMemorySpace("generic")
  case Global extends NVVMMemorySpace("global")
  case Shared extends NVVMMemorySpace("shared")
  case Constant extends NVVMMemorySpace("constant")
  case Local extends NVVMMemorySpace("local")
  case Tensor extends NVVMMemorySpace("tensor")
  case SharedCluster extends NVVMMemorySpace("shared_cluster")

val NVVMDialect = summonDialect[
  Tuple1[NVVMMemorySpace],
  EmptyTuple,
]
