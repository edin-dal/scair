package scair.dialects.nvvm

import scair.clair.*
import scair.dialects.builtin.*
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

// TODO: flags and link are optional upstream; AttrDefs has no optional parameters
case class NVVMTargetAttr(
    O: IntegerAttr = IntegerAttr(IntData(2), I32),
    triple: StringData = StringData("nvptx64-nvidia-cuda"),
    chip: StringData = StringData("sm_50"),
    features: StringData = StringData(""),
    flags: DictionaryAttr = DictionaryAttr(Map.empty),
    link: ArrayAttribute[Attribute] = ArrayAttribute(),
    verifyTarget: IntegerAttr = IntegerAttr(IntData(1), I1),
) extends DerivedAttribute["nvvm.target"] derives AttrDefs

val NVVMDialect = summonDialect[
  (NVVMMemorySpace, NVVMTargetAttr),
  EmptyTuple,
]
