package scair.dialects.linalg

import scair.clair.*
import scair.dialects.builtin.*
import scair.ir.*

// ██████╗░ ███████╗ ██╗░░░░░ ░█████╗░ ██╗░░╚██╗ ░█████╗░ ██╗░░░██╗ ████████╗
// ██╔══██╗ ██╔════╝ ██║░░░░░ ██╔══██╗ ╚██╗░██╔╝ ██╔══██╗ ██║░░░██║ ╚══██╔══╝
// ██████╔╝ █████╗░░ ██║░░░░░ ███████║ ░╚████╔╝░ ██║░░██║ ██║░░░██║ ░░░██║░░░
// ██╔══██╗ ██╔══╝░░ ██║░░░░░ ██╔══██║ ░░╚██╔╝░░ ██║░░██║ ██║░░░██║ ░░░██║░░░
// ██║░░██║ ███████╗ ███████╗ ██║░░██║ ░░░██║░░░ ╚█████╔╝ ╚██████╔╝ ░░░██║░░░
// ╚═╝░░╚═╝ ╚══════╝ ╚══════╝ ╚═╝░░╚═╝ ░░░╚═╝░░░ ░╚════╝░ ░╚═════╝░ ░░░╚═╝░░░

/* Ports the data-layout ops defined in MLIR's LinalgRelayoutOps.td. */

trait TypesMatchWith(one: RankedTensorType, two: RankedTensorType)
    extends Operation

abstract class LinalgRelayoutOp(
    val _dest: RankedTensorType,
    val _result: RankedTensorType,
) extends Operation
    with OpAsmOpInterface
    with DestinationStyleOpInterface
    with LinalgRelayoutOpInterface
    with ConditionallySpeculatable
    with NoMemoryEffect
    with ReifyRankedShapedTypeOpInterface
    with TypesMatchWith(_dest, _result):

  def getSpeculatability: Speculatability =
    Speculatability.RecursivelySpeculatable

/*≡==--=≡≡≡≡=--=≡≡*\
||    PACK OP     ||
\*≡==---=≡≡=---==≡*/

// TODO: `outer_dims_perm` is a `DefaultValuedOptionalAttr<DenseI64ArrayAttr,
// "{}">` upstream; modeled as a plain optional here, an absent attribute
// standing for the empty default.
case class Pack(
    source: Operand[RankedTensorType],
    dest: Operand[RankedTensorType],
    padding_value: Option[Operand[Attribute]] = None,
    outer_dims_perm: Option[DenseArrayAttr] = None,
    inner_dims_pos: DenseArrayAttr,
    inner_tiles: Seq[Operand[IndexType]] = Seq.empty,
    static_inner_tiles: DenseArrayAttr,
    result: Result[RankedTensorType],
) extends LinalgRelayoutOp(dest.typ, result.typ)
    with DerivedOperation["linalg.pack"] derives OpDefs

/*≡==--=≡≡≡≡≡≡=--=≡≡*\
||    UNPACK OP     ||
\*≡==---=≡≡≡≡=---==≡*/

case class UnPack(
    source: Operand[RankedTensorType],
    dest: Operand[RankedTensorType],
    outer_dims_perm: Option[DenseArrayAttr] = None,
    inner_dims_pos: DenseArrayAttr,
    inner_tiles: Seq[Operand[IndexType]] = Seq.empty,
    static_inner_tiles: DenseArrayAttr,
    result: Result[RankedTensorType],
) extends LinalgRelayoutOp(dest.typ, result.typ)
    with DerivedOperation["linalg.unpack"] derives OpDefs
