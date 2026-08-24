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

/* Ports the data-layout ops defined in MLIR's LinalgRelayoutOps.td.
 *
 * Both ops carry two variadic-ish operand groups (`padding_value` is optional
 * and `inner_tiles` is variadic), so the generic printer emits an
 * `operandSegmentSizes` property for them.
 *
 * Upstream both use a declarative assembly format built around
 * `custom<DynamicIndexList>($inner_tiles, $static_inner_tiles)`, which
 * interleaves the dynamic operands with the static `static_inner_tiles` entries.
 * ScaIR has no custom-directive support, so these ops print and parse in the
 * generic form, which MLIR accepts. */

trait TypesMatchWith extends Operation:
  val one: RankedTensorType
  val two: RankedTensorType

trait LinalgRelayoutOp(
    val _dest: RankedTensorType,
    val _result: RankedTensorType,
) extends Operation
    with OpAsmOpInterface
    with DestinationStyleOpInterface
    with LinalgRelayoutOpInterface
    with ConditionallySpeculatable
    with NoMemoryEffect
    with ReifyRankedShapedTypeOpInterface
    with TypesMatchWith:
  val one: RankedTensorType = _dest
  val two: RankedTensorType = _result

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
) extends DerivedOperation["linalg.pack"]
    with LinalgRelayoutOp(dest.typ, result.typ) derives OpDefs

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
) extends DerivedOperation["linalg.unpack"]
    with LinalgRelayoutOp(dest.typ, result.typ) derives OpDefs
