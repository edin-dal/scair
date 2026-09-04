package scair.dialects.linalg

import scair.clair.*
import scair.dialects.builtin.*
import scair.ir.*

// ██╗░░░░░ ██╗ ███╗░░██╗ ░█████╗░ ██╗░░░░░ ░██████╗░
// ██║░░░░░ ██║ ████╗░██║ ██╔══██╗ ██║░░░░░ ██╔════╝░
// ██║░░░░░ ██║ ██╔██╗██║ ███████║ ██║░░░░░ ██║░░██╗░
// ██║░░░░░ ██║ ██║╚████║ ██╔══██║ ██║░░░░░ ██║░░╚██╗
// ███████╗ ██║ ██║░╚███║ ██║░░██║ ███████╗ ╚██████╔╝
// ╚══════╝ ╚═╝ ╚═╝░░╚══╝ ╚═╝░░╚═╝ ╚══════╝ ░╚═════╝░
//
// ░█████╗░ ██████╗░ ███████╗ ██████╗░ ░█████╗░ ████████╗ ██╗ ░█████╗░ ███╗░░██╗ ░██████╗
// ██╔══██╗ ██╔══██╗ ██╔════╝ ██╔══██╗ ██╔══██╗ ╚══██╔══╝ ██║ ██╔══██╗ ████╗░██║ ██╔════╝
// ██║░░██║ ██████╔╝ █████╗░░ ██████╔╝ ███████║ ░░░██║░░░ ██║ ██║░░██║ ██╔██╗██║ ╚█████╗░
// ██║░░██║ ██╔═══╝░ ██╔══╝░░ ██╔══██╗ ██╔══██║ ░░░██║░░░ ██║ ██║░░██║ ██║╚████║ ░╚═══██╗
// ╚█████╔╝ ██║░░░░░ ███████╗ ██║░░██║ ██║░░██║ ░░░██║░░░ ██║ ╚█████╔╝ ██║░╚███║ ██████╔╝
// ░╚════╝░ ╚═╝░░░░░ ╚══════╝ ╚═╝░░╚═╝ ╚═╝░░╚═╝ ░░░╚═╝░░░ ╚═╝ ░╚════╝░ ╚═╝░░╚══╝ ╚═════╝░

trait Pure extends Operation with NoMemoryEffect

/* Ports the Linalg ops that do not correspond to library calls, i.e. those
 * defined in MLIR's LinalgOps.td. */

/*≡==--=≡≡≡≡=--=≡≡*\
||    YIELD OP    ||
\*≡==---=≡≡=---==≡*/

case class Yield(
    values: Seq[Operand[Attribute]] = Seq.empty
) extends DerivedOperation["linalg.yield"]
    with IsTerminator
    with ReturnLike
    with Pure derives OpDefs

/*≡==--=≡≡≡≡=--=≡≡*\
||    INDEX OP    ||
\*≡==---=≡≡=---==≡*/

// TODO: `dim` is a `ConfinedAttr<I64Attr, [IntMinValue<0>]>` upstream. ScaIR has
// no confined attributes yet, although it is technically possible to implement them,
// we are actively working on a more general, type level attribute validation framework that will allow for this,
// the constraint is dropped for now; it would
// belong in a `customVerify` once the rest of the dialect is in place.
//
// TODO: upstream's assembly format is `$dim attr-dict `:` type($result)`, i.e.
// `linalg.index 0 : index`. ScaIR prints that correctly but cannot parse it
// back: its attribute parser reads `0 : index` greedily as a typed
// `IntegerAttr`, leaving no `:` for the result type. The op therefore uses the
// generic form until the parser can be told to read an untyped integer here.
case class Index(
    result: Result[IndexType],
    dim: IntegerAttr,
) extends DerivedOperation["linalg.index"]
    with Pure derives OpDefs

/*≡==---=≡≡≡≡≡=---=≡≡*\
||     SOFTMAX OP    ||
\*≡==----=≡≡≡=----==≡*/

case class Softmax(
    input: Operand[ShapedType],
    output: Operand[ShapedType],
    result: Seq[Result[RankedTensorType]] = Seq.empty,
    dimension: IntegerAttr,
) extends DerivedOperation["linalg.softmax"]
    with DestinationStyleOpInterface
    with ReifyRankedShapedTypeOpInterface
    with AggregatedOpInterface
    with MemoryEffectsOpInterface
    with TilingInterface derives OpDefs

/*≡==---=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=---=≡≡*\
||  WINOGRAD FILTER TRANSFORM  ||
\*≡==----=≡≡≡≡≡≡≡≡≡≡≡≡≡=----==≡*/

// TODO: upstream restricts `filter`, `output` and `result` to rank-4 tensors
// (`TensorRankOf<[AnyType], [4]>`). ScaIR cannot express a rank constraint in
// the type yet, but we are working on a more general framework to allow for this.
// we have some early prototypes for this based in Phantom types, but the mechanism for them is
// not type-level, they are simply used to express that information in the type system,
// before being erased in lieu of derived runtime verifiers.
// They are just plain `RankedTensorType`s here.
case class WinogradFilterTransform(
    filter: Operand[RankedTensorType],
    output: Operand[RankedTensorType],
    result: Result[RankedTensorType],
    fmr: WinogradConv2DFmr,
) extends DerivedOperation["linalg.winograd_filter_transform"]
    with AllElementTypesMatch(filter.typ, output.typ)
    with DestinationStyleOpInterface
    with TilingInterface derives OpDefs

/*≡==---=≡≡≡≡≡≡≡≡≡≡≡≡≡=---=≡≡*\
||  WINOGRAD INPUT TRANSFORM ||
\*≡==----=≡≡≡≡≡≡≡≡≡≡≡=----==≡*/

// TODO: as above; upstream `input` is rank-4 and `output`/`result` are rank-6.
case class WinogradInputTransform(
    input: Operand[RankedTensorType],
    output: Operand[RankedTensorType],
    result: Result[RankedTensorType],
    fmr: WinogradConv2DFmr,
) extends DerivedOperation["linalg.winograd_input_transform"]
    with AllElementTypesMatch(input.typ, output.typ)
    with DestinationStyleOpInterface
    with TilingInterface derives OpDefs

/*≡==---=≡≡≡≡≡≡≡≡≡≡≡≡≡≡=---=≡≡*\
||  WINOGRAD OUTPUT TRANSFORM ||
\*≡==----=≡≡≡≡≡≡≡≡≡≡≡≡=----==≡*/

// TODO: as above; upstream `value` is rank-6 and `output`/`result` are rank-4.
case class WinogradOutputTransform(
    value: Operand[RankedTensorType],
    output: Operand[RankedTensorType],
    result: Result[RankedTensorType],
    fmr: WinogradConv2DFmr,
) extends DerivedOperation["linalg.winograd_output_transform"]
    with AllElementTypesMatch(value.typ, output.typ)
    with DestinationStyleOpInterface
    with TilingInterface derives OpDefs
