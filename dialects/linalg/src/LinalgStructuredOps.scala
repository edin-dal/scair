package scair.dialects.linalg

import scair.clair.*
import scair.dialects.builtin.*
import scair.ir.*

// ░██████╗ ████████╗ ██████╗░ ██╗░░░██╗ ░█████╗░ ████████╗ ██╗░░░██╗ ██████╗░ ███████╗ ██████╗░
// ██╔════╝ ╚══██╔══╝ ██╔══██╗ ██║░░░██║ ██╔══██╗ ╚══██╔══╝ ██║░░░██║ ██╔══██╗ ██╔════╝ ██╔══██╗
// ╚█████╗░ ░░░██║░░░ ██████╔╝ ██║░░░██║ ██║░░╚═╝ ░░░██║░░░ ██║░░░██║ ██████╔╝ █████╗░░ ██║░░██║
// ░╚═══██╗ ░░░██║░░░ ██╔══██╗ ██║░░░██║ ██║░░██╗ ░░░██║░░░ ██║░░░██║ ██╔══██╗ ██╔══╝░░ ██║░░██║
// ██████╔╝ ░░░██║░░░ ██║░░██║ ╚██████╔╝ ╚█████╔╝ ░░░██║░░░ ╚██████╔╝ ██║░░██║ ███████╗ ██████╔╝
// ╚═════╝░ ░░░╚═╝░░░ ╚═╝░░╚═╝ ░╚═════╝░ ░╚════╝░ ░░░╚═╝░░░ ░╚═════╝░ ╚═╝░░╚═╝ ╚══════╝ ╚═════╝░

/* Ports the structured Linalg ops that are written by hand upstream, i.e. those
 * defined in MLIR's LinalgStructuredOps.td. The ops generated from
 * LinalgNamedStructuredOps.yaml live in LinalgNamedStructuredOps.scala. */

/*≡==--==≡≡≡≡≡==--=≡≡*\
||    BASE TRAITS    ||
\*≡==---==≡≡≡==---==≡*/

/** The trait list that MLIR's `LinalgStructuredBase_Op` class attaches to every
  * structured op, gathered here so the ports below can mirror it without
  * repeating eight `with` clauses each.
  */
trait LinalgStructuredBase
    extends Operation,
      SingleBlockImplicitYieldTerminator,
      MemoryEffectsOpInterface,
      ConditionallySpeculatable,
      RecursiveMemoryEffects,
      DestinationStyleOpInterface,
      LinalgStructuredInterface,
      ReifyRankedShapedTypeOpInterface

trait LinalgContractionOpInterface extends Operation
trait LinalgConvolutionOpInterface extends Operation
trait LinalgFillOpInterface extends Operation

/*≡==--=≡≡≡≡≡≡=--=≡≡*\
||    GENERIC OP    ||
\*≡==---=≡≡≡≡=---==≡*/

// TODO: `iterator_types` is an `IteratorTypeArrayAttr` upstream, i.e. an array
// of `#linalg.iterator_type<parallel|reduction|window>` attributes. ScaIR's enum
// support (`scair.enums`) represents an enum as its ordinal `IntegerAttr`, and
// there is no attribute parser to read the symbolic form back, so the array is
// modeled here as an array of `IntegerAttr` ordinals into [[IteratorType]].
// Switch to `ArrayAttribute[IteratorType]` once ScaIR grows dialect-owned enum
// attributes.
case class Generic(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    indexing_maps: ArrayAttribute[AffineMapAttr],
    iterator_types: ArrayAttribute[IntegerAttr],
    doc: Option[StringData] = None,
    library_call: Option[StringData] = None,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    region: Region,
) extends DerivedOperation["linalg.generic"]
    with LinalgStructuredBase
    with OpAsmOpInterface derives OpDefs

/*≡==--=≡≡≡=--=≡≡*\
||    MAP OP     ||
\*≡==---=≡=---==≡*/

case class Map(
    inputs: Seq[Operand[ShapedType]] = Seq.empty,
    init: Operand[ShapedType],
    result: Seq[Result[TensorType]] = Seq.empty,
    mapper: Region,
) extends DerivedOperation["linalg.map"]
    with LinalgStructuredBase
    with OpAsmOpInterface derives OpDefs

/*≡==--=≡≡≡≡≡=--=≡≡*\
||    REDUCE OP    ||
\*≡==---=≡≡≡=---==≡*/

// TODO: upstream `dimensions` is confined to a strictly sorted
// `DenseI64ArrayAttr`; ScaIR has no confined attributes, so the sortedness
// constraint is dropped.
//
// TODO: upstream this op carries `SameVariadicOperandSize` and so has no
// `operandSegmentSizes` property. ScaIR emits and requires that property for
// any op with two variadic operand groups, so `linalg.reduce` printed by ScaIR
// carries one extra property compared to MLIR.
case class Reduce(
    inputs: Seq[Operand[ShapedType]] = Seq.empty,
    inits: Seq[Operand[ShapedType]] = Seq.empty,
    result: Seq[Result[TensorType]] = Seq.empty,
    dimensions: DenseArrayAttr,
    combiner: Region,
) extends DerivedOperation["linalg.reduce"]
    with LinalgStructuredBase
    with OpAsmOpInterface derives OpDefs

/*≡==--=≡≡≡≡≡≡≡=--=≡≡*\
||    TRANSPOSE OP   ||
\*≡==---=≡≡≡≡≡=---==≡*/

case class Transpose(
    input: Operand[ShapedType],
    init: Operand[ShapedType],
    result: Seq[Result[TensorType]] = Seq.empty,
    permutation: DenseArrayAttr,
    region: Region,
) extends DerivedOperation["linalg.transpose"]
    with LinalgStructuredBase
    with OpAsmOpInterface derives OpDefs

/*≡==--=≡≡≡≡≡≡≡=--=≡≡*\
||    BROADCAST OP   ||
\*≡==---=≡≡≡≡≡=---==≡*/

case class Broadcast(
    input: Operand[ShapedType],
    init: Operand[ShapedType],
    result: Seq[Result[TensorType]] = Seq.empty,
    dimensions: DenseArrayAttr,
    region: Region,
) extends DerivedOperation["linalg.broadcast"]
    with LinalgStructuredBase
    with OpAsmOpInterface derives OpDefs

/*≡==--=≡≡≡≡≡≡≡≡=--=≡≡*\
||   ELEMENTWISE OP   ||
\*≡==---=≡≡≡≡≡≡=---==≡*/

case class Elementwise(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    kind: ElementwiseKind,
    indexing_maps: Option[ArrayAttribute[AffineMapAttr]] = None,
    region: Region,
) extends DerivedOperation["linalg.elementwise"]
    with LinalgStructuredBase derives OpDefs

/*≡==--=≡≡≡≡≡≡=--=≡≡*\
||    MATMUL OP     ||
\*≡==---=≡≡≡≡=---==≡*/

// Note: upstream `indexing_maps` and `cast` are `DefaultValuedOptionalAttr`s,
// defaulting to the op's canonical maps and to `TypeFn::cast_signed`
// respectively. ScaIR has no defaulted attributes, so they are plain optionals
// and an absent attribute carries the upstream default implicitly.
case class Matmul(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    indexing_maps: Option[ArrayAttribute[AffineMapAttr]] = None,
    cast: Option[TypeFn] = Some(TypeFn.cast_signed),
    region: Region,
) extends DerivedOperation["linalg.matmul"]
    with LinalgStructuredBase
    with LinalgContractionOpInterface derives OpDefs

/*≡==--=≡≡≡≡≡≡≡=--=≡≡*\
||    CONTRACT OP    ||
\*≡==---=≡≡≡≡≡=---==≡*/

case class Contract(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[ShapedType]] = Seq.empty,
    indexing_maps: ArrayAttribute[AffineMapAttr],
    cast: Option[TypeFn] = Some(TypeFn.cast_signed),
    combiner: Region,
) extends DerivedOperation["linalg.contract"]
    with LinalgStructuredBase
    with LinalgContractionOpInterface derives OpDefs

/*≡==--=≡≡≡≡≡≡≡≡≡=--=≡≡*\
||   BATCH_MATMUL OP   ||
\*≡==---=≡≡≡≡≡≡≡=---==≡*/

case class BatchMatmul(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    indexing_maps: Option[ArrayAttribute[AffineMapAttr]] = None,
    cast: Option[TypeFn] = Some(TypeFn.cast_signed),
    region: Region,
) extends DerivedOperation["linalg.batch_matmul"]
    with LinalgStructuredBase
    with LinalgContractionOpInterface derives OpDefs

/*≡==--=≡≡≡≡≡≡≡≡≡≡≡≡≡≡=--=≡≡*\
||  BATCH_REDUCE_MATMUL OP  ||
\*≡==---=≡≡≡≡≡≡≡≡≡≡≡≡=---==≡*/

case class BatchReduceMatmul(
    inputs: Seq[Operand[Attribute]] = Seq.empty,
    outputs: Seq[Operand[ShapedType]] = Seq.empty,
    result_tensors: Seq[Result[RankedTensorType]] = Seq.empty,
    indexing_maps: Option[ArrayAttribute[AffineMapAttr]] = None,
    cast: Option[TypeFn] = Some(TypeFn.cast_signed),
    region: Region,
) extends DerivedOperation["linalg.batch_reduce_matmul"]
    with LinalgStructuredBase
    with LinalgContractionOpInterface derives OpDefs
