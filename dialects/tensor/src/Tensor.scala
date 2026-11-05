package scair.dialects.tensor

import scair.clair.*
import scair.dialects.builtin.*
import scair.ir.*

// ████████╗ ███████╗ ███╗░░██╗ ░██████╗ ░█████╗░ ██████╗░
// ╚══██╔══╝ ██╔════╝ ████╗░██║ ██╔════╝ ██╔══██╗ ██╔══██╗
// ░░░██║░░░ █████╗░░ ██╔██╗██║ ╚█████╗░ ██║░░██║ ██████╔╝
// ░░░██║░░░ ██╔══╝░░ ██║╚████║ ░╚═══██╗ ██║░░██║ ██╔══██╗
// ░░░██║░░░ ███████╗ ██║░╚███║ ██████╔╝ ╚█████╔╝ ██║░░██║
// ░░░╚═╝░░░ ╚══════╝ ╚═╝░░╚══╝ ╚═════╝░ ░╚════╝░ ╚═╝░░╚═╝

case class Bitcast(
    source: Operand[TensorType],
    dest: Result[TensorType],
) extends DerivedOperation["tensor.bitcast"]
    with AssemblyFormat["$source attr-dict `:` type($source) `to` type($dest)"]
    with Pure derives OpDefs

case class Cast(
    source: Operand[TensorType],
    dest: Result[TensorType],
) extends DerivedOperation["tensor.cast"]
    with AssemblyFormat["$source attr-dict `:` type($source) `to` type($dest)"]
    with Pure derives OpDefs

case class CollapseShape(
    src: Operand[TensorType],
    reassociation: ArrayAttribute[ArrayAttribute[IntegerAttr]],
    result: Result[TensorType],
) extends DerivedOperation["tensor.collapse_shape"]
    with AssemblyFormat[
      "$src $reassociation attr-dict `:` type($src) `into` type($result)"
    ]
    with Pure derives OpDefs

case class Concat(
    dim: IntegerAttr,
    inputs: Seq[Operand[RankedTensorType]] = Seq.empty,
    result: Result[RankedTensorType],
) extends DerivedOperation["tensor.concat"]
    with Pure derives OpDefs

// TODO: Upstream `tensor.dim` is `ConditionallySpeculatable`, speculatable
//       only when its index is known to be in range.
case class Dim(
    source: Operand[TensorType],
    index: Operand[IndexType],
    result: Result[IndexType],
) extends DerivedOperation["tensor.dim"]
    with NoMemoryEffect derives OpDefs

case class Empty(
    dynamicSizes: Seq[Operand[IndexType]] = Seq.empty,
    result: Result[RankedTensorType],
) extends DerivedOperation["tensor.empty"]
    with Pure derives OpDefs

case class ExpandShape(
    src: Operand[TensorType],
    reassociation: ArrayAttribute[ArrayAttribute[IntegerAttr]],
    output_shape: Seq[Operand[IndexType]] = Seq.empty,
    static_output_shape: DenseArrayAttr,
    result: Result[TensorType],
) extends DerivedOperation["tensor.expand_shape"]
    with Pure derives OpDefs

case class Extract(
    tensor: Operand[RankedTensorType],
    indices: Seq[Operand[IndexType]] = Seq.empty,
    result: Result[Attribute],
) extends DerivedOperation["tensor.extract"]
    with Pure derives OpDefs

case class ExtractSlice(
    source: Operand[RankedTensorType],
    offsets: Seq[Operand[IndexType]] = Seq.empty,
    sizes: Seq[Operand[IndexType]] = Seq.empty,
    strides: Seq[Operand[IndexType]] = Seq.empty,
    static_offsets: DenseArrayAttr,
    static_sizes: DenseArrayAttr,
    static_strides: DenseArrayAttr,
    result: Result[RankedTensorType],
) extends DerivedOperation["tensor.extract_slice"]
    with Pure derives OpDefs

case class FromElements(
    elements: Seq[Operand[Attribute]] = Seq.empty,
    result: Result[RankedTensorType],
) extends DerivedOperation["tensor.from_elements"]
    with Pure derives OpDefs

case class Generate(
    dynamicExtents: Seq[Operand[IndexType]] = Seq.empty,
    result: Result[RankedTensorType],
    body: Region,
) extends DerivedOperation["tensor.generate"]
    with RecursiveMemoryEffects derives OpDefs

case class Insert(
    scalar: Operand[Attribute],
    dest: Operand[RankedTensorType],
    indices: Seq[Operand[IndexType]] = Seq.empty,
    result: Result[RankedTensorType],
) extends DerivedOperation["tensor.insert"]
    with Pure derives OpDefs

case class InsertSlice(
    source: Operand[RankedTensorType],
    dest: Operand[RankedTensorType],
    offsets: Seq[Operand[IndexType]] = Seq.empty,
    sizes: Seq[Operand[IndexType]] = Seq.empty,
    strides: Seq[Operand[IndexType]] = Seq.empty,
    static_offsets: DenseArrayAttr,
    static_sizes: DenseArrayAttr,
    static_strides: DenseArrayAttr,
    result: Result[RankedTensorType],
) extends DerivedOperation["tensor.insert_slice"]
    with Pure derives OpDefs

case class Pad(
    source: Operand[RankedTensorType],
    low: Seq[Operand[IndexType]] = Seq.empty,
    high: Seq[Operand[IndexType]] = Seq.empty,
    static_low: DenseArrayAttr,
    static_high: DenseArrayAttr,
    nofold: Option[UnitAttr] = None,
    result: Result[RankedTensorType],
    region: Region,
) extends DerivedOperation["tensor.pad"]
    with Pure derives OpDefs

case class Rank(
    tensor: Operand[TensorType],
    result: Result[IndexType],
) extends DerivedOperation["tensor.rank"]
    with Pure derives OpDefs

case class Reshape(
    source: Operand[TensorType],
    shape: Operand[RankedTensorType],
    result: Result[TensorType],
) extends DerivedOperation["tensor.reshape"]
    with Pure derives OpDefs

case class Splat(
    input: Operand[Attribute],
    dynamicSizes: Seq[Operand[IndexType]] = Seq.empty,
    aggregate: Result[RankedTensorType],
) extends DerivedOperation["tensor.splat"]
    with Pure derives OpDefs

case class Yield(
    value: Operand[Attribute]
) extends DerivedOperation["tensor.yield"]
    with AssemblyFormat["$value attr-dict `:` type($value)"]
    with IsTerminator
    with Pure derives OpDefs

val TensorDialect =
  summonDialect[
    EmptyTuple,
    (
        Bitcast,
        Cast,
        CollapseShape,
        Concat,
        Dim,
        Empty,
        ExpandShape,
        Extract,
        ExtractSlice,
        FromElements,
        Generate,
        Insert,
        InsertSlice,
        Pad,
        Rank,
        Reshape,
        Splat,
        Yield,
    ),
  ]
