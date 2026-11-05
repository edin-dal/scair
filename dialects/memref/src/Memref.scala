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

// TODO: Upstream `memref.dim` is `ConditionallySpeculatable`, speculatable
//       only when its index is known to be in range. With no integer range
//       analysis here it stays unspeculatable, which is conservative.
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

case class Alloca(
    dynamicSizes: Seq[Operand[IndexType]] = Seq.empty,
    symbolOperands: Seq[Operand[IndexType]] = Seq.empty,
    memref: Result[MemrefType],
    alignment: Option[IntegerAttr] = None,
) extends DerivedOperation["memref.alloca"] derives OpDefs

case class AssumeAlignment(
    memref: Operand[MemrefType],
    alignment: IntegerAttr,
    result: Result[MemrefType],
) extends DerivedOperation["memref.assume_alignment"]
    with Pure derives OpDefs

// TODO: `kind` is arith's `AtomicRMWKind` enum, absent from dialects/arith.
case class AtomicRMW(
    kind: Attribute,
    value: Operand[IntegerType | FloatType],
    memref: Operand[MemrefType],
    indices: Seq[Operand[IndexType]] = Seq.empty,
    result: Result[IntegerType | FloatType],
) extends DerivedOperation["memref.atomic_rmw"] derives OpDefs

case class Cast(
    source: Operand[MemrefType],
    dest: Result[MemrefType],
) extends DerivedOperation["memref.cast"]
    with AssemblyFormat["$source attr-dict `:` type($source) `to` type($dest)"]
    with Pure derives OpDefs

case class CollapseShape(
    src: Operand[RankedMemrefType],
    reassociation: ArrayAttribute[ArrayAttribute[IntegerAttr]],
    result: Result[RankedMemrefType],
) extends DerivedOperation["memref.collapse_shape"]
    with AssemblyFormat[
      "$src $reassociation attr-dict `:` type($src) `into` type($result)"
    ]
    with Pure derives OpDefs

case class Copy(
    source: Operand[MemrefType],
    target: Operand[MemrefType],
) extends DerivedOperation["memref.copy"]
    with AssemblyFormat[
      "$source `,` $target attr-dict `:` type($source) `to` type($target)"
    ] derives OpDefs

case class ExpandShape(
    src: Operand[RankedMemrefType],
    reassociation: ArrayAttribute[ArrayAttribute[IntegerAttr]],
    output_shape: Seq[Operand[IndexType]] = Seq.empty,
    static_output_shape: DenseArrayAttr,
    result: Result[RankedMemrefType],
) extends DerivedOperation["memref.expand_shape"]
    with Pure derives OpDefs

case class ExtractStridedMetadata(
    source: Operand[RankedMemrefType],
    base_buffer: Result[RankedMemrefType],
    offset: Result[IndexType],
    sizes: Seq[Result[IndexType]] = Seq.empty,
    strides: Seq[Result[IndexType]] = Seq.empty,
) extends DerivedOperation["memref.extract_strided_metadata"]
    with SameVariadicResultSize
    with Pure derives OpDefs

case class GetGlobal(
    @irName("name") global: SymbolRefAttr,
    result: Result[MemrefType],
) extends DerivedOperation["memref.get_global"]
    with AssemblyFormat["$name `:` type($result) attr-dict"]
    with Pure derives OpDefs

case class Global(
    sym_name: StringData,
    sym_visibility: Option[StringData] = None,
    `type`: MemrefType,
    initial_value: Option[Attribute] = None,
    constant: Option[UnitAttr] = None,
    alignment: Option[IntegerAttr] = None,
) extends DerivedOperation["memref.global"]
    with Symbol derives OpDefs

case class MemorySpaceCast(
    source: Operand[MemrefType],
    dest: Result[MemrefType],
) extends DerivedOperation["memref.memory_space_cast"]
    with AssemblyFormat["$source attr-dict `:` type($source) `to` type($dest)"]
    with Pure derives OpDefs

case class Prefetch(
    memref: Operand[MemrefType],
    indices: Seq[Operand[IndexType]] = Seq.empty,
    isWrite: IntegerAttr,
    localityHint: IntegerAttr,
    isDataCache: IntegerAttr,
) extends DerivedOperation["memref.prefetch"] derives OpDefs

case class Rank(
    memref: Operand[MemrefType],
    result: Result[IndexType],
) extends DerivedOperation["memref.rank"]
    with Pure derives OpDefs

case class Realloc(
    source: Operand[RankedMemrefType],
    dynamicResultSize: Option[Operand[IndexType]] = None,
    alignment: Option[IntegerAttr] = None,
    result: Result[RankedMemrefType],
) extends DerivedOperation["memref.realloc"] derives OpDefs

case class Reshape(
    source: Operand[MemrefType],
    shape: Operand[RankedMemrefType],
    result: Result[MemrefType],
) extends DerivedOperation["memref.reshape"]
    with Pure derives OpDefs

case class Subview(
    source: Operand[MemrefType],
    offsets: Seq[Operand[IndexType]] = Seq.empty,
    sizes: Seq[Operand[IndexType]] = Seq.empty,
    strides: Seq[Operand[IndexType]] = Seq.empty,
    static_offsets: DenseArrayAttr,
    static_sizes: DenseArrayAttr,
    static_strides: DenseArrayAttr,
    result: Result[MemrefType],
) extends DerivedOperation["memref.subview"]
    with Pure derives OpDefs

case class Transpose(
    in: Operand[RankedMemrefType],
    permutation: AffineMapAttr,
    result: Result[RankedMemrefType],
) extends DerivedOperation["memref.transpose"]
    with Pure derives OpDefs

case class View(
    source: Operand[RankedMemrefType],
    byte_shift: Operand[IndexType],
    sizes: Seq[Operand[IndexType]] = Seq.empty,
    result: Result[MemrefType],
) extends DerivedOperation["memref.view"]
    with Pure derives OpDefs

val MemrefDialect =
  summonDialect[
    EmptyTuple,
    (
        Alloc,
        Dealloc,
        Load,
        Store,
        Dim,
        Alloca,
        AssumeAlignment,
        AtomicRMW,
        Cast,
        CollapseShape,
        Copy,
        ExpandShape,
        ExtractStridedMetadata,
        GetGlobal,
        Global,
        MemorySpaceCast,
        Prefetch,
        Rank,
        Realloc,
        Reshape,
        Subview,
        Transpose,
        View,
    ),
  ]
