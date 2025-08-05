package scair.dialects.memref

import scair.clair.codegen.*
import scair.clair.macros.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.dialects.arith.SameOperandsAndResultTypes
import scair.dialects.scf.Index
import scair.dialects.func.InferTypeOpAdaptor
import scair.dialects.LingoDB.DBOps.CastOp
import scair.dialects.arith.AllTypesMatch

/*≡=--==≡≡≡≡==--=≡*\
||   INTERFACES   ||
\*≡=---==≡≡==---=≡*/

trait CallOpInterface
trait MemRefsNormalizable
trait SymbolUserOpInterface
trait DeclareOpInterfaceMethods[T <: Tuple]
trait TypesMatchWith(tuples: (Seq[Attribute], Seq[Attribute])*)
trait ConstantLike
type Pure = NoMemoryEffect
trait OpAsmOpInterface
trait AffineScope
trait AutomaticAllocationScope
trait FunctionOpInterface
trait IsolatedFromAbove
trait HasParent[P <: Operation]
trait ReturnLike
trait ViewLikeOpInterface
trait AutomaticAllocationScopeResource
trait PromotableAllocationOpInterface
trait DestructurableAllocationOpInterface
trait SingleBlockImplicitTerminator[O <: Operation]
trait RecursiveMemoryEffects
trait NoRegionArguments
trait RegionBranchOpInterface
trait AttrSizedOperandSegments
trait CastOpInterface
trait CopyOpInterface
trait SameOperandsElementType
trait SameOperandsShape
trait ConditionallySpeculatable
trait ShapedDimOpInterface
trait InferIntRangeInterface
trait SameVariadicResultSize
trait Symbol
trait PromotableMemOpInterface
trait DestructurableAccesorOpInterface
trait SameOperandsAndResultElementType
trait SameOperandsAndResultShape
trait OffsetSizeAndStrideOpInterface
trait ReifyRankedShapedTypeOpInterface

/*≡=--==≡≡≡≡==--=≡*\
||   ATTRIBUTES   ||
\*≡=---==≡≡==---=≡*/

type UnitAttr = Attribute
type AnyMemRef = MemrefType
type I64Attr = IntegerAttr
type I32Attr = IntegerAttr
type I8 = IntegerAttr
type BoolAttr = Attribute
type DenseI64ArrayAttr = DenseArrayAttr
type AnyRankedOrUnrankedMemRef = MemrefType
type AnyStridedMemRef = MemrefType
type I64ArrayAttr = ArrayAttribute[IntegerAttr]
type IndexListArrayAttr = I64ArrayAttr
type AtomicRMWKindAttr = Attribute
type AnySignlessInteger = IntegerAttr
type AnyFloat = FloatAttr
type AnyStaticShapeMemRef = MemrefType
type MemRefTypeAttr = MemrefType

/*≡=--==≡≡≡≡≡==--=≡*\
||   CONSTRAINTS   ||
\*≡=---==≡≡≡==---=≡*/

// BindBy as in Constrain By
infix type !>[T <: Attribute, Types] = T
type IntPositive = IntegerAttr
type IntMinValue[V <: Int] = IntegerAttr
type IntMaxValue[V <: Int] = IntegerAttr

type MemRefRankOf[Types, Ranks] = MemrefType
type MemFreeAt[Stage <: Int, EffectRange <: Attribute] = MemrefType
type MemAlloc[Resource, Stage, Effect] = MemrefType

type DefaultResource = Attribute
type FullEffect = Attribute

type SizedRegion[V <: Int] = Region

type MemReadAt[Stage <: Int, EffectRange <: Attribute] = MemrefType
type MemWriteAt[Stage <: Int, EffectRange <: Attribute] = MemrefType

type HasRankGreaterOrEqualPred[Rank <: Int] = MemrefType
type AnyNon0RankedMemRef = MemrefType !> HasRankGreaterOrEqualPred[1]
type AnyNon0RankedOrUnrankedMemRef = UnrankedMemrefType | AnyNon0RankedMemRef

type AnyStridedMemRefOfRank[Rank <: Int] = AnyStridedMemRef &
  MemRefRankOf[TypeAttribute, Rank]

type MemRefOf[Tuple] = MemrefType

type MemRead = MemrefType
type MemWrite = MemrefType

type IntPowerOf[V <: Int] = IntegerAttr

/*≡=--==≡≡≡≡==--=≡*\
||   OPERATIONS   ||
\*≡=---==≡≡==---=≡*/

case class AssumeAlignment(
    val alignment: I32Attr !> IntPositive,
    val memref: Operand[AnyMemRef],
    val result: Result[AnyMemRef]
) extends DerivedOperation["memref.assume_alignment", AssumeAlignment],
      OpAsmOpInterface,
      SameOperandsAndResultTypes,
      ViewLikeOpInterface,
      Pure derives DerivedOperationCompanion

case class Alloc(
    val dynamicSizes: Seq[Operand[IndexType]],
    val symbolOperands: Seq[Operand[IndexType]],
    val memref: Result[AnyMemRef !> MemAlloc[DefaultResource, 0, FullEffect]],
    val alignment: Option[I64Attr !> IntMinValue[0]]
) extends DerivedOperation["memref.alloc", Alloc],
      AttrSizedOperandSegments,
      OpAsmOpInterface derives DerivedOperationCompanion

case class Realloc(
    val source: Operand[
      MemRefRankOf[TypeAttribute, 1] !> MemFreeAt[0, FullEffect]
    ],
    val dynamicResultSize: Option[Operand[IndexType]],
    val alignment: Option[IntegerAttr],
    val result: Result[
      MemRefRankOf[TypeAttribute, 1] !> MemAlloc[DefaultResource, 1, FullEffect]
    ]
) extends DerivedOperation["memref.realloc", Realloc]
    derives DerivedOperationCompanion

case class Alloca(
    val dynamicSizes: Seq[Operand[IndexType]],
    val symbolOperands: Seq[Operand[IndexType]],
    val memref: Result[AnyMemRef !> MemAlloc[DefaultResource, 0, FullEffect]],
    val alignment: Option[I64Attr !> IntMinValue[0]]
) extends DerivedOperation["memref.alloca", Alloca],
      AutomaticAllocationScopeResource,
      AttrSizedOperandSegments,
      PromotableAllocationOpInterface,
      DestructurableAllocationOpInterface,
      OpAsmOpInterface derives DerivedOperationCompanion

case class AllocaScope(
    val _results: Seq[Result[TypeAttribute]],
    val bodyRegion: SizedRegion[1]
) extends DerivedOperation["memref.alloca_scope", AllocaScope],
      AutomaticAllocationScope,
      RegionBranchOpInterface,
      SingleBlockImplicitTerminator[AllocaScopeReturn],
      RecursiveMemoryEffects,
      NoRegionArguments derives DerivedOperationCompanion

case class AllocaScopeReturn(
    val _results: Seq[Operand[TypeAttribute]]
) extends DerivedOperation["memref.alloca_scope.return", AllocaScopeReturn],
      HasParent[AllocaScope],
      Pure,
      ReturnLike,
      IsTerminator derives DerivedOperationCompanion

case class Cast(
    val source: Operand[AnyRankedOrUnrankedMemRef],
    val dest: Result[AnyRankedOrUnrankedMemRef]
) extends DerivedOperation["memref.cast", Cast],
      CastOpInterface,
      OpAsmOpInterface,
      MemRefsNormalizable,
      ViewLikeOpInterface,
      SameOperandsAndResultTypes,
      Pure derives DerivedOperationCompanion

case class Copy(
    val source: Operand[AnyRankedOrUnrankedMemRef !> MemReadAt[0, FullEffect]],
    val target: Result[AnyRankedOrUnrankedMemRef !> MemWriteAt[0, FullEffect]]
) extends DerivedOperation["memref.copy", Copy],
      CopyOpInterface,
      SameOperandsElementType,
      SameOperandsShape derives DerivedOperationCompanion

case class Dealloc(
    val memref: Operand[AnyRankedOrUnrankedMemRef !> MemFreeAt[0, FullEffect]]
) extends DerivedOperation["memref.dealloc", Dealloc],
      AssemblyFormat["$memref attr-dict `:` type($memref)"],
      MemRefsNormalizable derives DerivedOperationCompanion

case class Dim(
    source: Operand[AnyNon0RankedOrUnrankedMemRef],
    index: Operand[IndexType],
    result: Result[IndexType]
) extends DerivedOperation["memref.dim", Dim],
      OpAsmOpInterface,
      InferIntRangeInterface,
      MemRefsNormalizable,
      ConditionallySpeculatable,
      NoMemoryEffect,
      ShapedDimOpInterface derives DerivedOperationCompanion

case class DmaStart(
    val _operands: Seq[Operand[TypeAttribute]]
) extends DerivedOperation["memref.dma_start", DmaStart]
    derives DerivedOperationCompanion

case class DmaWait(
    val tagMemRef: Operand[AnyMemRef],
    val tagIndices: Seq[Operand[IndexType]],
    val numElements: Operand[IndexType]
) extends DerivedOperation["memref.dma_wait", DmaWait]
    derives DerivedOperationCompanion

case class ExtractAlignedPointerAsIndex(
    val source: Operand[AnyRankedOrUnrankedMemRef],
    val aligned_pointer: Result[IndexType]
) extends DerivedOperation[
      "memref.extract_aligned_pointer_as_index",
      ExtractAlignedPointerAsIndex
    ],
      OpAsmOpInterface,
      SameVariadicResultSize,
      Pure derives DerivedOperationCompanion

case class ExtractStridedMetadata(
    val source: Operand[AnyStridedMemRef],
    val base_buffer: Result[AnyStridedMemRefOfRank[0]],
    val offset: Result[IndexType],
    val sizes: Seq[Result[IndexType]],
    val strides: Seq[Result[IndexType]]
) extends DerivedOperation[
      "memref.extract_strided_metadata",
      ExtractStridedMetadata
    ],
      OpAsmOpInterface,
      InferTypeOpAdaptor,
      SameVariadicResultSize,
      ViewLikeOpInterface,
      Pure derives DerivedOperationCompanion

case class GenericAtomicRMW(
    val memref: Operand[
      MemRefOf[(AnySignlessInteger, AnyFloat)] !> (MemRead, MemWrite)
    ],
    val indices: Seq[Operand[IndexType]],
    val result: Result[AnySignlessInteger | AnyFloat],
    val atomic_body: Region
) extends DerivedOperation["memref.generic_atomic_rmw", GenericAtomicRMW],
      SingleBlockImplicitTerminator[AtomicYield],
      TypesMatchWith((Seq(memref.typ), Seq(result.typ)))
    derives DerivedOperationCompanion

case class AtomicYield(
    val result: Result[TypeAttribute]
) extends DerivedOperation["memref.atomic_yield", AtomicYield],
      HasParent[GenericAtomicRMW],
      ReturnLike,
      IsTerminator,
      Pure derives DerivedOperationCompanion

case class GetGlobal(
    val _name: SymbolRefAttr,
    val result: Result[AnyStaticShapeMemRef]
) extends DerivedOperation["memref.get_global", GetGlobal],
      SymbolUserOpInterface,
      Pure derives DerivedOperationCompanion

case class Global(
    val sym_name: SymbolRefAttr,
    val sym_visibility: Option[StringData],
    val _type: MemRefTypeAttr,
    val initial_value: Option[Attribute],
    val constant: UnitAttr,
    val alignment: Option[I64Attr]
) extends DerivedOperation["memref.global", Global],
      Symbol derives DerivedOperationCompanion

case class Load(
    memref: Operand[AnyMemRef !> MemRead],
    indices: Seq[Operand[IndexType]],
    nontemporal: BoolAttr,
    alignment: Option[I64Attr !> (IntPositive & IntPowerOf[2])],
    result: Result[Attribute]
) extends DerivedOperation["memref.load", Load],
      TypesMatchWith((Seq(memref.typ), Seq(result.typ))),
      PromotableMemOpInterface,
      DestructurableAccesorOpInterface,
      MemRefsNormalizable derives DerivedOperationCompanion

case class MemorySpaceCast(
    val source: Operand[AnyRankedOrUnrankedMemRef],
    val dest: Result[AnyRankedOrUnrankedMemRef]
) extends DerivedOperation["memref.memory_space_cast", MemorySpaceCast],
      CastOpInterface,
      OpAsmOpInterface,
      SameOperandsAndResultElementType,
      SameOperandsAndResultShape,
      MemRefsNormalizable,
      ViewLikeOpInterface,
      Pure derives DerivedOperationCompanion

case class Prefetch(
    val memref: Operand[AnyMemRef],
    val indices: Seq[Operand[IndexType]],
    val isWrite: BoolAttr,
    val localityHint: I32Attr !> (IntMinValue[0], IntMaxValue[3]),
    val isDataCache: BoolAttr
) extends DerivedOperation["memref.prefetch", Prefetch]
    derives DerivedOperationCompanion

case class ReinterpretCast(
    val source: Operand[AnyRankedOrUnrankedMemRef],
    val offsets: Seq[Operand[IndexType]],
    val sizes: Seq[Operand[IndexType]],
    val strides: Seq[Operand[IndexType]],
    val static_offsets: DenseI64ArrayAttr,
    val static_sizes: DenseI64ArrayAttr,
    val static_strides: DenseI64ArrayAttr,
    val result: Result[AnyStridedMemRef]
) extends DerivedOperation["memref.reinterpret_cast", ReinterpretCast],
      OpAsmOpInterface,
      AttrSizedOperandSegments,
      OffsetSizeAndStrideOpInterface,
      MemRefsNormalizable,
      ViewLikeOpInterface,
      Pure derives DerivedOperationCompanion

case class Rank(
    val memref: Operand[AnyRankedOrUnrankedMemRef],
    val result: Result[IndexType]
) extends DerivedOperation["memref.rank", Rank],
      Pure derives DerivedOperationCompanion

case class Reshape(
    val source: Operand[AnyRankedOrUnrankedMemRef],
    val shape: Operand[
      MemRefRankOf[(AnySignlessInteger, IndexType), 1] !> MemRead
    ],
    val result: Result[AnyRankedOrUnrankedMemRef]
) extends DerivedOperation["memref.reshape", Reshape],
      OpAsmOpInterface,
      ViewLikeOpInterface,
      Pure derives DerivedOperationCompanion

case class ExpandShape(
    val src: Operand[AnyStridedMemRef],
    val output_shape: Seq[Operand[IndexType]],
    val reassociation: IndexListArrayAttr,
    val static_output_shape: DenseI64ArrayAttr,
    val result: Result[AnyStridedMemRef]
) extends DerivedOperation["memref.expand_shape", ExpandShape],
      OpAsmOpInterface,
      ReifyRankedShapedTypeOpInterface,
      ViewLikeOpInterface,
      Pure derives DerivedOperationCompanion

case class CollapseShape(
    val src: Operand[AnyStridedMemRef],
    val reassociation: IndexListArrayAttr,
    val result: Result[AnyStridedMemRef]
) extends DerivedOperation["memref.collapse_shape", CollapseShape],
      OpAsmOpInterface,
      ViewLikeOpInterface,
      Pure derives DerivedOperationCompanion

case class Store(
    value: Operand[TypeAttribute],
    memref: Operand[AnyMemRef !> MemWrite],
    indices: Seq[Operand[IndexType]],
    nontemporal: BoolAttr,
    alignment: Option[I64Attr !> (IntPositive & IntPowerOf[2])]
) extends DerivedOperation["memref.store", Store],
      PromotableMemOpInterface,
      DestructurableAccesorOpInterface,
      TypesMatchWith((Seq(memref.typ), Seq(value.typ))),
      MemRefsNormalizable derives DerivedOperationCompanion

case class SubView(
    val source: Operand[AnyMemRef],
    val offsets: Seq[Operand[IndexType]],
    val sizes: Seq[Operand[IndexType]],
    val strides: Seq[Operand[IndexType]],
    val static_offsets: DenseI64ArrayAttr,
    val static_sizes: DenseI64ArrayAttr,
    val static_strides: DenseI64ArrayAttr,
    val result: Result[AnyMemRef]
) extends DerivedOperation["memref.subview", SubView],
      OpAsmOpInterface,
      ViewLikeOpInterface,
      AttrSizedOperandSegments,
      OffsetSizeAndStrideOpInterface,
      Pure derives DerivedOperationCompanion

case class Transpose(
    val in: Operand[AnyStridedMemRef],
    val permutation: AffineMapAttr,
    val result: Result[AnyStridedMemRef]
) extends DerivedOperation["memref.transpose", Transpose],
      OpAsmOpInterface,
      Pure derives DerivedOperationCompanion

case class View(
    val source: Operand[MemRefRankOf[I8, 1]],
    val byte_shift: Operand[IndexType],
    val sizes: Seq[Operand[IndexType]],
    val result: Result[AnyMemRef]
) extends DerivedOperation["memref.view", View],
      OpAsmOpInterface,
      ViewLikeOpInterface,
      Pure derives DerivedOperationCompanion

case class AtomicRMW(
    val kind: AtomicRMWKindAttr,
    val value: Operand[AnySignlessInteger | AnyFloat],
    val memref: Operand[
      MemRefOf[(AnySignlessInteger, AnyFloat)] !> (MemRead, MemWrite)
    ],
    val indices: Seq[Operand[IndexType]],
    val result: Result[AnySignlessInteger | AnyFloat]
) extends DerivedOperation["memref.atomic_rmw", AtomicRMW],
      AllTypesMatch(value.typ, result.typ) derives DerivedOperationCompanion

val MemrefDialect =
  summonDialect[EmptyTuple, (Alloc, Dealloc, Load, Store, Dim)](Seq())
