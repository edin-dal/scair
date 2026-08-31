package scair.dialects.affine

import scair.clair.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.utils.OK

// ░█████╗░ ███████╗ ███████╗ ██╗ ███╗░░██╗ ███████╗
// ██╔══██╗ ██╔════╝ ██╔════╝ ██║ ████╗░██║ ██╔════╝
// ███████║ █████╗░░ █████╗░░ ██║ ██╔██╗██║ █████╗░░
// ██╔══██║ ██╔══╝░░ ██╔══╝░░ ██║ ██║╚████║ ██╔══╝░░
// ██║░░██║ ██║░░░░░ ██║░░░░░ ██║ ██║░╚███║ ███████╗
// ╚═╝░░╚═╝ ╚═╝░░░░░ ╚═╝░░░░░ ╚═╝ ╚═╝░░╚══╝ ╚══════╝

// ░█████╗░ ██████╗░ ███████╗ ██████╗░ ░█████╗░ ████████╗ ██╗ ░█████╗░ ███╗░░██╗ ░██████╗
// ██╔══██╗ ██╔══██╗ ██╔════╝ ██╔══██╗ ██╔══██╗ ╚══██╔══╝ ██║ ██╔══██╗ ████╗░██║ ██╔════╝
// ██║░░██║ ██████╔╝ █████╗░░ ██████╔╝ ███████║ ░░░██║░░░ ██║ ██║░░██║ ██╔██╗██║ ╚█████╗░
// ██║░░██║ ██╔═══╝░ ██╔══╝░░ ██╔══██╗ ██╔══██║ ░░░██║░░░ ██║ ██║░░██║ ██║╚████║ ░╚═══██╗
// ╚█████╔╝ ██║░░░░░ ███████╗ ██║░░██║ ██║░░██║ ░░░██║░░░ ██║ ╚█████╔╝ ██║░╚███║ ██████╔╝
// ░╚════╝░ ╚═╝░░░░░ ╚══════╝ ╚═╝░░╚═╝ ╚═╝░░╚═╝ ░░░╚═╝░░░ ╚═╝ ░╚════╝░ ╚═╝░░╚══╝ ╚═════╝░

/*≡==---==≡≡≡≡==---=≡≡*\
||      APPLY OP      ||
\*≡==----==≡≡==----==≡*/

case class Apply(
    mapOperands: Seq[Operand[IndexType]] = Seq.empty,
    res: Result[IndexType],
    map: AffineMapAttr,
) extends DerivedOperation["affine.apply"]
    with Pure derives OpDefs

/*≡==---=≡≡≡≡=---=≡≡*\
||      FOR OP      ||
\*≡==----=≡≡=----==≡*/

case class For(
    lowerBoundOperands: Seq[Operand[IndexType]] = Seq.empty,
    upperBoundOperands: Seq[Operand[IndexType]] = Seq.empty,
    inits: Seq[Operand[Attribute]] = Seq.empty,
    res: Seq[Result[Attribute]] = Seq.empty,
    lowerBoundMap: AffineMapAttr,
    upperBoundMap: AffineMapAttr,
    step: IntegerAttr,
    body: Region,
) extends DerivedOperation["affine.for"]
    with RecursiveMemoryEffects
    with RecursivelySpeculatable derives OpDefs

/*≡==---==≡≡≡≡≡==---=≡≡*\
||     PARALLEL OP     ||
\*≡==----==≡≡≡==----==≡*/

case class Parallel(
    mapOperands: Seq[Operand[IndexType]] = Seq.empty,
    steps: Option[ArrayAttribute[IntegerAttr]] = None,
    reductions: Attribute,
    lowerBoundsMap: AffineMapAttr,
    lowerBoundsGroups: DenseIntOrFPElementsAttr,
    upperBoundsMap: AffineMapAttr,
    upperBoundsGroups: DenseIntOrFPElementsAttr,
    res: Seq[Result[Attribute]] = Seq.empty,
    body: Region,
) extends DerivedOperation["affine.parallel"]
    with RecursiveMemoryEffects derives OpDefs

/*≡==--=≡≡≡=--=≡≡*\
||     IF OP     ||
\*≡==---=≡=---==≡*/

case class If(
    args: Seq[Operand[Attribute]] = Seq.empty,
    res: Seq[Result[Attribute]] = Seq.empty,
    condition: AffineSetAttr,
    thenRegion: Region,
    elseRegion: Region,
) extends DerivedOperation["affine.if"]
    with RecursiveMemoryEffects
    with RecursivelySpeculatable derives OpDefs

/*≡==--=≡≡≡≡=--=≡≡*\
||    STORE OP    ||
\*≡==--==≡≡==--==≡*/

case class Store(
    value: Operand[Attribute],
    memref: Operand[MemrefType],
    indices: Seq[Operand[IndexType]] = Seq.empty,
    map: AffineMapAttr,
) extends DerivedOperation["affine.store"] derives OpDefs

/*≡==---=≡≡≡=---=≡≡*\
||     LOAD OP     ||
\*≡==----=≡=----==≡*/

case class Load(
    memref: Operand[MemrefType],
    indices: Seq[Operand[IndexType]] = Seq.empty,
    result: Result[Attribute],
    map: AffineMapAttr,
) extends DerivedOperation["affine.load"] derives OpDefs

/*≡==--=≡≡≡≡=--=≡≡*\
||     MIN OP     ||
\*≡==---=≡≡=---==≡*/

case class Min(
    arguments: Seq[Operand[IndexType]] = Seq.empty,
    result: Result[IndexType],
    map: AffineMapAttr,
) extends DerivedOperation["affine.min"]
    with Pure derives OpDefs

/*≡==--=≡≡≡≡=--=≡≡*\
||    YIELD OP    ||
\*≡==---=≡≡=---==≡*/

case class Yield(
    arguments: Seq[Operand[Attribute]] = Seq.empty
) extends DerivedOperation["affine.yield"]
    with IsTerminator
    with AssemblyFormat["attr-dict ($arguments^ `:` type($arguments))?"]
    with Pure derives OpDefs

/*≡==--=≡≡≡≡=--=≡≡*\
||     MAX OP     ||
\*≡==---=≡≡=---==≡*/

case class Max(
    arguments: Seq[Operand[IndexType]] = Seq.empty,
    result: Result[IndexType],
    map: AffineMapAttr,
) extends DerivedOperation["affine.max"]
    with Pure derives OpDefs

/*≡==---=≡≡≡≡≡≡≡=---=≡≡*\
||   VECTOR_LOAD OP    ||
\*≡==----=≡≡≡≡≡=----==≡*/

case class VectorLoad(
    memref: Operand[MemrefType],
    indices: Seq[Operand[IndexType]] = Seq.empty,
    result: Result[VectorType],
    map: AffineMapAttr,
) extends DerivedOperation["affine.vector_load"] derives OpDefs

/*≡==---=≡≡≡≡≡≡≡≡=---=≡≡*\
||   VECTOR_STORE OP    ||
\*≡==----=≡≡≡≡≡≡=----==≡*/

case class VectorStore(
    value: Operand[VectorType],
    memref: Operand[MemrefType],
    indices: Seq[Operand[IndexType]] = Seq.empty,
    map: AffineMapAttr,
) extends DerivedOperation["affine.vector_store"] derives OpDefs

/*≡==---=≡≡≡≡≡=---=≡≡*\
||    PREFETCH OP    ||
\*≡==----=≡≡≡=----==≡*/

// TODO: `isWrite` and `isDataCache` are `BoolAttr` upstream. ScaIR has no
// `BoolAttr` (an MLIR `BoolAttr` is an `IntegerAttr` of i1), so they are modeled
// here as `IntegerAttr` carrying an i1 value (printed/parsed as `true`/`false`).
// Switch to a dedicated `BoolAttr` if one is added.
case class Prefetch(
    memref: Operand[MemrefType],
    indices: Seq[Operand[IndexType]] = Seq.empty,
    isWrite: IntegerAttr,
    localityHint: IntegerAttr,
    isDataCache: IntegerAttr,
    map: AffineMapAttr,
) extends DerivedOperation["affine.prefetch"] derives OpDefs

/*≡==---=≡≡≡≡≡≡≡≡≡≡≡=---=≡≡*\
||   DELINEARIZE_INDEX OP  ||
\*≡==----=≡≡≡≡≡≡≡≡≡=----==≡*/

// TODO: `static_basis` is a `DenseI64ArrayAttr` upstream; modeled here with the
// generic `DenseArrayAttr` (its element type carries the i64). The field name is
// snake_case so the generic printer emits the MLIR attribute name `static_basis`
// (verified to round-trip through real mlir-opt).
case class DelinearizeIndex(
    linearIndex: Operand[IndexType],
    dynamicBasis: Seq[Operand[IndexType]] = Seq.empty,
    multiIndex: Seq[Result[IndexType]] = Seq.empty,
    static_basis: DenseArrayAttr,
) extends DerivedOperation["affine.delinearize_index"]
    with Pure derives OpDefs

/*≡==---=≡≡≡≡≡≡≡≡≡=---=≡≡*\
||   LINEARIZE_INDEX OP  ||
\*≡==----=≡≡≡≡≡≡≡=----==≡*/

// `disjoint` is a UnitProp in MLIR, slightly different from a UnitAttr!
// TODO: `static_basis` is a `DenseI64ArrayAttr` upstream, modeled with
// `DenseArrayAttr` as in `DelinearizeIndex`. This op has two variadic operand
// lists, so the generic printer emits an `operandSegmentSizes` property (as
// `affine.for` already does).
case class LinearizeIndex(
    multiIndex: Seq[Operand[IndexType]] = Seq.empty,
    dynamicBasis: Seq[Operand[IndexType]] = Seq.empty,
    linearIndex: Result[IndexType],
    static_basis: DenseArrayAttr,
    disjoint: Option[UnitAttr] = None,
) extends DerivedOperation["affine.linearize_index"]
    with Pure derives OpDefs

/*≡==---=≡≡≡≡≡≡=---=≡≡*\
||    DMA_START OP    ||
\*≡==----=≡≡≡≡=----==≡*/

// The upstream ODS models the transfer as a single variadic operand list plus the
// three affine maps (the structured src/dst/tag memref+index grouping and the
// trailing num_elements/stride operands are derived from the maps rather than
// expressed as separate operand segments).
case class DmaStart(
    arguments: Seq[Operand[Attribute]] = Seq.empty
) extends DerivedOperation["affine.dma_start"] derives OpDefs:
  def src_map: AffineMapAttr = attributes("src_map").asInstanceOf[AffineMapAttr]
  def dst_map: AffineMapAttr = attributes("dst_map").asInstanceOf[AffineMapAttr]
  def tag_map: AffineMapAttr = attributes("tag_map").asInstanceOf[AffineMapAttr]

  override def customVerify(): OK[Operation] =
    attributes.get("src_map") match
      case Some(_: AffineMapAttr) => OK(this)
      case _                      => Err("src_map must be an AffineMapAttr")
    attributes.get("dst_map") match
      case Some(_: AffineMapAttr) => OK(this)
      case _                      => Err("dst_map must be an AffineMapAttr")
    attributes.get("tag_map") match
      case Some(_: AffineMapAttr) => OK(this)
      case _                      => Err("tag_map must be an AffineMapAttr")

/*≡==---=≡≡≡≡=---=≡≡*\
||   DMA_WAIT OP    ||
\*≡==----=≡≡=----==≡*/

case class DmaWait(
    arguments: Seq[Operand[Attribute]] = Seq.empty
) extends DerivedOperation["affine.dma_wait"] derives OpDefs:
  def tag_map: AffineMapAttr = attributes("tag_map").asInstanceOf[AffineMapAttr]

  override def customVerify(): OK[Operation] =
    attributes.get("tag_map") match
      case Some(_: AffineMapAttr) => OK(this)
      case _                      => Err("tag_map must be an AffineMapAttr")

val AffineDialect = summonDialect[
  EmptyTuple,
  (
      Apply,
      For,
      Parallel,
      If,
      Store,
      Load,
      Min,
      Yield,
      Max,
      VectorLoad,
      VectorStore,
      Prefetch,
      DelinearizeIndex,
      LinearizeIndex,
      DmaStart,
      DmaWait,
  ),
]
