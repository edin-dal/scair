package scair.transformations

import scair.dialects.builtin.UnrealizedConversionCastOp
import scair.ir.*

import scala.collection.mutable

//
// ░█████╗░ ░█████╗░ ███╗░░██╗ ██╗░░░██╗ ███████╗ ██████╗░ ░██████╗ ██╗ ░█████╗░ ███╗░░██╗
// ██╔══██╗ ██╔══██╗ ████╗░██║ ██║░░░██║ ██╔════╝ ██╔══██╗ ██╔════╝ ██║ ██╔══██╗ ████╗░██║
// ██║░░╚═╝ ██║░░██║ ██╔██╗██║ ╚██╗░██╔╝ █████╗░░ ██████╔╝ ╚█████╗░ ██║ ██║░░██║ ██╔██╗██║
// ██║░░██╗ ██║░░██║ ██║╚████║ ░╚████╔╝░ ██╔══╝░░ ██╔══██╗ ░╚═══██╗ ██║ ██║░░██║ ██║╚████║
// ╚█████╔╝ ╚█████╔╝ ██║░╚███║ ░░╚██╔╝░░ ███████╗ ██║░░██║ ██████╔╝ ██║ ╚█████╔╝ ██║░╚███║
// ░╚════╝░ ░╚════╝░ ╚═╝░░╚══╝ ░░░╚═╝░░░ ╚══════╝ ╚═╝░░╚═╝ ╚═════╝░ ╚═╝ ░╚════╝░ ╚═╝░░╚══╝
//

/*≡==--==≡≡≡≡≡≡≡==--=≡≡*\
||   Type conversion   ||
\*≡==---==≡≡≡≡≡==---==≡*/

abstract class TypeConversionPattern:
  def convert(attr: Attribute)(using TypeConverter): Option[Attribute]

/** Defines a TypeConversionPattern from a partial function on attributes.
  *
  * The enclosing converter is in scope, so a pattern may call `convertType` on
  * nested attributes:
  * {{{
  * typeConversion { case MemRefType(element, shape) =>
  *   MemRefType(convertType(element), shape)
  * }
  * }}}
  */
inline def typeConversion(
    inline partial: TypeConverter ?=> PartialFunction[Attribute, Attribute]
): TypeConversionPattern =
  object typeConversion extends TypeConversionPattern:
    override def convert(attr: Attribute)(using
        converter: TypeConverter
    ): Option[Attribute] =
      partial(using converter).lift(attr)

  typeConversion

/** Maps attributes to the attributes they convert to, 1:1.
  *
  * `convertType` is total: an attribute no pattern matches converts to itself.
  */
final class TypeConverter(val patterns: Seq[TypeConversionPattern]):

  private given TypeConverter = this

  private val converted = mutable.Map.empty[Attribute, Attribute]

  def convertType(attr: Attribute): Attribute =
    // Not `getOrElseUpdate`: a pattern may recurse into `convertType` for a
    // nested attribute, and reentering `getOrElseUpdate` is undefined.
    converted.get(attr) match
      case Some(c) => c
      case None    =>
        val c = patterns.view.flatMap(_.convert(attr)).headOption
          .getOrElse(attr)
        converted(attr) = c
        c

/** The converter in scope, for use in the body of a conversion pattern. */
def convertType(attr: Attribute)(using converter: TypeConverter): Attribute =
  converter.convertType(attr)

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||   Operation conversion   ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

/** The operands of the matched operation, converted.
  *
  * Invariant: `operands(i).typ == convertType(op.operands(i).typ)`. An operand
  * whose defining operation was itself converted is the converted value; one
  * that was not is bridged by a materialized cast.
  *
  * The operands are materialized on first read, so a pattern that never reads
  * them never causes a cast to be emitted.
  */
final class Adaptor private[transformations] (
    private val materialise: () => Seq[Value[Attribute]]
):

  lazy val operands: Seq[Value[Attribute]] = materialise()

  def apply(index: Int): Value[Attribute] = operands(index)

/** The adaptor in scope, for use in the body of a conversion pattern. */
def adaptor(using a: Adaptor): Adaptor = a

/** A [[RewriteResult]] as the driver sees it: `conversionPattern` maps
  * `PatternAction.Abort` to "no conversion", so only `Erase` reaches here.
  */
type ConversionResult = PatternAction.Erase.type | Operation | Seq[Operation] |
  (Operation | Seq[Operation], Value[?] | Seq[Value[?]])

abstract class ConversionPattern:

  def convert(op: Operation, adaptor: Adaptor)(using
      TypeConverter
  ): Option[ConversionResult]

/** Defines a ConversionPattern from a partial function, with the same return
  * protocol as [[pattern]]:
  *   - `Operation` / `Seq[Operation]`: the operations replacing the matched
  *     one, whose results are those of the last of them
  *   - `(Operation | Seq[Operation], Value[?] | Seq[Value[?]])`: the same, with
  *     the replacing results given explicitly
  *   - `PatternAction.Erase`: drop the matched operation
  *   - `PatternAction.Abort`, or not matching at all: leave the operation to
  *     the next pattern, and to the driver if no pattern converts it
  *
  * The converter and the adaptor are in scope:
  * {{{
  * conversionPattern { case op: arith.AddI =>
  *   llvm.Add(adaptor(0), adaptor(1), Result(convertType(op.result.typ)))
  * }
  * }}}
  */
inline def conversionPattern(
    inline partial: (TypeConverter, Adaptor) ?=> PartialFunction[
      Operation,
      RewriteResult,
    ]
): ConversionPattern =
  object conversionPattern extends ConversionPattern:
    override def convert(op: Operation, adaptor: Adaptor)(using
        converter: TypeConverter
    ): Option[ConversionResult] =
      partial(using converter, adaptor).lift(op) match
        case Some(PatternAction.Abort) => None
        case other => other.asInstanceOf[Option[ConversionResult]]

  conversionPattern

/*≡==--==≡≡≡≡==--=≡≡*\
||     Driver       ||
\*≡==---==≡≡==---==≡*/

/** A one-shot dialect conversion.
  *
  * Each region is swept block by block: a block whose argument types convert is
  * rebuilt with the converted signature, and every operation is drained from
  * its block and placed into the target. An operation is placed in one of three
  * ways:
  *
  *   - converted by a pattern, from the converted operands the adaptor carries;
  *   - moved over verbatim, when no pattern converts it and none of its
  *     operands was converted - its results and their uses are left untouched;
  *   - rebuilt, when no pattern converts it but some operand was, with each
  *     such operand cast back to the type it had.
  *
  * Casts are therefore only emitted where converted and unconverted operations
  * meet; a region that converts end to end emits none.
  *
  * A block is given a converted signature only when the operation holding its
  * region was itself converted, so that a conversion does not rewrite the
  * signature of an operation it leaves alone.
  *
  * Two invariants hold of every materialized cast:
  *
  *   - `adaptor.operands(i).typ == convertType(op.operands(i).typ)`, so a
  *     pattern always builds on operands of the types it converts to, even
  *     where the defining operation was left unconverted.
  *   - a cast is inserted at the definition of the value it converts, not at
  *     its use, so that it dominates every use and can be shared by all of
  *     them.
  *
  * The root operation is not offered to patterns; its regions are converted.
  */
final class ConversionDriver(
    val typeConverter: TypeConverter,
    val patterns: Seq[ConversionPattern],
):

  private given TypeConverter = typeConverter

  // Materialized once, as GreedyRewritePatternApplier does: `patterns` may be
  // any Seq, and it is walked for every operation placed.
  private val patternArray = patterns.toArray

  /** Values that have been converted, keyed by the value they replace. */
  private val valueMap =
    mutable.Map.empty[Value[Attribute], Value[Attribute]]

  /** Blocks rebuilt with a converted signature, keyed by the block they
    * replace. Only holds blocks whose argument types actually changed.
    */
  private val blockMap = mutable.Map.empty[Block, Block]

  private val casts =
    mutable.Map.empty[(Value[Attribute], Attribute), Value[Attribute]]

  /** The last cast materialized for a value a given operation or block defines.
    * Without it every cast would go immediately after the definition, so a
    * definition needing several would emit them in reverse.
    *
    * The anchor is kept rather than an `InsertPoint`, which captures its
    * neighbour eagerly and would go stale as the block is swept.
    */
  private val lastCast = mutable.Map.empty[Operation | Block, Operation]

  def convert(root: Operation): Unit =
    root.regions.foreach(convertRegion(_, convertSignatures = false))

  /*≡==--==≡≡≡≡≡≡≡==--=≡≡*\
  ||   Materialization   ||
  \*≡==---==≡≡≡≡≡==---==≡*/

  /** The value `value` converted to `target`, casting if it is not already of
    * that type. One cast is shared by every use of a given value at a given
    * type.
    */
  private def materialize(
      value: Value[Attribute],
      target: Attribute,
  ): Value[Attribute] =
    if value.typ == target then value
    else
      casts.getOrElseUpdate(
        (value, target), {
          val cast = UnrealizedConversionCastOp(
            inputs = Seq(value),
            outputs = Seq(Result(target)),
          )
          // Inserted at the definition rather than at the use, so that the
          // cast dominates every use of `value` and all of them can share it.
          val owner: Operation | Block = value.owner match
            case Some(owner: Operation) if owner.containerBlock.isDefined =>
              owner
            case Some(block: Block) => block
            case _                  =>
              // Blocks are swept in reverse post-order, which is a linear
              // extension of dominance, and block arguments are mapped before
              // any block is swept. So a value is always defined - and placed -
              // before it is read, unless the input violates dominance.
              throw new Exception(
                s"Cannot convert a value to $target before it is defined; " +
                  "the input does not respect dominance."
              )

          lastCast.get(owner) match
            case Some(previous) => RewriteMethods.insertOpsAfter(previous, cast)
            case None           =>
              owner match
                case op: Operation => RewriteMethods.insertOpsAfter(op, cast)
                case block: Block  =>
                  RewriteMethods.insertOpsAt(InsertPoint.atStartOf(block), cast)
          lastCast(owner) = cast

          cast.outputs.head
        },
      )

  /*≡==--==≡≡≡==--=≡≡*\
  ||     Regions     ||
  \*≡==---==≡==---==≡*/

  /** Sweep `region`.
    *
    * Its blocks are given converted signatures only if the operation holding it
    * was itself converted - an operation no pattern converts keeps the regions
    * it declares, and the values crossing into them keep their types. This is
    * what keeps, say, an arithmetic conversion from rewriting the entry block
    * of a `func.func` whose `function_type` it does not convert.
    */
  private def convertRegion(region: Region, convertSignatures: Boolean): Unit =
    // Rebuild the blocks whose signature converts first, so that a value
    // defined by a block argument is mapped before any block is swept.
    if convertSignatures then
      for block <- region.blocks do
        val types = block.arguments.map(a => typeConverter.convertType(a.typ))
        if !types.corresponds(block.arguments)((t, a) => t == a.typ) then
          val fresh = Block(types, Seq.empty)
          blockMap(block) = fresh
          valueMap ++= block.arguments.zip(fresh.arguments)

    for block <- reversePostOrder(region) do
      sweep(block, blockMap.getOrElse(block, block))

    for block <- region.blocks.toSeq; fresh <- blockMap.get(block) do
      region.replaceBlock(block, fresh)

  /** The blocks of `region`, entry first, in reverse post-order of the CFG
    * their terminators describe. Blocks the entry cannot reach keep their
    * original relative order, after the ones it can.
    */
  private def reversePostOrder(region: Region): Seq[Block] =
    // The overwhelmingly common region - one straight-line block - needs none
    // of the machinery below.
    if region.blocks.length <= 1 then return region.blocks

    val visited = mutable.HashSet.empty[Block]
    val postOrder = mutable.ArrayBuffer.empty[Block]

    def visit(block: Block): Unit =
      if visited.add(block) then
        block.operations.lastOption.foreach(_.successors.foreach(visit))
        postOrder += block

    region.blocks.headOption.foreach(visit)

    if visited.size == region.blocks.length then postOrder.reverseIterator.toSeq
    else postOrder.reverseIterator.toSeq ++ region.blocks.filterNot(visited)

  /*≡==--==≡≡≡≡==--=≡≡*\
  ||      Blocks      ||
  \*≡==---==≡≡==---==≡*/

  /** Drain `source` into `target`, converting as we go.
    *
    * Operations are placed into a detached staging block and moved over at the
    * end, so that `source` and `target` may be the same block - which they are
    * whenever a block's signature did not convert.
    *
    * Each operation is detached just before it is placed, so the use lists of
    * the values it defines keep reflecting exactly those users that have not
    * been swept yet.
    */
  private def sweep(source: Block, target: Block): Unit =
    // Staging is only needed to drain a block into itself; otherwise the
    // target is empty and can be filled directly.
    val staging = if source eq target then Block() else target

    var pending = source.operations.headOption
    while pending.isDefined do
      val op = pending.get
      pending = op.next
      source.detachOp(op)
      place(op, staging)

    if staging ne target then
      RewriteMethods
        .moveOpsAt(InsertPoint.atEndOf(target), staging.operations.toSeq)

  /*≡==--==≡≡≡≡≡≡≡==--=≡≡*\
  ||     Operations      ||
  \*≡==---==≡≡≡≡≡==---==≡*/

  private def place(op: Operation, target: Block): Unit =
    // Unsupported: an operation branching to a block whose signature converted.
    //
    //   ^entry:
    //     cf.br ^bb1(%i : index)   // no pattern for cf.br
    //   ^bb1(%a : index):          // signature converted to (%a : i64)
    //
    // `%i` has to be converted *to* i64, where an ordinary operand of an
    // unconverted operation is converted *back to* index. `Operation.operands`
    // is flat, with no successor operand segmentation, so the two cannot be
    // told apart. TODO: close this in the IR - a successorOperands accessor
    // beside `Operation.successors`, which clair already models distinctly -
    // rather than by working around the throw here.
    if op.successors.exists(blockMap.contains) then
      throw new Exception(
        s"Cannot convert '${op.name}': it branches to a block whose signature " +
          "was converted, and successor operands cannot be identified."
      )

    val mapped = op.operands.map(v => valueMap.getOrElse(v, v))

    val adaptor = Adaptor(() =>
      op.operands.zip(mapped).map((operand, value) =>
        materialize(value, typeConverter.convertType(operand.typ))
      )
    )

    var converted: Option[ConversionResult] = None
    var index = 0
    while converted.isEmpty && index < patternArray.length do
      converted = patternArray(index).convert(op, adaptor)
      index += 1

    converted match
      case Some(result) => placeConverted(op, result, target)
      case None         => placeUnconverted(op, mapped, target)

  private def placeConverted(
      op: Operation,
      result: ConversionResult,
      target: Block,
  ): Unit =

    // A pattern erasing an operation whose results are still used throws:
    // there is no replacement to map those results to.
    //
    //   %0 = foo.a : index
    //   foo.b(%0)          // no pattern; still uses %0
    if result == PatternAction.Erase && op.results.exists(_.uses.nonEmpty) then
      throw new Exception(
        s"Cannot erase '${op.name}': its results are still used."
      )

    val (newOps, newResults) = result match
      case PatternAction.Erase => (Seq.empty, Seq.empty)
      case replacement         =>
        val (ops, results) = asReplacement(replacement)
        (ops, results.getOrElse(ops.lastOption.toSeq.flatMap(_.results)))

    // A pattern returning a different number of results than the operation
    // has throws too.
    if newResults.length != op.results.length then
      throw new Exception(
        s"Converting '${op.name}' expected ${op.results.length} new results " +
          s"but got ${newResults.length}"
      )

    target.addOps(newOps)
    valueMap ++= op.results.zip(newResults)

    // Erased unsafely: the replaced results outlive the operation defining them
    // as keys of `valueMap`, and as operands of the operations not yet swept.
    // `Owner.unapply` on one of them yields a detached operation.
    op.erase(safeErase = false)

    newOps
      .foreach(_.regions.foreach(convertRegion(_, convertSignatures = true)))

  private def placeUnconverted(
      op: Operation,
      mapped: Seq[Value[Attribute]],
      target: Block,
  ): Unit =
    val placed =
      if op.operands.lazyZip(mapped).forall(_ eq _) then op
      else
        val operands = op.operands.lazyZip(mapped)
          .map((operand, value) => materialize(value, operand.typ))
        // `results = op.results` keeps the results, and therefore their uses,
        // identical - so nothing has to be mapped for them.
        op.updated(operands = operands, results = op.results)

    target.addOp(placed)
    placed.regions.foreach(convertRegion(_, convertSignatures = false))
