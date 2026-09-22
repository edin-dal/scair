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

/** Attempts to convert an attribute using the enclosing type converter. */
type TypeConversionPattern =
  (Attribute, TypeConverter) => Option[Attribute]

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
  (attr, converter) => partial(using converter).lift(attr)

/** Maps attributes to the attributes they convert to, 1:1.
  *
  * `convertType` is total: an attribute no pattern matches converts to itself.
  */
final class TypeConverter(val patterns: Seq[TypeConversionPattern]):

  def convertType(attr: Attribute): Attribute =
    // Not `getOrElseUpdate`: a pattern may recurse into `convertType` for a
    // nested attribute, and reentering `getOrElseUpdate` is undefined.
    patterns.view.flatMap(pattern => pattern(attr, this)).headOption
          .getOrElse(attr)

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

/** Attempts to convert an operation using its converted operands and types. */
type ConversionPattern =
  (Operation, Adaptor, TypeConverter) => Option[ConversionResult]

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
  (op, adaptor, converter) =>
    partial(using converter, adaptor).lift(op) match
      case Some(PatternAction.Abort) => None
      case other => other.asInstanceOf[Option[ConversionResult]]

/*≡==--==≡≡≡≡==--=≡≡*\
||     Driver       ||
\*≡==---==≡≡==---==≡*/

/** A one-shot dialect conversion.
  *
  * Each region is converted in place, block by block: a block whose argument
  * types convert is given fresh arguments of the converted types, and every
  * operation is swept in turn. An operation is placed in one of three ways:
  *
  *   - converted by a pattern, from the converted operands the adaptor carries;
  *   - left as is, when no pattern converts it and none of its operands was
  *     converted - its results and their uses are left untouched;
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

  // Materialized once, as GreedyRewritePatternApplier does: `patterns` may be
  // any Seq, and it is walked for every operation placed.
  private val patternArray = patterns.toArray

  /** Values that have been converted, keyed by the value they replace. */
  private val valueMap =
    mutable.Map.empty[Value[Attribute], Value[Attribute]]

  /** Blocks given a converted signature. Only holds blocks whose argument
    * types actually changed.
    */
  private val retyped = mutable.HashSet.empty[Block]

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
              // Blocks are swept in depth-first pre-order, which is a linear
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
    // Convert every signature first, so that a value defined by a block
    // argument is mapped - and a branch to a retyped block is caught - before
    // any block is swept.
    if convertSignatures then
      for block <- region.blocks; i <- block.arguments.indices do
        val arg = block.arguments(i)
        val typ = typeConverter.convertType(arg.typ)
        if typ != arg.typ then
          val fresh = BlockArgument(typ)
          fresh.owner = Some(block)
          block.arguments(i) = fresh
          valueMap(arg) = fresh
          retyped += block

    for block <- dominanceOrder(region) do
      // Each operation's successor is taken before it is placed: whatever
      // placing it inserts goes before it, or after definitions already swept.
      var pending = block.operations.headOption
      while pending.isDefined do
        val op = pending.get
        pending = op.next
        place(op)

  /** The blocks of `region`, entry first, in depth-first pre-order of the CFG
    * their terminators describe - so every block comes after its dominators.
    * Blocks the entry cannot reach are ignored.
    */
  private def dominanceOrder(region: Region): Seq[Block] =
    // The overwhelmingly common region - one straight-line block - needs none
    // of the machinery below.
    if region.blocks.length <= 1 then return region.blocks

    val order = mutable.LinkedHashSet.empty[Block]

    def visit(block: Block): Unit =
      if order.add(block) then
        block.operations.lastOption.foreach(_.successors.foreach(visit))

    visit(region.blocks.head)
    order.toSeq

  /*≡==--==≡≡≡≡≡≡≡==--=≡≡*\
  ||     Operations      ||
  \*≡==---==≡≡≡≡≡==---==≡*/

  private def place(op: Operation): Unit =
    val mapped = op.operands.map(v => valueMap.getOrElse(v, v))

    val adaptor = Adaptor(() =>
      op.operands.zip(mapped).map((operand, value) =>
        materialize(value, typeConverter.convertType(operand.typ))
      )
    )

    var converted: Option[ConversionResult] = None
    var index = 0
    while converted.isEmpty && index < patternArray.length do
      converted = patternArray(index)(op, adaptor, typeConverter)
      index += 1

    converted match
      case Some(result) => placeConverted(op, result)
      case None         => placeUnconverted(op, mapped)

  private def placeConverted(op: Operation, result: ConversionResult): Unit =

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

    val block = op.containerBlock.get
    block.insertOpsBefore(op, newOps)
    valueMap ++= op.results.zip(newResults)

    // Erased unsafely: the replaced results outlive the operation defining them
    // as keys of `valueMap`, and as operands of the operations not yet swept.
    // `Owner.unapply` on one of them yields a detached operation.
    block.eraseOp(op, safeErase = false)

    newOps
      .foreach(_.regions.foreach(convertRegion(_, convertSignatures = true)))

  private def placeUnconverted(
      op: Operation,
      mapped: Seq[Value[Attribute]],
  ): Unit =
    // Unsupported: an unconverted operation branching to a retyped block.
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
    if op.successors.exists(retyped) then
      throw new Exception(
        s"Cannot convert '${op.name}': it branches to a block whose signature " +
          "was converted, and successor operands cannot be identified."
      )

    val placed =
      if op.operands.lazyZip(mapped).forall(_ eq _) then op
      else
        val operands = op.operands.lazyZip(mapped)
          .map((operand, value) => materialize(value, operand.typ))
        // `results = op.results` keeps the results, and therefore their uses,
        // identical - so nothing has to be mapped for them.
        val placed = op.updated(operands = operands, results = op.results)
        val block = op.containerBlock.get
        block.insertOpBefore(op, placed)
        block.detachOp(op)
        placed

    placed.regions.foreach(convertRegion(_, convertSignatures = false))
