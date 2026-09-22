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
    val op: Operation,
    private val materialise: () => Seq[Value[Attribute]],
):

  lazy val operands: Seq[Value[Attribute]] = materialise()

  def apply(index: Int): Value[Attribute] = operands(index)

  /** The matched operation with its operands converted.
    *
    * Only its operands are meaningful: its results are fresh values of the
    * unconverted types, and it is not attached to any block.
    */
  lazy val remapped: Operation = op.updated(operands = operands)

/** The adaptor in scope, for use in the body of a conversion pattern. */
def adaptor(using a: Adaptor): Adaptor = a

type ConversionResult = PatternAction | Operation | Seq[Operation] |
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
      ConversionResult,
    ]
): ConversionPattern =
  object conversionPattern extends ConversionPattern:
    override def convert(op: Operation, adaptor: Adaptor)(using
        converter: TypeConverter
    ): Option[ConversionResult] =
      partial(using converter, adaptor).lift(op) match
        case Some(PatternAction.Abort) => None
        case other                     => other

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

  /** Values that have been converted, keyed by the value they replace. */
  private val valueMap =
    mutable.Map.empty[Value[Attribute], Value[Attribute]]

  /** Blocks rebuilt with a converted signature, keyed by the block they
    * replace. Only holds blocks whose argument types actually changed.
    */
  private val blockMap = mutable.Map.empty[Block, Block]

  private val casts =
    mutable.Map.empty[(Value[Attribute], Attribute), Value[Attribute]]

  /** The last cast materialized at the start of a block, so that the casts of
    * that block's arguments read in the order they were materialized rather
    * than in reverse.
    */
  private val lastArgumentCast = mutable.Map.empty[Block, Operation]

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
          // Inserted at the definition rather than at the use, so that the cast
          // dominates every use of `value` and all of them can share it.
          value.owner match
            case Some(owner: Operation) if owner.containerBlock.isDefined =>
              RewriteMethods.insertOpsAfter(owner, cast)
            case Some(block: Block) =>
              lastArgumentCast.get(block) match
                case Some(previous) =>
                  RewriteMethods.insertOpsAfter(previous, cast)
                case None =>
                  RewriteMethods.insertOpsAt(InsertPoint.atStartOf(block), cast)
              lastArgumentCast(block) = cast
            case _ =>
              // Blocks are swept in reverse post-order, which is a linear
              // extension of dominance, and block arguments are mapped before
              // any block is swept. So a value is always defined - and placed -
              // before it is read, unless the input violates dominance.
              throw new Exception(
                s"Cannot convert a value to $target before it is defined; " +
                  "the input does not respect dominance."
              )
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
    val rebuilt =
      for
        block <- if convertSignatures then region.blocks.toSeq else Seq.empty
        types = block.arguments.map(a => typeConverter.convertType(a.typ)).toSeq
        if types != block.arguments.map(_.typ).toSeq
      yield
        val fresh = Block(types, Seq.empty)
        blockMap(block) = fresh
        valueMap ++= block.arguments.zip(fresh.arguments)
        (block, fresh)

    for block <- reversePostOrder(region) do
      sweep(block, blockMap.getOrElse(block, block))

    for (block, fresh) <- rebuilt do region.replaceBlock(block, fresh)

  /** The blocks of `region`, entry first, in reverse post-order of the CFG
    * their terminators describe. Blocks the entry cannot reach keep their
    * original relative order, after the ones it can.
    */
  private def reversePostOrder(region: Region): Seq[Block] =
    val visited = mutable.LinkedHashSet.empty[Block]
    val postOrder = mutable.ArrayBuffer.empty[Block]

    def visit(block: Block): Unit =
      if visited.add(block) then
        block.operations.lastOption.toSeq.flatMap(_.successors).foreach(visit)
        postOrder += block

    region.blocks.headOption.foreach(visit)
    postOrder.reverse.toSeq ++ region.blocks.filterNot(visited.contains)

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
    val staging = Block()

    var pending = source.operations.headOption
    while pending.isDefined do
      val op = pending.get
      pending = op.next
      source.detachOp(op)
      place(op, staging)

    val placed = staging.operations.toSeq
    placed.foreach(staging.detachOp)
    target.addOps(placed)

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
    // told apart.
    if op.successors.exists(blockMap.contains) then
      throw new Exception(
        s"Cannot convert '${op.name}': it branches to a block whose signature " +
          "was converted, and successor operands cannot be identified."
      )

    val mapped = op.operands.map(v => valueMap.getOrElse(v, v))

    val adaptor = Adaptor(
      op,
      () =>
        op.operands.zip(mapped).map((operand, value) =>
          materialize(value, typeConverter.convertType(operand.typ))
        ),
    )

    // The operation's regions are detached before its patterns run, so that a
    // pattern can move one onto the operation replacing it:
    //
    //   case op: func.Func => llvm.Func(op.sym_name, op.function_type, ..., op.body)
    //
    // `Operation.attachRegion` throws on a region that still has a
    // containerOperation. They are attached back if no pattern converts the
    // operation.
    op.regions.foreach(_.detached)

    patterns.view.flatMap(_.convert(op, adaptor)).headOption match
      case Some(result) => placeConverted(op, result, target)
      case None         => placeUnconverted(op, mapped, target)

  private def placeConverted(
      op: Operation,
      result: ConversionResult,
      target: Block,
  ): Unit =

    val (newOps, newResults) = result match
      case PatternAction.Erase =>
        (Seq.empty[Operation], Seq.empty[Value[Attribute]])
      case PatternAction.Abort =>
        throw new Exception("Unreachable: aborted patterns do not convert.")
      case (ops, results): (Operation | Seq[Operation], ?) =>
        (
          normalize(ops),
          results match
            case r: Value[?]       => Seq(r.asInstanceOf[Value[Attribute]])
            case rs: Seq[Value[?]] => rs.asInstanceOf[Seq[Value[Attribute]]],
        )
      case ops: (Operation | Seq[Operation]) =>
        val normalized = normalize(ops)
        (
          normalized,
          normalized.lastOption.map(_.results.map(r => r: Value[Attribute]))
            .getOrElse(Seq.empty),
        )

    // A pattern erasing an operation whose results are still used throws: there
    // is no replacement to map those results to.
    //
    //   %0 = foo.a : index
    //   foo.b(%0)          // no pattern; still uses %0
    //
    // A pattern returning a different number of results than the operation has
    // throws too.
    if newOps.isEmpty && newResults.isEmpty then
      if op.results.exists(_.uses.nonEmpty) then
        throw new Exception(
          s"Cannot erase '${op.name}': its results are still used."
        )
    else if newResults.length != op.results.length then
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
    if op.operands.zip(mapped).forall((operand, value) => operand eq value)
    then
      op.regions.foreach(op.attachRegion)
      target.addOp(op)
      op.regions.foreach(convertRegion(_, convertSignatures = false))
    else
      val operands = op.operands.zip(mapped)
        .map((operand, value) => materialize(value, operand.typ))
      // `results = op.results` keeps the results, and therefore their uses,
      // identical - so nothing has to be mapped for them.
      val newOp = op.updated(operands = operands, results = op.results)
      target.addOp(newOp)
      newOp.regions.foreach(convertRegion(_, convertSignatures = false))

  private def normalize(ops: Operation | Seq[Operation]): Seq[Operation] =
    ops match
      case op: Operation => Seq(op)
      case ops: Seq[?]   => ops.asInstanceOf[Seq[Operation]]
