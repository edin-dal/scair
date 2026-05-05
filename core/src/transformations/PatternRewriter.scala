package scair.transformations

import scair.helpers.*
import scair.ir.*

import scala.annotation.tailrec
import scala.collection.mutable.LinkedHashSet

// ██████╗░ ░█████╗░ ████████╗ ████████╗ ███████╗ ██████╗░ ███╗░░██╗
// ██╔══██╗ ██╔══██╗ ╚══██╔══╝ ╚══██╔══╝ ██╔════╝ ██╔══██╗ ████╗░██║
// ██████╔╝ ███████║ ░░░██║░░░ ░░░██║░░░ █████╗░░ ██████╔╝ ██╔██╗██║
// ██╔═══╝░ ██╔══██║ ░░░██║░░░ ░░░██║░░░ ██╔══╝░░ ██╔══██╗ ██║╚████║
// ██║░░░░░ ██║░░██║ ░░░██║░░░ ░░░██║░░░ ███████╗ ██║░░██║ ██║░╚███║
// ╚═╝░░░░░ ╚═╝░░╚═╝ ░░░╚═╝░░░ ░░░╚═╝░░░ ╚══════╝ ╚═╝░░╚═╝ ╚═╝░░╚══╝

// ██████╗░ ███████╗ ██╗░░░░░░░██╗ ██████╗░ ██╗ ████████╗ ███████╗ ██████╗░
// ██╔══██╗ ██╔════╝ ██║░░██╗░░██║ ██╔══██╗ ██║ ╚══██╔══╝ ██╔════╝ ██╔══██╗
// ██████╔╝ █████╗░░ ╚██╗████╗██╔╝ ██████╔╝ ██║ ░░░██║░░░ █████╗░░ ██████╔╝
// ██╔══██╗ ██╔══╝░░ ░████╔═████║░ ██╔══██╗ ██║ ░░░██║░░░ ██╔══╝░░ ██╔══██╗
// ██║░░██║ ███████╗ ░╚██╔╝░╚██╔╝░ ██║░░██║ ██║ ░░░██║░░░ ███████╗ ██║░░██║
// ╚═╝░░╚═╝ ╚══════╝ ░░╚═╝░░░╚═╝░░ ╚═╝░░╚═╝ ╚═╝ ░░░╚═╝░░░ ╚══════╝ ╚═╝░░╚═╝

/*≡==--==≡≡≡==--=≡≡*\
||   Utils realm   ||
\*≡==---==≡==---==≡*/

object InsertPoint:

  def before(op: Operation) =
    op.containerBlock match
      case None =>
        throw new Exception(
          "Operation insertion point must have a parent block."
        )
      case Some(block) => InsertPoint(block, Some(op))

  def after(op: Operation) =
    op.containerBlock match
      case None =>
        throw new Exception(
          "Operation insertion point must have a parent block."
        )
      case Some(block) =>
        InsertPoint(block, op.next)

  def atStartOf(block: Block) =
    InsertPoint(block, block.operations.headOption)

  def atEndOf(block: Block) = InsertPoint(block)

case class InsertPoint(
    val block: Block,
    val insertBefore: Option[Operation] = None,
)

/*≡==--==≡≡≡≡==--=≡≡*\
||   Static realm   ||
\*≡==---==≡≡==---==≡*/

trait Rewriter:

  def operationRemovalHandler: Operation => Unit = (op: Operation) =>
    // default handler does nothing
    // apart from invalidating the block's op order :D
    op.containerBlock match
      case Some(block) =>
        block.isOpOrderValid = false
      case None => ()

  def operationInsertionHandler: (Operation) => Unit = (
    op: Operation
  ) =>
    // default handler does nothing
    // apart from invalidating the block's op order :D
    op.containerBlock match
      case Some(block) =>
        block.isOpOrderValid = false
      case None => ()

  def eraseOp(op: Operation, safeErase: Boolean = true) =
    op.containerBlock match
      case Some(block) =>
        operationRemovalHandler(op)
        block.eraseOp(op, safeErase)
      case _ =>
        throw new Exception("Cannot erase an operation that has no parents.")

  def insertOpsAt(
      insertionPoint: InsertPoint,
      ops: Operation | Seq[Operation],
  ): Unit =

    val operations = ops match
      case x: Operation => Seq(x)
      case y: Seq[?]    => y.asInstanceOf[Seq[Operation]]

    insertionPoint.insertBefore match
      case Some(op) =>
        insertionPoint.block.insertOpsBefore(
          op,
          operations,
        )
      case None =>
        insertionPoint.block.addOps(operations)

    operations.foreach(operationInsertionHandler)

  def insertOpsBefore(
      op: Operation,
      newOps: Operation | Seq[Operation],
  ): Unit =
    insertOpsAt(InsertPoint.before(op), newOps)

  def insertOpsAfter(
      op: Operation,
      newOps: Operation | Seq[Operation],
  ): Unit =
    insertOpsAt(InsertPoint.after(op), newOps)

  def replaceOp(
      op: Operation,
      newOps: Operation | Seq[Operation],
      newResults: Option[Seq[Value[Attribute]]] = None,
  ): Unit =

    val block = op.containerBlock match
      case Some(x) => x
      case None    =>
        throw new Exception("Cannot replace an operation without a parent")

    val ops = newOps match
      case x: Operation => Seq(x)
      case y: Seq[?]    => y.asInstanceOf[Seq[Operation]]

    val results = newResults match
      case Some(x) => x
      case None    =>
        if ops.length == 0 then ListType() else ops.last.results

    if op.results.length != results.length then
      throw new Exception(
        s"Expected ${op.results.length} new results but got ${results.length}"
      )

    RewriteMethods.insertOpsBefore(op, ops)

    for (old_res, new_res) <- (op.results zip results) do
      replaceValue(old_res, new_res)

    RewriteMethods.eraseOp(op, safeErase = false)
    operationRemovalHandler(op)
    ops.foreach(operationInsertionHandler)

  def replaceValue(
      value: Value[Attribute],
      newValue: Value[Attribute],
  ): Unit =
    if !(newValue eq value) then
      for (op, uses) <- value.uses.groupBy(_.operation) do
        // TODO: This should be enforced by a nicer design!
        if op.containerBlock.nonEmpty then
          val indices = Set.from(uses.map(_.index))
          val newOperands = op.operands
            .mapWithIndex((v, i) => if indices.contains(i) then newValue else v)
          val newOp =
            op.updated(
              results = op.results,
              operands = newOperands,
            )
          replaceOp(
            op = op,
            newOps = newOp,
            newResults = Some(newOp.results),
          )

object RewriteMethods extends Rewriter

/*≡==--==≡≡≡≡==--=≡≡*\
||  Abstract realm  ||
\*≡==---==≡≡==---==≡*/

//             OPERATION REWRITER              //

type PatternRewriter = PatternRewriteWalker#PatternRewriter

abstract class RewritePattern:

  def matchAndRewrite(op: Operation, rewriter: PatternRewriter): Unit

case class GreedyRewritePatternApplier(patterns: Seq[RewritePattern])
    extends RewritePattern:

  @tailrec
  private final def matchAndRewriteRec(
      op: Operation,
      rewriter: PatternRewriter,
      patterns: Seq[RewritePattern],
  ): Unit =
    patterns match
      case Nil    => ()
      case h :: t =>
        h.matchAndRewrite(op, rewriter)
        if !rewriter.hasDoneAction then matchAndRewriteRec(op, rewriter, t)

  override def matchAndRewrite(
      op: Operation,
      rewriter: PatternRewriter,
  ): Unit =
    matchAndRewriteRec(op, rewriter, patterns)

//    OPERATION REWRITE WALKER    //
class PatternRewriteWalker(
    val pattern: RewritePattern
):

  class PatternRewriter(
      var currentOp: Operation
  ) extends Rewriter:
    var hasDoneAction: Boolean = false

    override def operationRemovalHandler: Operation => Unit =
      (op: Operation) =>
        // here the logic is simple - we invalidate the op order every time an operation is removed from the block
        op.containerBlock match
          case Some(block) =>
            block.isOpOrderValid = false
          case None => ()
        clearWorklist(op)
        op.operands.foreach((o) =>
          o.owner match
            case Some(owner: Operation) => populateWorklist(owner)
            case _                      => ()
        )

    override def operationInsertionHandler: Operation => Unit =
      (op: Operation) =>
        // similarly, we invalidate the op order every time an operation is added from the block
        op.containerBlock match
          case Some(block) => block.isOpOrderValid = false
          case None        => ()
        populateWorklist(op)

    def eraseOp(op: Operation): Unit =
      super.eraseOp(op)

    def eraseMatchedOp(): Unit =
      super.eraseOp(currentOp)

    def insertOpAtLocation(
        insertionPoint: InsertPoint,
        ops: Operation | Seq[Operation],
    ): Unit =
      super.insertOpsAt(insertionPoint, ops)

    def insertOpBeforeMatchedOp(
        ops: Operation | Seq[Operation]
    ): Unit =
      super.insertOpsBefore(currentOp, ops)

    def insertOpAfterMatchedOp(
        ops: Operation | Seq[Operation]
    ): Unit =
      super.insertOpsBefore(currentOp, ops)

    def insertOpAtEndOf(
        block: Block,
        ops: Operation | Seq[Operation],
    ): Unit =
      insertOpAtLocation(InsertPoint.atEndOf(block), ops)

    def insertOpAtStartOf(
        block: Block,
        ops: Operation | Seq[Operation],
    ): Unit =
      insertOpAtLocation(InsertPoint.atStartOf(block), ops)

    override def insertOpsBefore(
        op: Operation,
        newOps: Operation | Seq[Operation],
    ): Unit =
      super.insertOpsBefore(op, newOps)
      hasDoneAction = true

    override def insertOpsAfter(
        op: Operation,
        newOps: Operation | Seq[Operation],
    ): Unit =
      super.insertOpsAfter(op, newOps)
      hasDoneAction = true

    override def replaceOp(
        op: Operation,
        newOps: Operation | Seq[Operation],
        newResults: Option[Seq[Value[Attribute]]] = None,
    ): Unit =
      super.replaceOp(op, newOps, newResults)
      hasDoneAction = true

  private val worklist: LinkedHashSet[Operation] = LinkedHashSet.empty

  def rewrite(op: Operation): Unit =
    populateWorklist(op)
    return processWorklist()

  def rewrite(block: Block): Unit =
    populateWorklist(block)
    return processWorklist()

  def rewrite(region: Region): Unit =
    populateWorklist(region)
    return processWorklist()

  private def popWorklist: Operation =
    val op = worklist.head
    worklist.remove(op)
    op

  inline private def populateWorklist(region: Region): Unit =
    region.blocks.foreach(populateWorklist)

  inline private def populateWorklist(block: Block): Unit =
    block.operations.foreach(populateWorklist)

  private def populateWorklist(op: Operation): Unit =
    worklist += op
    op.regions.foreach(populateWorklist)

  private def clearWorklist(op: Operation): Unit =
    worklist -= op
    op.regions.foreach((x: Region) =>
      x.blocks.foreach((y: Block) => y.operations.foreach(clearWorklist(_)))
    )

  private def processWorklist(): Boolean =

    var rewriter_done_action = false

    if worklist.isEmpty then return rewriter_done_action

    var op = popWorklist
    val rewriter = PatternRewriter(op)

    while true do
      rewriter.hasDoneAction = false
      rewriter.currentOp = op

      try pattern.matchAndRewrite(op, rewriter)
      catch
        case e: Exception => throw e // throw new Exception("Caught exception!")

      rewriter_done_action |= rewriter.hasDoneAction

      if worklist.isEmpty then return rewriter_done_action

      op = popWorklist
    return rewriter_done_action
