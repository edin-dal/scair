package scair.transformations

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

object InsertPoint {

  def before(op: Operation) =
    op.container_block match
      case None =>
        throw new Exception(
          "Operation insertion point must have a parent block."
        )
      case Some(block) => InsertPoint(block, Some(op))

  def after(op: Operation) =
    op.container_block match
      case None =>
        throw new Exception(
          "Operation insertion point must have a parent block."
        )
      case Some(block) =>
        InsertPoint(block, op.next)

  def at_start_of(block: Block) =
    InsertPoint(block, block.operations.headOption)

  def at_end_of(block: Block) = InsertPoint(block)
}

case class InsertPoint(
    val block: Block,
    val insert_before: Option[Operation] = None
)

/*≡==--==≡≡≡≡==--=≡≡*\
||   Static realm   ||
\*≡==---==≡≡==---==≡*/

trait Rewriter {

  def operation_removal_handler: Operation => Unit = (op: Operation) => {
    // default handler does nothing
  }

  def operation_insertion_handler: (Operation) => Unit = (
    op: Operation
  ) => {
    // default handler does nothing
  }

  def erase_op(op: Operation, safe_erase: Boolean = true) = {
    op.container_block match {
      case Some(block) =>
        operation_removal_handler(op)
        block.erase_op(op, safe_erase)
      case _ =>
        throw new Exception("Cannot erase an operation that has no parents.")
    }
  }

  def insert_ops_at(
      insertion_point: InsertPoint,
      ops: Operation | Seq[Operation]
  ): Unit = {

    val operations = ops match {
      case x: Operation => Seq(x)
      case y: Seq[_]    => y.asInstanceOf[Seq[Operation]]
    }

    insertion_point.insert_before match {
      case Some(op) =>
        insertion_point.block.insert_ops_before(
          op,
          operations
        )
      case None =>
        insertion_point.block.add_ops(operations)
    }

    operations.foreach(operation_insertion_handler)
  }

  def insert_ops_before(
      op: Operation,
      new_ops: Operation | Seq[Operation]
  ): Unit = {
    insert_ops_at(InsertPoint.before(op), new_ops)
  }

  def insert_ops_after(
      op: Operation,
      new_ops: Operation | Seq[Operation]
  ): Unit = {
    insert_ops_at(InsertPoint.after(op), new_ops)
  }

  def replace_op(
      op: Operation,
      new_ops: Operation | Seq[Operation],
      new_results: Option[Seq[Value[Attribute]]] = None
  ): Unit = {

    val block = op.container_block match {
      case Some(x) => x
      case None    =>
        throw new Exception("Cannot replace an operation without a parent")
    }

    val ops = new_ops match {
      case x: Operation => Seq(x)
      case y: Seq[_]    => y.asInstanceOf[Seq[Operation]]
    }

    val results = new_results match {
      case Some(x) => x
      case None    =>
        if (ops.length == 0) then ListType() else ops.last.results
    }

    if (op.results.length != results.length) then {
      throw new Exception(
        s"Expected ${op.results.length} new results but got ${results.length}"
      )
    }

    RewriteMethods.insert_ops_before(op, ops)

    for ((old_res, new_res) <- (op.results zip results)) {
      replace_value(old_res, new_res)
    }

    RewriteMethods.erase_op(op, safe_erase = false)
    operation_removal_handler(op)
    ops.foreach(operation_insertion_handler)
  }

  def replace_value(
      value: Value[Attribute],
      new_value: Value[Attribute]
  ): Unit = {
    if !(new_value eq value) then {
      for ((op, uses) <- value.uses.groupBy(_.operation)) {
        // TODO: This should be enforced by a nicer design!
        if op.container_block.nonEmpty then {
          val indices = Set.from(uses.map(_.index))
          val new_operands = op.operands.zipWithIndex.map((v, i) =>
            if indices.contains(i) then new_value else v
          )
          val new_op =
            op.updated(
              results = op.results,
              operands = new_operands
            )
          replace_op(
            op = op,
            new_ops = new_op,
            new_results = Some(new_op.results)
          )
        }
      }
    }
  }

}

object RewriteMethods extends Rewriter

/*≡==--==≡≡≡≡==--=≡≡*\
||  Abstract realm  ||
\*≡==---==≡≡==---==≡*/

//             OPERATION REWRITER              //

type PatternRewriter = PatternRewriteWalker#PatternRewriter

abstract class RewritePattern {

  def match_and_rewrite(op: Operation, rewriter: PatternRewriter): Unit

}

case class GreedyRewritePatternApplier(patterns: Seq[RewritePattern])
    extends RewritePattern {

  @tailrec
  private final def match_and_rewrite_rec(
      op: Operation,
      rewriter: PatternRewriter,
      patterns: Seq[RewritePattern]
  ): Unit = {
    patterns match
      case Nil    => ()
      case h :: t =>
        h.match_and_rewrite(op, rewriter)
        if !rewriter.has_done_action then match_and_rewrite_rec(op, rewriter, t)
  }

  override def match_and_rewrite(
      op: Operation,
      rewriter: PatternRewriter
  ): Unit =
    match_and_rewrite_rec(op, rewriter, patterns)

}

//    OPERATION REWRITE WALKER    //
class PatternRewriteWalker(
    val pattern: RewritePattern
) {

  class PatternRewriter(
      var current_op: Operation
  ) extends Rewriter {
    var has_done_action: Boolean = false

    override def operation_removal_handler: Operation => Unit =
      (op: Operation) =>
        clear_worklist(op)
        op.operands.foreach((o) =>
          o.owner match
            case Some(owner: Operation) => populate_worklist(owner)
            case _                      => ()
        )

    override def operation_insertion_handler: Operation => Unit =
      populate_worklist

    def erase_op(op: Operation): Unit = {
      super.erase_op(op)
    }

    def erase_matched_op(): Unit = {
      super.erase_op(current_op)
    }

    def insert_op_at_location(
        insertion_point: InsertPoint,
        ops: Operation | Seq[Operation]
    ): Unit = {
      super.insert_ops_at(insertion_point, ops)
    }

    def insert_op_before_matched_op(
        ops: Operation | Seq[Operation]
    ): Unit = {
      super.insert_ops_before(current_op, ops)
    }

    def insert_op_after_matched_op(
        ops: Operation | Seq[Operation]
    ): Unit = {
      super.insert_ops_before(current_op, ops)
    }

    def insert_op_at_end_of(
        block: Block,
        ops: Operation | Seq[Operation]
    ): Unit = {
      insert_op_at_location(InsertPoint.at_end_of(block), ops)
    }

    def insert_op_at_start_of(
        block: Block,
        ops: Operation | Seq[Operation]
    ): Unit = {
      insert_op_at_location(InsertPoint.at_start_of(block), ops)
    }

    override def insert_ops_before(
        op: Operation,
        new_ops: Operation | Seq[Operation]
    ): Unit = {
      super.insert_ops_before(op, new_ops)
      has_done_action = true
    }

    override def insert_ops_after(
        op: Operation,
        new_ops: Operation | Seq[Operation]
    ): Unit = {
      super.insert_ops_after(op, new_ops)
      has_done_action = true
    }

    override def replace_op(
        op: Operation,
        new_ops: Operation | Seq[Operation],
        new_results: Option[Seq[Value[Attribute]]] = None
    ): Unit = {
      super.replace_op(op, new_ops, new_results)
      has_done_action = true
    }

  }

  private val worklist: LinkedHashSet[Operation] = LinkedHashSet.empty

  def rewrite(op: Operation): Unit =
    populate_worklist(op)
    return process_worklist()

  def rewrite(block: Block): Unit =
    populate_worklist(block)
    return process_worklist()

  def rewrite(region: Region): Unit =
    populate_worklist(region)
    return process_worklist()

  private def pop_worklist: Operation =
    val op = worklist.head
    worklist.remove(op)
    op

  inline private def populate_worklist(region: Region): Unit =
    region.blocks.foreach(populate_worklist)

  inline private def populate_worklist(block: Block): Unit =
    block.operations.foreach(populate_worklist)

  private def populate_worklist(op: Operation): Unit = {
    worklist += op
    op.regions.foreach(populate_worklist)
  }

  private def clear_worklist(op: Operation): Unit = {
    worklist -= op
    op.regions.foreach((x: Region) =>
      x.blocks.foreach((y: Block) => y.operations.foreach(clear_worklist(_)))
    )
  }

  private def process_worklist(): Boolean = {

    var rewriter_done_action = false

    if (worklist.isEmpty) return rewriter_done_action

    var op = pop_worklist
    val rewriter = PatternRewriter(op)

    while (true) {
      rewriter.has_done_action = false
      rewriter.current_op = op

      try {
        pattern.match_and_rewrite(op, rewriter)
      } catch {
        case e: Exception => throw e // throw new Exception("Caught exception!")
      }

      rewriter_done_action |= rewriter.has_done_action

      if (worklist.isEmpty) return rewriter_done_action

      op = pop_worklist
    }
    return rewriter_done_action
  }

}
