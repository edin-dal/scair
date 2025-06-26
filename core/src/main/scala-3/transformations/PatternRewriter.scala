package scair.transformations

import scair.dialects.builtin.ModuleOp
import scair.ir.*

import scala.annotation.tailrec
import scala.collection.mutable.Stack

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

  def before(op: Operation): InsertPoint = {
    (op.container_block == None) match {
      case true => throw new Exception(
          "Operation insertion point must have a parent block."
        )
      case false =>
        val block = op.container_block.get
        InsertPoint(block, Some(op))
    }
  }

  def after(op: Operation): InsertPoint = {
    (op.container_block == None) match {
      case true => throw new Exception(
          "Operation insertion point must have a parent block."
        )
      case false =>
        val block = op.container_block.get
        val opIdx = block.getIndexOf(op)
        (opIdx == block.operations.length - 1) match {
          case true  => new InsertPoint(block)
          case false => InsertPoint(block, Some(block.operations(opIdx + 1)))
        }
    }
  }

  def at_start_of(block: Block): InsertPoint = {
    val ops = block.operations
    ops.length match {
      case 0 => new InsertPoint(block)
      case _ => InsertPoint(block, Some(ops(0)))
    }
  }

  def at_end_of(block: Block): InsertPoint = { new InsertPoint(block) }

}

case class InsertPoint(val block: Block, val insert_before: Option[Operation]) {

  // custom constructor
  def this(block: Block) = { this(block, None) }

  if (insert_before != None) then {
    if !(insert_before.get.container_block `equals` Some(block)) then {
      throw new Error(
        "Given operation's container and given block do not match: " +
          "InsertPoint must be an operation inside a given block."
      )
    }
  }

}

/*≡==--==≡≡≡≡==--=≡≡*\
||   Static realm   ||
\*≡==---==≡≡==---==≡*/

trait Rewriter {

  def operation_removal_handler: Operation => Unit = (op: Operation) => {
    // default handler does nothing
  }

  def operation_insertion_handler: (Operation) => Unit = (op: Operation) => {
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
    val operations2 = Seq.iterableFactory
    insertion_point.insert_before match {
      case Some(op) => insertion_point.block.insert_ops_before(op, operations)
      case None     => insertion_point.block.add_ops(operations)
    }
    operations.foreach(operation_insertion_handler)
  }

  def insert_ops_before(
      op: Operation,
      new_ops: Operation | Seq[Operation]
  ): Unit = { insert_ops_at(InsertPoint.before(op), new_ops) }

  def insert_ops_after(
      op: Operation,
      new_ops: Operation | Seq[Operation]
  ): Unit = { insert_ops_at(InsertPoint.after(op), new_ops) }

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
      case None    => if (ops.length == 0) then ListType() else ops.last.results
    }

    if (op.results.length != results.length) then {
      throw new Exception(
        s"Expected ${op.results.length} new results but got ${results.length}"
      )
    }

    insert_ops_after(op, ops)

    for ((old_res, new_res) <- (op.results zip results)) {
      replace_value(old_res, new_res)
    }

    erase_op(op, safe_erase = false)
  }

  def replace_value(
      value: Value[Attribute],
      new_value: Value[Attribute]
  ): Unit = {
    if !(new_value eq value) then {
      for (use <- Seq.from(value.uses)) {
        val op = use.operation
        val new_op = op.updated(
          results = op.results,
          operands = op.operands.updated(use.index, new_value)
        )
        replace_op(
          op = op,
          new_ops = new_op,
          new_results = Some(new_op.results)
        )
      }
      value.uses.clear()
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

  def match_and_rewrite(op: Operation, rewriter: PatternRewriter): Unit = ???

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
  ): Unit = match_and_rewrite_rec(op, rewriter, patterns)

}

//    OPERATION REWRITE WALKER    //
class PatternRewriteWalker(val pattern: RewritePattern) {

  class PatternRewriter(var current_op: Operation) extends Rewriter {
    var has_done_action: Boolean = false

    override def operation_removal_handler: Operation => Unit = clear_worklist

    override def operation_insertion_handler: Operation => Unit =
      populate_worklist

    def erase_op(op: Operation): Unit = { super.erase_op(op) }

    def erase_matched_op(): Unit = { super.erase_op(current_op) }

    def insert_op_at_location(
        insertion_point: InsertPoint,
        ops: Operation | Seq[Operation]
    ): Unit = { super.insert_ops_at(insertion_point, ops) }

    def insert_op_before_matched_op(ops: Operation | Seq[Operation]): Unit = {
      super.insert_ops_before(current_op, ops)
    }

    def insert_op_after_matched_op(ops: Operation | Seq[Operation]): Unit = {
      super.insert_ops_before(current_op, ops)
    }

    def insert_op_at_end_of(
        block: Block,
        ops: Operation | Seq[Operation]
    ): Unit = { insert_op_at_location(InsertPoint.at_end_of(block), ops) }

    def insert_op_at_start_of(
        block: Block,
        ops: Operation | Seq[Operation]
    ): Unit = { insert_op_at_location(InsertPoint.at_start_of(block), ops) }

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

  private var worklist = Stack[Operation]()

  def rewrite_module(module: ModuleOp): Unit = { return rewrite_op(module) }

  def rewrite_op(op: Operation): Unit = {
    populate_worklist(op)
    var op_was_modified = process_worklist()

    return op_was_modified
  }

  private def populate_worklist(op: Operation): Unit = {
    worklist.push(op)
    op.regions.reverseIterator.foreach((x: Region) =>
      x.blocks.reverseIterator.foreach((y: Block) =>
        y.operations.reverseIterator.foreach(populate_worklist(_))
      )
    )
  }

  private def clear_worklist(op: Operation): Unit = {
    worklist -= op
    op.regions.reverseIterator.foreach((x: Region) =>
      x.blocks.reverseIterator.foreach((y: Block) =>
        y.operations.reverseIterator.foreach(clear_worklist(_))
      )
    )
  }

  private def process_worklist(): Boolean = {

    var rewriter_done_action = false

    if (worklist.length == 0) return rewriter_done_action

    var op = worklist.pop()
    val rewriter = PatternRewriter(op)

    while (true) {
      rewriter.has_done_action = false
      rewriter.current_op = op

      try { pattern.match_and_rewrite(op, rewriter) }
      catch {
        case e: Exception => throw e // throw new Exception("Caught exception!")
      }

      rewriter_done_action |= rewriter.has_done_action

      if (worklist.length == 0) return rewriter_done_action
      op = worklist.pop()
    }
    return rewriter_done_action
  }

}
