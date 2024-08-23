package scair.transformations

import scala.collection.mutable.Stack

import scair.dialects.builtin.ModuleOp
import scair._

// ==---------== //
//  Utils realm  //
// ==---------== //

object InsertPoint {

  def before(op: Operation): InsertPoint = {
    (op.container_block == None) match {
      case true =>
        throw new Exception(
          "Operation insertion point must have a parent block."
        )
      case false =>
        val block = op.container_block.get
        InsertPoint(block, Some(op))
    }
  }

  def after(op: Operation): InsertPoint = {
    (op.container_block == None) match {
      case true =>
        throw new Exception(
          "Operation insertion point must have a parent block."
        )
      case false =>
        val block = op.container_block.get
        val opIdx = block.getIndexOf(op)
        (opIdx == block.operations.length - 1) match {
          case true => new InsertPoint(block)
          case false =>
            InsertPoint(block, Some(block.operations(opIdx + 1)))
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

  def at_end_of(block: Block): InsertPoint = {
    new InsertPoint(block)
  }
}

case class InsertPoint(val block: Block, val insert_before: Option[Operation]) {

  // custom constructor
  def this(block: Block) = {
    this(block, None)
  }

  if (insert_before != None) then {
    if !(insert_before.get.container_block equals Some(block)) then {
      throw new Error(
        "Given operation's container and given block do not match: " +
          "InsertPoint must be an operation inside a given block."
      )
    }
  }
}

// ==----------== //
//  Static realm  //
// ==----------== //

object RewriteMethods {

  def erase_op(op: Operation) = {
    op.container_block match {
      case Some(block) =>
        block.erase_op(op)
      case _ =>
        throw new Exception("Cannot erase an operation that has no parents.")
    }
  }

  def insert_ops_at(
      insertion_point: InsertPoint,
      ops: Operation | Seq[Operation]
  ): Unit = {

    val operations = ops match {
      case x: Operation      => Seq(x)
      case y: Seq[Operation] => y
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
}

// ==------------== //
//  Abstract realm  //
// ==------------== //

//             OPERATION REWRITER              //
class PatternRewriter(
    var current_op: Operation
) {
  var has_done_action: Boolean = false

  def erase_op(op: Operation): Unit = {
    RewriteMethods.erase_op(op)
    has_done_action = true
  }

  def erase_matched_op(): Unit = {
    RewriteMethods.erase_op(current_op)
    has_done_action = true
  }

  def insert_op_at_location(
      insertion_point: InsertPoint,
      ops: Operation | Seq[Operation]
  ): Unit = {
    RewriteMethods.insert_ops_at(insertion_point, ops)
    has_done_action = true
  }

  def insert_op_before_matched_op(ops: Operation | Seq[Operation]): Unit = {
    RewriteMethods.insert_ops_before(current_op, ops)
    has_done_action = true
  }

  def insert_op_after_matched_op(ops: Operation | Seq[Operation]): Unit = {
    RewriteMethods.insert_ops_before(current_op, ops)
    has_done_action = true
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

  def insert_ops_before(
      op: Operation,
      new_ops: Operation | Seq[Operation]
  ): Unit = {
    RewriteMethods.insert_ops_before(op, new_ops)
    has_done_action = true
  }

  def insert_ops_after(
      op: Operation,
      new_ops: Operation | Seq[Operation]
  ): Unit = {
    RewriteMethods.insert_ops_after(op, new_ops)
    has_done_action = true
  }
}

abstract class RewritePattern {
  def match_and_rewrite(op: Operation, rewriter: PatternRewriter): Unit = ???
}

//    OPERATION REWRITE WALKER    //
class PatternRewriteWalker(
    val pattern: RewritePattern
) {

  private var worklist = Stack[Operation]()

  def rewrite_module(module: ModuleOp): Unit = {
    return rewrite_op(module)
  }

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

  private def process_worklist(): Boolean = {

    var rewriter_done_action = false

    if (worklist.length == 0) return rewriter_done_action

    var op = worklist.pop()
    val rewriter = PatternRewriter(op)

    while (true) {
      rewriter.has_done_action = false
      rewriter.current_op = op

      try {
        pattern.match_and_rewrite(op, rewriter)
      } catch {
        case e: Exception => throw new Exception("Caught exception!")
      }

      rewriter_done_action |= rewriter.has_done_action

      if (worklist.length == 0) return rewriter_done_action
      op = worklist.pop()
    }
    return rewriter_done_action
  }
}

// abstract class RewritePattern {
//   def match_and_rewrite(op: Operation) = {}
// }
