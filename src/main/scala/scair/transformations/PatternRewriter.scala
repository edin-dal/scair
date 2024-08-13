package scair.transformations

import scala.collection.mutable.Stack

import scair.dialects.builtin.ModuleOp
import scair._

// ==----------== //
//  Static realm  //
// ==----------== //

object RewriteMethods {
  def erase_op(op: Operation, safe_erase: Boolean = true) = {
    op match {
      case x: ModuleOp =>
      case _ =>
        throw new Exception("Cannot erase an operation that has no parents.")
    }

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

object PR {

  var worklist = Stack[Operation]()

  def populate_worklist(op: Operation): Unit = {
    op.regions.reverseIterator.foreach((x: Region) =>
      x.blocks.reverseIterator.foreach((y: Block) =>
        y.operations.reverseIterator.foreach(worklist.push(_))
      )
    )
  }

  def main(args: Array[String]): Unit = {

    // val op1_1 = RegisteredOperation("op1")
    // val op1_2 = RegisteredOperation("op2")
    // val op2_1 = RegisteredOperation("op3")
    // val op2_2 = RegisteredOperation("op4")
    // val op3_1 = RegisteredOperation("op5")
    // val op3_2 = RegisteredOperation("op6")
    // val op4_1 = RegisteredOperation("op7")
    // val op4_2 = RegisteredOperation("op8")

    // val block1 = new Block(Seq(op1_1, op1_2))
    // val block2 = new Block(Seq(op2_1, op2_2))
    // val block3 = new Block(Seq(op3_1, op3_2))
    // val block4 = new Block(Seq(op4_1, op4_2))

    // val region1 = new Region(Seq(block1, block2))
    // val region2 = new Region(Seq(block3, block4))

    // val masterop =
    //   RegisteredOperation("master", regions = ListType(region1, region2))

    // populate_worklist(masterop)
    // println(worklist.pop().name)
    // println(worklist.pop().name)
    // println(worklist.pop().name)
    // println(worklist.pop().name)
    // println(worklist.pop().name)
    // println(worklist.pop().name)
    // println(worklist.pop().name)
    // println(worklist.pop().name)
    // println(worklist.length)
  }
}

// abstract class RewritePattern {
//   def match_and_rewrite(op: Operation) = {}
// }
