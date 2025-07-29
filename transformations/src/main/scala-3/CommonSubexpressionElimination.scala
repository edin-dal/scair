package scair.transformations.cse

import scair.ir.*
import scair.transformations.*

import scala.collection.mutable.Map
import scala.collection.mutable.Set

case class OperationInfo(val op: Operation) {

  override def hashCode(): Int =
    (op.name, op.attributes, op.properties, op.results.typ, op.operands)
      .hashCode()

  override def equals(obj: Any): Boolean = obj match
    case OperationInfo(b: Operation) =>
      val a = this.op
      a.name == b.name &&
      a.attributes == b.attributes &&
      a.properties == b.properties &&
      a.operands == b.operands &&
      a.results.typ == b.results.typ &&
      // TODO: Should be structural equivalence!
      a.regions == b.regions
    case _ => false

}

given Conversion[Operation, OperationInfo] = OperationInfo.apply

case class CSE(
    val knownOps: Map[OperationInfo, Operation] =
      Map[OperationInfo, Operation](),
    val toErase: Set[Operation] = Set[Operation]()
)(using rewriter: Rewriter) {

  def simplify(op: Operation): Unit =
    op match
      case free: NoMemoryEffect =>
        knownOps.get(op) match
          case Some(known) =>
            (op.results zip known.results).foreach(rewriter.replace_value)
            toErase.add(op)
          case None => knownOps(op) = op
      case _ => ()

  def simplify(block: Block): Unit =
    // To modify the block during iteration
    0 until block.operations.size foreach { i =>
      val op = block.operations(i)
      op.regions.foreach(region => CSE().simplify(region))
      simplify(op)
    }
    toErase.foreach(rewriter.erase_op(_, true))

  def simplify(region: Region): Unit =
    region.blocks match
      case Seq(oneBlock) => CSE().simplify(oneBlock)

}

object CommonSubexpressionElimination extends ModulePass {
  override val name = "cse"

  override def transform(op: Operation): Operation =
    given Rewriter = RewriteMethods
    val c = CSE()
    op.regions.foreach(c.simplify)
    op

}
