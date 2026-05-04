package scair.ir

import scair.collection.IntrusiveList
import scair.helpers.*
import scair.ir.*

//
// ██████╗░ ██╗░░░░░ ░█████╗░ ░█████╗░ ██╗░░██╗
// ██╔══██╗ ██║░░░░░ ██╔══██╗ ██╔══██╗ ██║░██╔╝
// ██████╦╝ ██║░░░░░ ██║░░██║ ██║░░╚═╝ █████═╝░
// ██╔══██╗ ██║░░░░░ ██║░░██║ ██║░░██╗ ██╔═██╗░
// ██████╦╝ ███████╗ ╚█████╔╝ ╚█████╔╝ ██║░╚██╗
// ╚═════╝░ ╚══════╝ ░╚════╝░ ░╚════╝░ ╚═╝░░╚═╝
//
// ░█████╗░ ██████╗░ ███████╗ ██████╗░ ░█████╗░ ████████╗ ██╗ ░█████╗░ ███╗░░██╗ ░██████╗
// ██╔══██╗ ██╔══██╗ ██╔════╝ ██╔══██╗ ██╔══██╗ ╚══██╔══╝ ██║ ██╔══██╗ ████╗░██║ ██╔════╝
// ██║░░██║ ██████╔╝ █████╗░░ ██████╔╝ ███████║ ░░░██║░░░ ██║ ██║░░██║ ██╔██╗██║ ╚█████╗░
// ██║░░██║ ██╔═══╝░ ██╔══╝░░ ██╔══██╗ ██╔══██║ ░░░██║░░░ ██║ ██║░░██║ ██║╚████║ ░╚═══██╗
// ╚█████╔╝ ██║░░░░░ ███████╗ ██║░░██║ ██║░░██║ ░░░██║░░░ ██║ ╚█████╔╝ ██║░╚███║ ██████╔╝
// ░╚════╝░ ╚═╝░░░░░ ╚══════╝ ╚═╝░░╚═╝ ╚═╝░░╚═╝ ░░░╚═╝░░░ ╚═╝ ░╚════╝░ ╚═╝░░╚══╝ ╚═════╝░
//

object BlockOperations:

  def apply(elems: Operation*): BlockOperations =
    from(elems)

  def empty: BlockOperations = new BlockOperations

  def from(i: IterableOnce[Operation]) =
    val list = new BlockOperations
    list.addAll(i)
    list.computeBlockOrder()

  def unapplySeq(list: BlockOperations): Some[Seq[Operation]] =
    Some(list.toSeq)

class BlockOperations extends IntrusiveList[Operation]:

  private inline def handleOperationInsertion(op: Operation) =
    op.operands.foreachWithIndex((o, i) => o.uses += Use(op, i))

  private inline def handleOperationRemoval(op: Operation) =
    op.operands
      .foreachWithIndex((o, i) => o.uses.filterInPlace(_.operation != op))

  final def computeBlockOrder(): this.type =
    var idx = 0
    this.foreach { op =>
      op.blockIndex = idx
      idx += 1
      op.recomputeOpOrder()
    }
    this

  override final def addOne(elem: Operation): this.type =
    handleOperationInsertion(elem)
    super.addOne(elem)

  override final def prepend(elem: Operation): this.type =
    handleOperationInsertion(elem)
    super.prepend(elem)

  override final def insert(v: Operation, elem: Operation): Unit =
    super.insert(v, elem)
    handleOperationInsertion(elem)

  override final def subtractOne(elem: Operation): this.type =
    handleOperationRemoval(elem)
    super.subtractOne(elem)

  override final def update(v: Operation, elem: Operation): Unit =
    handleOperationRemoval(v)
    super.update(v, elem)
    handleOperationInsertion(elem)
