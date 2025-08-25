package scair.ir

import scair.ir.*
import scair.utils.IntrusiveList

object BlockOperations:

  def apply(elems: Operation*): BlockOperations =
    from(elems)

  def empty: BlockOperations = new BlockOperations

  def from(i: IterableOnce[Operation]) =
    val list = new BlockOperations
    list.addAll(i)

  def unapplySeq(list: BlockOperations): Some[Seq[Operation]] =
    Some(list.toSeq)

class BlockOperations extends IntrusiveList[Operation]:

  private final def handleOperationInsertion(op: Operation) =
    op.operands.zipWithIndex.foreach((o, i) => o.uses += Use(op, i))

  private final def handleOperationRemoval(op: Operation) =
    op.operands.zipWithIndex.foreach((o, i) =>
      o.uses.filterInPlace(_.operation != op)
    )

  override final def addOne(elem: Operation): this.type =
    super.addOne(elem)
    handleOperationInsertion(elem)
    this

  override final def prepend(elem: Operation): this.type =
    super.prepend(elem)
    handleOperationInsertion(elem)
    this

  override final def insert(v: Operation, elem: Operation): Unit =
    super.insert(v, elem)
    handleOperationInsertion(elem)

  override final def insertAll(
      idx: Operation,
      elems: IterableOnce[Operation]
  ): Unit =
    super.insertAll(idx, elems)
    elems.foreach(handleOperationInsertion)

  override final def remove(idx: Int): Operation =
    val op = super.remove(idx)
    handleOperationRemoval(op)
    op

  override final def subtractOne(elem: Operation): this.type =
    val op = super.subtractOne(elem)
    handleOperationRemoval(elem)
    this

  override final def update(v: Operation, elem: Operation): Unit =
    handleOperationRemoval(v)
    super.update(v, elem)
    handleOperationInsertion(elem)
