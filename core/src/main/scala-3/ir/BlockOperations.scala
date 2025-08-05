package scair.ir

// import scair.ir.*

opaque type BlockOperations <: collection.Seq[Operation] = ListType[Operation]

private def handleOperationInsertion(op: Operation) =
  op.operands.zipWithIndex.foreach((o, i) => o.uses += Use(op, i))

private def handleOperationRemoval(op: Operation) =
  op.operands.zipWithIndex.foreach((o, i) =>
    o.uses.filterInPlace(_.operation != op)
  )

extension (b: BlockOperations)

//   def foreach(f: Operation => Unit) = b.foreach(f)
//   def lastIndexOf(op: Operation): Int = b.lastIndexOf(op)
  def insertAll(index: Int, ops: IterableOnce[Operation]): Unit =
    b.insertAll(index, ops)
    ops.foreach(handleOperationInsertion)

//   def length: Int = b.length
//   def zipWithIndex: ListType[(Operation, Int)] = b.zipWithIndex
//   def foldLeft[A](z: A)(op: (A, Operation) => A): A =
//     b.foldLeft(z)(op)
  def update(i: Int, op: Operation): Unit =
    handleOperationRemoval(b(i))
    b.update(i, op)
    handleOperationInsertion(op)

  def -=(op: Operation): Unit =
    b -= op
    handleOperationRemoval(op)

  def ++=(ops: IterableOnce[Operation]): Unit =
    b ++= ops
    ops.foreach(handleOperationInsertion)

object BlockOperations:
  def unapplySeq(x: BlockOperations) = ListType.unapplySeq(x)

  def apply(ops: Operation*): BlockOperations =
    ops.foreach(handleOperationInsertion)
    ListType(ops*)

  def from(coll: collection.IterableOnce[Operation]): BlockOperations =
    coll.foreach(handleOperationInsertion)
    ListType.from(coll)

  def empty: BlockOperations = ListType.empty[Operation]
