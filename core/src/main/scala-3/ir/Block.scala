package scair.ir

// ██╗ ██████╗░
// ██║ ██╔══██╗
// ██║ ██████╔╝
// ██║ ██╔══██╗
// ██║ ██║░░██║
// ╚═╝ ╚═╝░░╚═╝

/*≡==--==≡≡≡≡==--=≡≡*\
||      BLOCKS      ||
\*≡==---==≡≡==---==≡*/

object Block {

  def apply(
      arguments_types: Iterable[Attribute] = Seq(),
      operations: Iterable[Operation] = Seq()
  ): Block = new Block(arguments_types, operations)

  def apply(operations: Iterable[Operation]): Block = new Block(operations)

  def apply(
      arguments_types: Iterable[Attribute],
      operations_expr: Iterable[Value[Attribute]] => Iterable[Operation]
  ): Block =
    new Block(arguments_types, operations_expr)

}

case class Block private (
    val arguments: ListType[Value[Attribute]],
    val operations: ListType[Operation]
) {

  //private tupled for other helpers
  private def this(args: (Iterable[Value[Attribute]], Iterable[Operation])) =
    this(ListType.from(args._1), ListType.from(args._2))

  def this(
      arguments_types: Iterable[Attribute] = Seq(),
      operations: Iterable[Operation] = Seq()
  ) =
    this(
      ListType.from(arguments_types.map(Value(_))),
      ListType.from(operations)
    )

  def this(operations: Iterable[Operation]) =
    this(Seq(), operations)


  def this(
      argument_types: Iterable[Attribute],
      operations_expr: Iterable[Value[Attribute]] => Iterable[Operation]
  ) =
    this({
      val args = argument_types.map(Value(_))
      (args, operations_expr(args))
    })

  var container_region: Option[Region] = None

  private def attach_op(op: Operation): Unit = {
    op.container_block match {
      case Some(x) =>
        throw new Exception(
          "Can't attach an operation already attached to a block."
        )
      case None =>
        op.is_ancestor(this) match {
          case true =>
            throw new Exception(
              "Can't add an operation to a block that is contained within that operation"
            )
          case false =>
            op.container_block = Some(this)
        }
    }
  }

  def add_op(new_op: Operation): Unit = {
    val oplen = operations.length
    attach_op(new_op)
    operations.insertAll(oplen, ListType(new_op))
  }

  def add_ops(new_ops: Seq[Operation]): Unit = {
    val oplen = operations.length
    for (op <- new_ops) {
      attach_op(op)
    }
    operations.insertAll(oplen, ListType(new_ops: _*))
  }

  def insert_op_before(existing_op: Operation, new_op: Operation): Unit = {
    (existing_op.container_block equals Some(this)) match {
      case true =>
        attach_op(new_op)
        operations.insertAll(getIndexOf(existing_op), ListType(new_op))
      case false =>
        throw new Exception(
          "Can't insert the new operation into the block, as the operation that was " +
            "given as a point of reference does not exist in the current block."
        )
    }
  }

  def insert_ops_before(
      existing_op: Operation,
      new_ops: Seq[Operation]
  ): Unit = {
    (existing_op.container_block equals Some(this)) match {
      case true =>
        for (op <- new_ops) {
          attach_op(op)
        }
        operations.insertAll(getIndexOf(existing_op), ListType(new_ops: _*))
      case false =>
        throw new Exception(
          "Can't insert the new operation into the block, as the operation that was " +
            "given as a point of reference does not exist in the current block."
        )
    }
  }

  def insert_op_after(existing_op: Operation, new_op: Operation): Unit = {
    (existing_op.container_block equals Some(this)) match {
      case true =>
        attach_op(new_op)
        operations.insertAll(getIndexOf(existing_op) + 1, ListType(new_op))
      case false =>
        throw new Exception(
          "Can't insert the new operation into the block, as the operation that was " +
            "given as a point of reference does not exist in the current block."
        )
    }
  }

  def insert_ops_after(
      existing_op: Operation,
      new_ops: Seq[Operation]
  ): Unit = {
    (existing_op.container_block equals Some(this)) match {
      case true =>
        for (op <- new_ops) {
          attach_op(op)
        }
        operations.insertAll(getIndexOf(existing_op) + 1, ListType(new_ops: _*))
      case false =>
        throw new Exception(
          "Can't insert the new operation into the block, as the operation that was " +
            "given as a point of reference does not exist in the current block."
        )
    }
  }

  def drop_all_references: Unit = {
    container_region = None
    for (op <- operations) op.drop_all_references
  }

  def detach_op(op: Operation): Operation = {
    (op.container_block equals Some(this)) match {
      case true =>
        op.container_block = None
        operations -= op
        return op
      case false =>
        throw new Exception(
          "Operation can only be detached from a block in which it is contained."
        )
    }
  }

  def erase_op(op: Operation) = {
    detach_op(op)
    op.erase()
  }

  def getIndexOf(op: Operation): Int = {
    operations.lastIndexOf(op) match {
      case -1 => throw new Exception("Operation not present in the block.")
      case x  => x
    }

  }

  def verify(): Unit = {
    for (op <- operations) op.verify()
    for (arg <- arguments) arg.verify()
  }

  override def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }

}
