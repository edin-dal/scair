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

/** Companion object for the Block class, providing simple `apply`s just
  * forwarding to constructors.
  */
object Block {

  def apply(
      arguments_types: Iterable[Attribute] | Attribute = Seq(),
      operations: Iterable[Operation] | Operation = Seq()
  ): Block = new Block(arguments_types, operations)

  def apply(operations: Iterable[Operation] | Operation): Block =
    new Block(
      operations
    )

  def apply(
      arguments_types: Iterable[Attribute],
      operations_expr: Iterable[Value[Attribute]] => Iterable[Operation]
  ): Block =
    new Block(arguments_types, operations_expr)

  def apply(
      arguments_types: Attribute,
      operations_expr: Value[Attribute] => Iterable[Operation]
  ): Block =
    new Block(arguments_types, operations_expr)

}

/** A basic block.
  *
  * @param arguments
  *   The list of owned block arguments.
  * @param operations
  *   The list of contained operations.
  */
case class Block private (
    val arguments: ListType[Value[Attribute]],
    val operations: ListType[Operation]
) extends IRNode {

  final override def parent: Option[Region] = container_region

  operations.foreach(attach_op)

  /** Constructs a Block instance with the given argument types and operations.
    *
    * @param arguments_types
    *   The types of the arguments, either as a single Attribute or an Iterable
    *   of Attributes.
    * @param operations
    *   The operations, either as a single MLIROperation or an Iterable of
    *   MLIROperations.
    */
  def this(
      arguments_types: Iterable[Attribute] | Attribute = Seq(),
      operations: Iterable[Operation] | Operation = Seq()
  ) =
    this(
      ListType.from((arguments_types match {
        case single: Attribute     => Seq(single)
        case multiple: Iterable[_] => multiple.asInstanceOf[Iterable[Attribute]]
      }).map(Value(_))),
      ListType.from((operations match {
        case single: Operation => Seq(single)
        case multiple: Iterable[_] =>
          multiple.asInstanceOf[Iterable[Operation]]
      }))
    )

  /** Private tupled constructor mirroring the private primary constructor. Only
    * here for readability of other auxiliary constructors and strange
    * constraints on their syntax.
    *
    * @param args
    *   A tuple containing the argument values and operations.
    */
  private def this(
      args: (
          Iterable[Value[Attribute]] | Value[Attribute],
          Iterable[Operation] | Operation
      )
  ) =
    this(
      ListType.from(args._1 match {
        case single: Value[Attribute] => Seq(single)
        case multiple: Iterable[_] =>
          multiple.asInstanceOf[Iterable[Value[Attribute]]]
      }),
      ListType.from(args._2 match {
        case single: Operation => Seq(single)
        case multiple: Iterable[_] =>
          multiple.asInstanceOf[Iterable[Operation]]
      })
    )

  /** Constructs a Block instance with the given operations and no block
    * arguments.
    *
    * @param operations
    *   The operations, either as a single MLIROperation or an Iterable of
    *   MLIROperations.
    */
  def this(operations: Iterable[Operation] | Operation) =
    this(Seq(), operations)

  /** Constructs a Block instance with the given argument type and a function to
    * generate operations given the created block argument.
    *
    * @param argument_type
    *   The type of the argument.
    * @param operations_expr
    *   A function creating the contained operation(s) given the block argument.
    */
  def this(
      argument_type: Iterable[Attribute],
      operations_expr: Iterable[Value[Attribute]] => Iterable[Operation] |
        Operation
  ) =
    this({
      val args = argument_type.map(Value(_))
      (args, operations_expr(args))
    })

  /** Constructs a Block instance with the given arguments type and function to
    * generate operations given the created block arguments.
    *
    * @param argument_type
    *   The types of the arguments as an Iterable of Attributes.
    * @param operations_expr
    *   A function creating the contained operation(s) given the block
    *   arguments.
    */
  def this(
      argument_type: Attribute,
      operations_expr: Value[Attribute] => Iterable[Operation] | Operation
  ) =
    this({
      val arg = Value(argument_type)
      (arg, operations_expr(arg))
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
    operations.insertAll(oplen, ListType(new_ops*))
  }

  def insert_op_before(
      existing_op: Operation,
      new_op: Operation
  ): Unit = {
    (existing_op.container_block `equals` Some(this)) match {
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
    (existing_op.container_block `equals` Some(this)) match {
      case true =>
        for (op <- new_ops) {
          attach_op(op)
        }
        operations.insertAll(getIndexOf(existing_op), ListType(new_ops*))
      case false =>
        throw new Exception(
          "Can't insert the new operation into the block, as the operation that was " +
            "given as a point of reference does not exist in the current block."
        )
    }
  }

  def insert_op_after(
      existing_op: Operation,
      new_op: Operation
  ): Unit = {
    (existing_op.container_block `equals` Some(this)) match {
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
    (existing_op.container_block `equals` Some(this)) match {
      case true =>
        for (op <- new_ops) {
          attach_op(op)
        }
        operations.insertAll(getIndexOf(existing_op) + 1, ListType(new_ops*))
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
    (op.container_block `equals` Some(this)) match {
      case true =>
        op.container_block = None
        operations -= op
        op
      case false =>
        throw new Exception(
          "MLIROperation can only be detached from a block in which it is contained."
        )
    }
  }

  def erase_op(op: Operation, safe_erase: Boolean = true) = {
    detach_op(op)
    op.erase(safe_erase)
  }

  def getIndexOf(op: Operation): Int = {
    operations.lastIndexOf(op) match {
      case -1 => throw new Exception("MLIROperation not present in the block.")
      case x  => x
    }

  }

  def verify(): Either[String, Unit] = {

    lazy val verifyRecArgs: Int => Either[String, Unit] = { (i: Int) =>
      if i == arguments.length then Right(())
      else
        arguments(i).verify() match {
          case Right(v)  => verifyRecArgs(i + 1)
          case Left(x) => Left(x)
        }
    }

    lazy val verifyRecOps: Int => Either[String, Unit] = { (i: Int) =>
      if i == operations.length then Right(())
      else
        operations(i).verify() match {
          case Right(v) =>
            operations(i) = v
            verifyRecOps(i + 1)
          case Left(x) => Left(x)
        }
    }

    verifyRecArgs(0) match
      case Right(_)  => verifyRecOps(0)
      case Left(x) => Left(x)
  }

  override def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }

}
