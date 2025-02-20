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
      operations: Iterable[MLIROperation] | MLIROperation = Seq()
  ): Block = new Block(arguments_types, operations)

  def apply(operations: Iterable[MLIROperation] | MLIROperation): Block =
    new Block(
      operations
    )

  def apply(
      arguments_types: Iterable[Attribute],
      operations_expr: Iterable[Value[Attribute]] => Iterable[MLIROperation]
  ): Block =
    new Block(arguments_types, operations_expr)

  def apply(
      arguments_types: Attribute,
      operations_expr: Value[Attribute] => Iterable[MLIROperation]
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
    val operations: ListType[MLIROperation]
) {

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
      operations: Iterable[MLIROperation] | MLIROperation = Seq()
  ) =
    this(
      ListType.from((arguments_types match {
        case single: Attribute             => Seq(single)
        case multiple: Iterable[Attribute] => multiple
      }).map(Value(_))),
      ListType.from((operations match {
        case single: MLIROperation             => Seq(single)
        case multiple: Iterable[MLIROperation] => multiple
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
          Iterable[MLIROperation] | MLIROperation
      )
  ) =
    this(
      ListType.from(args._1 match {
        case single: Value[Attribute]             => Seq(single)
        case multiple: Iterable[Value[Attribute]] => multiple
      }),
      ListType.from(args._2 match {
        case single: MLIROperation             => Seq(single)
        case multiple: Iterable[MLIROperation] => multiple
      })
    )

  /** Constructs a Block instance with the given operations and no block
    * arguments.
    *
    * @param operations
    *   The operations, either as a single MLIROperation or an Iterable of
    *   MLIROperations.
    */
  def this(operations: Iterable[MLIROperation] | MLIROperation) =
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
      operations_expr: Iterable[Value[Attribute]] => Iterable[MLIROperation] |
        MLIROperation
  ) =
    this({
      val args = argument_type.map(Value(_))
      (args, operations_expr(args))
    })

  /** Constructs a Block instance with the given arguments type and a function
    * to generate operations given the created block arguments.
    *
    * @param argument_type
    *   The types of the arguments as an Iterable of Attributes.
    * @param operations_expr
    *   A function creating the contained operation(s) given the block
    *   arguments.
    */
  def this(
      argument_type: Attribute,
      operations_expr: Value[Attribute] => Iterable[MLIROperation] |
        MLIROperation
  ) =
    this({
      val arg = Value(argument_type)
      (arg, operations_expr(arg))
    })

  var container_region: Option[Region] = None

  private def attach_op(op: MLIROperation): Unit = {
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

  def add_op(new_op: MLIROperation): Unit = {
    val oplen = operations.length
    attach_op(new_op)
    operations.insertAll(oplen, ListType(new_op))
  }

  def add_ops(new_ops: Seq[MLIROperation]): Unit = {
    val oplen = operations.length
    for (op <- new_ops) {
      attach_op(op)
    }
    operations.insertAll(oplen, ListType(new_ops: _*))
  }

  def insert_op_before(
      existing_op: MLIROperation,
      new_op: MLIROperation
  ): Unit = {
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
      existing_op: MLIROperation,
      new_ops: Seq[MLIROperation]
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

  def insert_op_after(
      existing_op: MLIROperation,
      new_op: MLIROperation
  ): Unit = {
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
      existing_op: MLIROperation,
      new_ops: Seq[MLIROperation]
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

  def detach_op(op: MLIROperation): MLIROperation = {
    (op.container_block equals Some(this)) match {
      case true =>
        op.container_block = None
        operations -= op
        return op
      case false =>
        throw new Exception(
          "MLIROperation can only be detached from a block in which it is contained."
        )
    }
  }

  def erase_op(op: MLIROperation) = {
    detach_op(op)
    op.erase()
  }

  def getIndexOf(op: MLIROperation): Int = {
    operations.lastIndexOf(op) match {
      case -1 => throw new Exception("MLIROperation not present in the block.")
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
