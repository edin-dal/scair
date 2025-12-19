package scair.ir

import scair.utils.*

import scala.collection.mutable

//
// ██████╗░ ██╗░░░░░ ░█████╗░ ░█████╗░ ██╗░░██╗
// ██╔══██╗ ██║░░░░░ ██╔══██╗ ██╔══██╗ ██║░██╔╝
// ██████╦╝ ██║░░░░░ ██║░░██║ ██║░░╚═╝ █████═╝░
// ██╔══██╗ ██║░░░░░ ██║░░██║ ██║░░██╗ ██╔═██╗░
// ██████╦╝ ███████╗ ╚█████╔╝ ╚█████╔╝ ██║░╚██╗
// ╚═════╝░ ╚══════╝ ░╚════╝░ ░╚════╝░ ╚═╝░░╚═╝
//

/*≡==--==≡≡≡≡==--=≡≡*\
||      BLOCKS      ||
\*≡==---==≡≡==---==≡*/

/** Companion object for the Block class, providing simple `apply`s just
  * forwarding to constructors.
  */
object Block:

  def apply(
      argumentsTypes: Iterable[Attribute] | Attribute = Seq(),
      operations: Iterable[Operation] | Operation = Seq(),
  ): Block = new Block(argumentsTypes, operations)

  def apply(operations: Iterable[Operation] | Operation): Block =
    new Block(
      operations
    )

  def apply(
      argumentsTypes: Iterable[Attribute],
      operationsExpr: Iterable[Value[Attribute]] => Iterable[Operation],
  ): Block =
    new Block(argumentsTypes, operationsExpr)

  def apply(
      argumentsTypes: Attribute,
      operationsExpr: Value[Attribute] => Iterable[Operation],
  ): Block =
    new Block(argumentsTypes, operationsExpr)

/** A basic block.
  *
  * @param arguments
  *   The list of owned block arguments.
  * @param operations
  *   The list of contained operations.
  */
case class Block private (
    val arguments: ListType[Value[Attribute]],
    val operations: BlockOperations,
) extends IRNode:

  final override def deepCopy(using
      blockMapper: mutable.Map[Block, Block] = mutable.Map.empty,
      valueMapper: mutable.Map[Value[Attribute], Value[Attribute]] = mutable.Map
        .empty,
  ): Block =
    Block(
      argumentsTypes = arguments.map(_.typ),
      (args) =>
        valueMapper addAll (arguments zip args)
        operations.map(_.deepCopy),
    )

  final override def parent: Option[Region] = containerRegion

  operations.foreach(attachOp)

  arguments.foreach(a =>
    if a.owner != None then
      throw new Exception(
        s"Block argument '${a.typ}' already has an owner: ${a.owner.get}"
      )
    else a.owner = Some(this)
  )

  /** Constructs a Block instance with the given argument types and operations.
    *
    * @param argumentsTypes
    *   The types of the arguments, either as a single Attribute or an Iterable
    *   of Attributes.
    * @param operations
    *   The operations, either as a single MLIROperation or an Iterable of
    *   MLIROperations.
    */
  def this(
      argumentsTypes: Iterable[Attribute] | Attribute = Seq(),
      operations: Iterable[Operation] | Operation = Seq(),
  ) =
    this(
      ListType.from((argumentsTypes match
        case single: Attribute     => Seq(single)
        case multiple: Iterable[?] => multiple.asInstanceOf[Iterable[Attribute]]
      ).map(Value(_))),
      BlockOperations
        .from((operations match
          case single: Operation     => Seq(single)
          case multiple: Iterable[?] =>
            multiple.asInstanceOf[Iterable[Operation]]
        )),
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
          Iterable[Operation] | Operation,
      )
  ) =
    this(
      ListType
        .from(args._1 match
          case single: Value[Attribute] => Seq(single)
          case multiple: Iterable[?]    =>
            multiple.asInstanceOf[Iterable[Value[Attribute]]]),
      BlockOperations
        .from(args._2 match
          case single: Operation     => Seq(single)
          case multiple: Iterable[?] =>
            multiple.asInstanceOf[Iterable[Operation]]),
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
    * @param argumentType
    *   The type of the argument.
    * @param operationsExpr
    *   A function creating the contained operation(s) given the block argument.
    */
  def this(
      argumentType: Iterable[Attribute],
      operationsExpr: Iterable[Value[Attribute]] => Iterable[Operation] |
        Operation,
  ) =
    this({
      val args = argumentType.map(Value(_))
      (args, operationsExpr(args))
    })

  /** Constructs a Block instance with the given arguments type and function to
    * generate operations given the created block arguments.
    *
    * @param argumentType
    *   The types of the arguments as an Iterable of Attributes.
    * @param operationsExpr
    *   A function creating the contained operation(s) given the block
    *   arguments.
    */
  def this(
      argumentType: Attribute,
      operationsExpr: Value[Attribute] => Iterable[Operation] | Operation,
  ) =
    this({
      val arg = Value(argumentType)
      (arg, operationsExpr(arg))
    })

  var containerRegion: Option[Region] = None

  private def attachOp(op: Operation): Unit =
    op.containerBlock match
      case Some(x) =>
        throw new Exception(
          "Can't attach an operation already attached to a block."
        )
      case None =>
        op.isAncestor(this) match
          case true =>
            throw new Exception(
              "Can't add an operation to a block that is contained within that operation"
            )
          case false =>
            op.containerBlock = Some(this)

  def addOp(newOp: Operation): Unit =
    val oplen = operations.length
    attachOp(newOp)
    operations.insertAll(oplen, ListType(newOp))

  def addOps(newOps: Seq[Operation]): Unit =
    val oplen = operations.length
    for op <- newOps do attachOp(op)
    operations.insertAll(oplen, newOps)

  def insertOpBefore(
      existingOp: Operation,
      newOp: Operation,
  ): Unit = (existingOp.containerBlock `equals` Some(this)) match
    case true =>
      attachOp(newOp)
      operations.insert(existingOp, newOp)
    case false =>
      throw new Exception(
        "Can't insert the new operation into the block, as the operation that was " +
          "given as a point of reference does not exist in the current block."
      )

  def insertOpsBefore(
      existingOp: Operation,
      newOps: Seq[Operation],
  ): Unit = (existingOp.containerBlock `equals` Some(this)) match
    case true =>
      for op <- newOps do attachOp(op)
      operations.insertAll(existingOp, newOps)
    case false =>
      throw new Exception(
        "Can't insert the new operation into the block, as the operation that was " +
          "given as a point of reference does not exist in the current block."
      )

  def insertOpAfter(
      existingOp: Operation,
      newOp: Operation,
  ): Unit = (existingOp.containerBlock `equals` Some(this)) match
    case true =>
      attachOp(newOp)
      existingOp.next match
        case Some(n) => operations.insert(n, newOp)
        case None    => operations.addOne(newOp)
    case false =>
      throw new Exception(
        "Can't insert the new operation into the block, as the operation that was " +
          "given as a point of reference does not exist in the current block."
      )

  def insertOpsAfter(
      existingOp: Operation,
      newOps: Seq[Operation],
  ): Unit = (existingOp.containerBlock `equals` Some(this)) match
    case true =>
      for op <- newOps do attachOp(op)
      existingOp.next match
        case Some(n) => operations.insertAll(n, newOps)
        case None    => operations.addAll(newOps)
    case false =>
      throw new Exception(
        "Can't insert the new operation into the block, as the operation that was " +
          "given as a point of reference does not exist in the current block."
      )

  def detachOp(op: Operation): Operation =
    (op.containerBlock `equals` Some(this)) match
      case true =>
        op.containerBlock = None
        operations -= op
        op
      case false =>
        throw new Exception(
          "MLIROperation can only be detached from a block in which it is contained."
        )

  def eraseOp(op: Operation, safeErase: Boolean = true) =
    detachOp(op)
    op.erase(safeErase)

  def structured: OK[Unit] =
    operations.foldLeft[OK[Unit]](OK())((res, op) =>
      res.flatMap(_ =>
        op.structured.map(v =>
          if !(v eq op) then
            operations(op) = v
            v.containerBlock = Some(this)
        )
      )
    )

  def verify(): OK[Unit] =
    arguments
      .foldLeft[OK[Unit]](OK())((res, arg) => res.flatMap(_ => arg.verify()))
      .flatMap(_ =>
        operations.foldLeft[OK[Unit]](OK())((res, op) =>
          res.flatMap(_ => op.verify().map(_ => ()))
        )
      )

  override def equals(o: Any): Boolean =
    return this eq o.asInstanceOf[AnyRef]
