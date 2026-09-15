package scair.ir

import scair.helpers.foreachInline
import scair.utils.*

import scala.annotation.tailrec
import scala.annotation.targetName
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

  private def applyArgs(
      arguments: ListType[Value[Attribute]],
      operations: BlockOperations,
  ): Block = new Block(arguments, operations)

  private def applyArgs(
      arguments: Iterable[Value[Attribute]],
      operations: Iterable[Operation],
  ): Block = Block(ListType.from(arguments), BlockOperations.from(operations))

  @targetName("apply2")
  def apply(
      argumentsTypes: Iterable[Attribute],
      operations: Iterable[Operation],
  ): Block = Block.applyArgs(argumentsTypes.map(Value.apply), operations)

  /** Constructs a Block instance with the given argument type and a function to
    * generate operations given the created block argument.
    *
    * @param argumentTypes
    *   The type of the argument.
    * @param operationsExpr
    *   A function creating the contained operation(s) given the block argument.
    */
  @targetName("apply4")
  def apply[A <: Attribute](
      argumentsTypes: Iterable[A],
      operationsExpr: Iterable[Value[A]] => Iterable[Operation],
  ): Block =
    val args = argumentsTypes.map(Value(_))
    val operations = operationsExpr(args)

    Block.applyArgs(args, operations)

  /** Constructs a Block instance with the given arguments type and function to
    * generate operations given the created block arguments.
    *
    * @param argumentType
    *   The types of the arguments as an Iterable of Attributes.
    * @param operationsExpr
    *   A function creating the contained operation(s) given the block
    *   arguments.
    */
  @targetName("apply5")
  def apply[A <: Attribute](
      argumentType: A,
      operationsExpr: Value[A] => Iterable[Operation],
  ): Block =
    val arg = Value(argumentType)
    val ops = operationsExpr(arg)
    Block.applyArgs(ListType(arg), ops)

  /*
   * Type-level helper, mapping a tuple of Attribute types to a tuple of value types of
   * those types. Scala's Tuple.Map is assuming unbounded functors, which Value is not,
   * as its parameter is bounded by Attribute.
   */
  type MapValues[T <: Tuple] <: Tuple = T match
    case h *: t =>
      h match
        case Attribute => Value[h] *: MapValues[t]
    case EmptyTuple => EmptyTuple

  // This constructors requires the above type helper, but the above type helper is not
  // reduced during overload resolution, so we need to give it a different name.
  // It seems this one falls in a very doable subset of this issue though. It might be
  // worth crafting a minimal example and opening a discussion with Scala developpers.
  @targetName("apply8")
  inline def typed[T <: Tuple](
      argumentsTypes: T,
      operationsExpr: MapValues[T] => Iterable[Operation],
  ): Block =
    val argsArray = argumentsTypes.toArray
      .map(a => Value(a.asInstanceOf[Attribute]))
    val args =
      Tuple.fromArray(argsArray).asInstanceOf[MapValues[T]]
    val ops = operationsExpr(args)
    Block.applyArgs(argsArray, ops)

  /** Constructs a Block instance with the given operations and no block
    * arguments.
    *
    * @param operations
    *   The operations, either as a single MLIROperation or an Iterable of
    *   MLIROperations.
    */
  @targetName("apply3")
  def apply(operations: Iterable[Operation]): Block =
    Block.applyArgs(
      arguments = ListType.empty,
      operations = operations,
    )

  @targetName("apply4")
  def apply(operation: Operation): Block =
    Block.applyArgs(
      arguments = ListType.empty,
      operations = BlockOperations(operation),
    )

  def apply(): Block =
    Block.applyArgs(ListType.empty[Value[Attribute]], BlockOperations.empty)

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

  /*≡==--==≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
  ||  BLOCK INITIALIZATION  ||
  \*≡==---==≡≡≡≡≡≡≡≡==---==≡*/

  // it is implicitly true, as BlockOperations data structure will calculate operation order on construction
  // also the Parser populates the indexes when adding operations into the block - kind of ad-hoc but works :D
  // tbd if this is a good design
  var isOpOrderValid = true

  var containerRegion: Option[Region] = None
  final override def parent: Option[Region] = containerRegion

  arguments.foreach(a =>
    if a.owner != None then
      throw new Exception(
        s"Block argument '${a.typ}' already has an owner: ${a.owner.get}"
      )
    else a.owner = Some(this)
  )

  operations.foreach(attachOp)

  /*≡==--==≡≡≡≡≡≡≡≡==--=≡≡*\
  ||   BLOCK TRAVERSAL   ||
  \*≡==---==≡≡≡≡≡≡==---==≡*/

  /** Applies f to each operation, inlined: no closure nor iterator. The next
    * operation is read before f is applied, so f may erase the current one.
    */
  inline final def forEachOp(inline f: Operation => Unit): Unit =
    operations.foreachInline(f)

  /** Walks all operations of this block in pre-order. See WalkResult. */
  final def walk(f: Operation => WalkResult): WalkResult =
    var result = WalkResult.Advance
    forEachOp(op => if result ne WalkResult.Interrupt then result = op.walk(f))
    result

  /** Walks all operations of this block in post-order. See WalkResult. */
  final def walkPostOrder(f: Operation => WalkResult): WalkResult =
    var result = WalkResult.Advance
    forEachOp(op =>
      if result ne WalkResult.Interrupt then result = op.walkPostOrder(f)
    )
    result

  /** Walks all operations of this block in pre-order, applying f. */
  inline final def walkAll(inline f: Operation => Unit): Unit =
    walk(op =>
      f(op)
      WalkResult.Advance
    )

  /** The operation directly in this block containing `op` - `op` itself if it
    * is directly in this block - if any.
    */
  @tailrec
  final def findAncestorOp(op: Operation): Option[Operation] =
    op.containerBlock match
      case Some(block) if block eq this => Some(op)
      case Some(block)                  =>
        block.containerRegion match
          case Some(region) =>
            region.containerOperation match
              case Some(parent) => findAncestorOp(parent)
              case None         => None
          case None => None
      case None => None

  /*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
  || BLOCK TRANSFORMATIONS ||
  \*≡==---==≡≡≡≡≡≡≡==---==≡*/

  /** Whether `op` is directly contained in this block. */
  inline private def isContainerOf(op: Operation): Boolean =
    op.containerBlock match
      case Some(block) => block eq this
      case None        => false

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

  def detachOp(op: Operation): Operation =
    isContainerOf(op) match
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

  def addOp(newOp: Operation): Unit =
    val oplen = operations.length
    attachOp(newOp)
    operations.insertAll(oplen, ListType(newOp))

  def addOps(newOps: Seq[Operation]): Unit =
    val oplen = operations.length
    newOps.foreachInline(attachOp)
    operations.insertAll(oplen, newOps)

  def insertOpBefore(
      existingOp: Operation,
      newOp: Operation,
  ): Unit = isContainerOf(existingOp) match
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
  ): Unit = isContainerOf(existingOp) match
    case true =>
      newOps.foreachInline(attachOp)
      operations.insertAll(existingOp, newOps)
    case false =>
      throw new Exception(
        "Can't insert the new operation into the block, as the operation that was " +
          "given as a point of reference does not exist in the current block."
      )

  def insertOpAfter(
      existingOp: Operation,
      newOp: Operation,
  ): Unit = isContainerOf(existingOp) match
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
  ): Unit = isContainerOf(existingOp) match
    case true =>
      newOps.foreachInline(attachOp)
      existingOp.next match
        case Some(n) => operations.insertAll(n, newOps)
        case None    => operations.addAll(newOps)
    case false =>
      throw new Exception(
        "Can't insert the new operation into the block, as the operation that was " +
          "given as a point of reference does not exist in the current block."
      )

  /*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
  ||   BLOCK STRUCTURING   ||
  \*≡==---==≡≡≡≡≡≡≡==---==≡*/

  override def recomputeOpOrder(): Unit =
    if !isOpOrderValid then
      isOpOrderValid = true
      operations.computeBlockOrder()

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

  /*≡==--==≡≡≡≡≡≡==--=≡≡*\
  ||   OBJECT METHODS   ||
  \*≡==---==≡≡≡≡==---==≡*/

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

  override def equals(o: Any): Boolean =
    return this eq o.asInstanceOf[AnyRef]
