package scair.ir

import scair.helpers.foreachInline
import scair.utils.*

import scala.annotation.tailrec
import scala.annotation.targetName
import scala.collection.mutable
//
// ██████╗░ ███████╗ ░██████╗░ ██╗ ░█████╗░ ███╗░░██╗
// ██╔══██╗ ██╔════╝ ██╔════╝░ ██║ ██╔══██╗ ████╗░██║
// ██████╔╝ █████╗░░ ██║░░██╗░ ██║ ██║░░██║ ██╔██╗██║
// ██╔══██╗ ██╔══╝░░ ██║░░╚██╗ ██║ ██║░░██║ ██║╚████║
// ██║░░██║ ███████╗ ╚██████╔╝ ██║ ╚█████╔╝ ██║░╚███║
// ╚═╝░░╚═╝ ╚══════╝ ░╚═════╝░ ╚═╝ ░╚════╝░ ╚═╝░░╚══╝
//

/*≡==--==≡≡≡==--=≡≡*\
||     REGIONS     ||
\*≡==---==≡==---==≡*/

object Region:

  @targetName("applySeq")
  def apply(blocks: Seq[Block]): Region =
    Region(blocks*)

  def apply(operations: Iterable[Operation]): Region =
    Region(Block(operations))

  def apply(operation: Operation): Region =
    Region(Block(operation))

case class Region(
    blocks: Block*
) extends IRNode:

  /*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
  ||   REGION INITIALIZATION   ||
  \*≡==---==≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

  final override def parent = containerOperation

  var containerOperation: Option[Operation] = None

  blocks.foreach(attachBlock)

  /*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
  ||   REGION TRANSFORMATIONS   ||
  \*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

  // Heavily debatable
  def detached =
    containerOperation = None
    this

  private def attachBlock(block: Block): Unit =

    block.containerRegion match
      case Some(x) =>
        throw new Exception(
          "Can't attach a block already attached to a region."
        )
      case None =>
        block.isAncestor(this) match
          case true =>
            throw new Exception(
              "Can't add a block to a region that is contained within that operation"
            )
          case false =>
            block.containerRegion = Some(this)

  /*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
  ||   REGION TRAVERSAL   ||
  \*≡==---==≡≡≡≡≡≡≡==---==≡*/

  /** Applies f to each block, inlined: no closure nor iterator. */
  inline final def forEachBlock(inline f: Block => Unit): Unit =
    blocks.foreachInline(f)

  /** Walks all operations of this region in pre-order. See WalkResult. */
  final def walk(f: Operation => WalkResult): WalkResult =
    var result = WalkResult.Advance
    forEachBlock(block =>
      if result ne WalkResult.Interrupt then result = block.walk(f)
    )
    result

  /** Walks all operations of this region in post-order. See WalkResult. */
  final def walkPostOrder(f: Operation => WalkResult): WalkResult =
    var result = WalkResult.Advance
    forEachBlock(block =>
      if result ne WalkResult.Interrupt then result = block.walkPostOrder(f)
    )
    result

  /** Walks all operations of this region in pre-order, applying f. */
  inline final def walkAll(inline f: Operation => Unit): Unit =
    walk(op =>
      f(op)
      WalkResult.Advance
    )

  /** The operation directly in this region containing `op` - `op` itself if it
    * is directly in this region - if any.
    */
  @tailrec
  final def findAncestorOp(op: Operation): Option[Operation] =
    op.containerBlock match
      case Some(block) =>
        block.containerRegion match
          case Some(region) if region eq this => Some(op)
          case Some(region)                   =>
            region.containerOperation match
              case Some(parent) => findAncestorOp(parent)
              case None         => None
          case None => None
      case None => None

  /*≡==--==≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
  ||   REGION STRUCTURING   ||
  \*≡==---==≡≡≡≡≡≡≡≡==---==≡*/

  override def recomputeOpOrder(): Unit =
    forEachBlock(_.recomputeOpOrder())

  def structured =
    blocks.foldLeft[OK[Unit]](OK())((res, block) =>
      res.flatMap(_ => block.structured)
    )

  def verify(): OK[Unit] =
    blocks.foldLeft[OK[Unit]](OK())((res, block) =>
      res.flatMap(_ => block.verify())
    )

  /*≡==--==≡≡≡≡≡≡==--=≡≡*\
  ||   OBJECT METHODS   ||
  \*≡==---==≡≡≡≡==---==≡*/

  final override def deepCopy(using
      blockMapper: mutable.Map[Block, Block] = mutable.Map.empty,
      valueMapper: mutable.Map[Value[Attribute], Value[Attribute]] = mutable.Map
        .empty,
  ): Region =
    Region(blocks.map(_.deepCopy))

  override def equals(o: Any): Boolean =
    return this eq o.asInstanceOf[AnyRef]
