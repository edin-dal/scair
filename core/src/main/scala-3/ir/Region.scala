package scair.ir

import scair.utils.OK

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

  final override def deepCopy(using
      blockMapper: mutable.Map[Block, Block] = mutable.Map.empty,
      valueMapper: mutable.Map[Value[Attribute], Value[Attribute]] = mutable.Map
        .empty,
  ): Region =
    Region(blocks.map(_.deepCopy))

  final override def parent = containerOperation

  var containerOperation: Option[Operation] = None

  blocks.foreach(attachBlock)

  def structured =
    blocks.foldLeft[OK[Unit]](Right(()))((res, block) =>
      res.flatMap(_ => block.structured)
    )

  def verify(): OK[Unit] =
    blocks.foldLeft[OK[Unit]](Right(()))((res, block) =>
      res.flatMap(_ => block.verify())
    )

  override def equals(o: Any): Boolean =
    return this eq o.asInstanceOf[AnyRef]

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
