package scair.ir

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
      valueMapper: mutable.Map[Value[Attribute], Value[Attribute]] =
        mutable.Map.empty
  ): Region =
    Region(blocks.map(_.deepCopy))

  final override def parent = container_operation

  var container_operation: Option[Operation] = None

  blocks.foreach(attach_block)

  def structured =
    blocks.foldLeft[Either[String, Unit]](Right(()))((res, block) =>
      res.flatMap(_ => block.structured)
    )

  def verify(): Either[String, Unit] =
    blocks.foldLeft[Either[String, Unit]](Right(()))((res, block) =>
      res.flatMap(_ => block.verify())
    )

  override def equals(o: Any): Boolean =
    return this eq o.asInstanceOf[AnyRef]

  // Heavily debatable
  def detached =
    container_operation = None
    this

  private def attach_block(block: Block): Unit =

    block.container_region match
      case Some(x) =>
        throw new Exception(
          "Can't attach a block already attached to a region."
        )
      case None =>
        block.is_ancestor(this) match
          case true =>
            throw new Exception(
              "Can't add a block to a region that is contained within that operation"
            )
          case false =>
            block.container_region = Some(this)
