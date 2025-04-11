package scair.ir

// ██╗ ██████╗░
// ██║ ██╔══██╗
// ██║ ██████╔╝
// ██║ ██╔══██╗
// ██║ ██║░░██║
// ╚═╝ ╚═╝░░╚═╝

/*≡==--==≡≡≡==--=≡≡*\
||     REGIONS     ||
\*≡==---==≡==---==≡*/

case class Region(
    blocks: Seq[Block]
) extends IRNode {

  final override def parent = container_operation

  var container_operation: Option[Operation] = None

  def drop_all_references: Unit = {
    container_operation = None
    for (block <- blocks) block.drop_all_references
  }

  def verify(): Unit = {
    for (block <- blocks) block.verify()
  }

  override def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }
  
  private def attach_block(block: Block): Unit = {

    // block.container_region match {
    //   case Some(x) =>
    //     throw new Exception(
    //       "Can't attach a block already attached to a region."
    //     )
    //   case None =>
        block.is_ancestor(this) match {
          case true =>
            throw new Exception(
              "Can't add a block to a region that is contained within that operation"
            )
          case false =>
            block.container_region = Some(this)
        }
    // }
  }

}
