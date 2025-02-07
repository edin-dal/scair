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
) {

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

}