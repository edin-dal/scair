package scair

case class Region(
    blocks: Seq[Block],
    parent: Option[Operation] = None
) {

  override def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }
}

case class Block(
    operations: Seq[Operation],
    arguments: Seq[Value]
) {

  override def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }
}

abstract class Attribute(val name: String)
abstract class Type(override val name: String) extends Attribute(name)

case class Value(
    // val name: String,
    typ: Attribute
) {

  override def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }
}

case class Operation(
    name: String,
    operands: Seq[Value],
    var successors: Seq[Block],
    results: Seq[Value],
    regions: Seq[Region]
) {

  override def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }
}

object IR {
  def main(args: Array[String]): Unit = {
    println("TODO compiler")
  }
}
