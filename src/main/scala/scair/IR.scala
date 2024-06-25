package scair
import scala.collection.immutable

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
    operands: Seq[Value] = Seq(),
    var successors: Seq[Block] = Seq(),
    results: Seq[Value] = Seq[Value](),
    regions: Seq[Region] = Seq[Region](),
    dictionaryProperties: immutable.Map[String, Attribute] =
      immutable.Map.empty[String, Attribute],
    dictionaryAttributes: immutable.Map[String, Attribute] =
      immutable.Map.empty[String, Attribute]
) {
  // override def equals(o: Any): Boolean = {
  //   return this eq o.asInstanceOf[AnyRef]
  // }
}

object IR {
  def main(args: Array[String]): Unit = {
    println("TODO compiler")
  }
}
