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
  // temporary solution
  def productElementName(i: Int): String = i match {
    case 0 => "name"
    case 1 => "operands"
    case 2 => "successors"
    case 3 => "results"
    case 4 => "regions"
    case 5 => "dictionaryProperties"
    case 6 => "dictionaryAttributes"
  }
  override def hashCode(): Int = {
    val arr = this.productArity
    var code = arr
    var i = 0
    while (i < arr) {
      val elem = this.productElement(i)
      val elemName = this.productElementName(i)
      code = code * 41 + (if (elem == null || elemName == "successors") 0
                          else elem.hashCode())
      i += 1
    }
    code
  }
  override def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }
}

object IR {
  def main(args: Array[String]): Unit = {
    println("TODO compiler")
  }
}
