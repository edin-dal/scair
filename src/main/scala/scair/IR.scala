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
    operations: Seq[Operation] = Seq(),
    arguments: Seq[Value] = Seq()
) {
  override def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }
}

abstract sealed class Attribute(val name: String)
abstract trait TypeAttribute extends Attribute

abstract class ParametrizedAttribute(
    override val name: String,
    val parameters: Option[Attribute] | Attribute*
) extends Attribute(name)

abstract class DataAttribute[D](override val name: String, val data: D)
    extends Attribute(name) {
  override def toString = data.toString
}

case class Value(
    // val name: String,
    typ: Attribute
) {
  override def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }
}

abstract sealed case class Operation(
    operands: Seq[Value] = Seq(),
    successors: collection.mutable.ArrayBuffer[Block] =
      collection.mutable.ArrayBuffer(),
    results: Seq[Value] = Seq[Value](),
    regions: Seq[Region] = Seq[Region](),
    dictionaryProperties: immutable.Map[String, Attribute] =
      immutable.Map.empty[String, Attribute],
    dictionaryAttributes: immutable.Map[String, Attribute] =
      immutable.Map.empty[String, Attribute]
) {
  def name: String

  override def hashCode(): Int = {
    return this.productArity * 41 +
      this.name.hashCode() +
      this.operands.hashCode() +
      this.results.hashCode() +
      this.regions.hashCode() +
      this.dictionaryProperties.hashCode() +
      this.dictionaryAttributes.hashCode()
  }
  override def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }
}

class UnregisteredOperation(
    val name: String,
    override val operands: Seq[Value] = Seq(),
    override val successors: collection.mutable.ArrayBuffer[Block] =
      collection.mutable.ArrayBuffer(),
    override val results: Seq[Value] = Seq[Value](),
    override val regions: Seq[Region] = Seq[Region](),
    override val dictionaryProperties: immutable.Map[String, Attribute] =
      immutable.Map.empty[String, Attribute],
    override val dictionaryAttributes: immutable.Map[String, Attribute] =
      immutable.Map.empty[String, Attribute]
) extends Operation {}

object UnregisteredOperation {
  def unapply(op: UnregisteredOperation): (
      String,
      Seq[Value],
      collection.mutable.Seq[Block],
      Seq[Value],
      Seq[Region],
      Map[String, Attribute],
      Map[String, Attribute]
  ) = {
    return (
      op.name,
      op.operands,
      op.successors,
      op.results,
      op.regions,
      op.dictionaryProperties,
      op.dictionaryAttributes
    )
  }
}
final case class Dialect(
    val operation: Seq[Class[Operation]],
    val attributes: Seq[Class[Attribute]]
) {
  // val operations :
}

object IR {
  def main(args: Array[String]): Unit = {
    println("TODO compiler")
  }
}
