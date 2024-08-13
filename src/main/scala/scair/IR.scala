package scair
import scala.collection.mutable.{Map, LinkedHashMap, ListBuffer}
import scair.Parser._
import fastparse._

val DictType = LinkedHashMap
type DictType[A, B] = LinkedHashMap[A, B]

val ListType = ListBuffer
type ListType[A] = ListBuffer[A]

sealed trait Attribute {
  def name: String
  def prefix: String = "#"
  def verify(): Unit = ()
}

trait TypeAttribute extends Attribute {
  override def prefix: String = "!"
}

abstract class ParametrizedAttribute(
    override val name: String,
    val parameters: Seq[Attribute] = Seq()
) extends Attribute {
  override def toString =
    s"${prefix}${name}<${parameters.map(x => x.toString).mkString(", ")}>"
}

abstract class DataAttribute[D](
    override val name: String,
    val data: D
) extends Attribute {
  override def toString = data.toString
}

case class Value[T <: Attribute](
    var typ: T
) {
  def verify(): Unit = typ.verify()
  override def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }
}

case class Block(
    operations: Seq[Operation] = Seq(),
    arguments: ListType[Value[Attribute]] = ListType()
) {

  var container_region: Option[Region] = None

  def verify(): Unit = {
    for (op <- operations) op.verify()
    for (arg <- arguments) arg.verify()
  }

  override def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }
}

case class Region(
    blocks: Seq[Block]
) {

  var container_operation: Option[Operation] = None

  def verify(): Unit = {
    for (block <- blocks) block.verify()
  }
  override def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }
}

sealed abstract class Operation(
    val name: String,
    val operands: ListType[Value[Attribute]] = ListType(),
    val successors: ListType[Block] = ListType(),
    val results: ListType[Value[Attribute]] = ListType[Value[Attribute]](),
    val regions: ListType[Region] = ListType[Region](),
    val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) {

  var container_block: Option[Block] = None

  def custom_verify(): Unit = ()

  final def verify(): Unit = {
    for (result <- results) result.verify()
    for (region <- regions) region.verify()
    for ((key, attr) <- dictionaryProperties) attr.verify()
    for ((key, attr) <- dictionaryAttributes) attr.verify()
    custom_verify()
  }

  override def hashCode(): Int = {
    return 7 * 41 +
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

final case class UnregisteredOperation(
    override val name: String,
    override val operands: ListType[Value[Attribute]] = ListType(),
    override val successors: ListType[Block] = ListType(),
    override val results: ListType[Value[Attribute]] =
      ListType[Value[Attribute]](),
    override val regions: ListType[Region] = ListType[Region](),
    override val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends Operation(name = name)

class RegisteredOperation(
    override val name: String,
    override val operands: ListType[Value[Attribute]] = ListType(),
    override val successors: ListType[Block] = ListType(),
    override val results: ListType[Value[Attribute]] =
      ListType[Value[Attribute]](),
    override val regions: ListType[Region] = ListType[Region](),
    override val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends Operation(name = name)

trait DialectOperation {
  def name: String
  def parse[$: P](resNames: Seq[String], parser: Parser): P[Operation] =
    throw new Exception(
      s"No custom Parser implemented for Operation '${name}'"
    )
  type FactoryType = (
      ListType[Value[Attribute]] /* = operands */,
      ListType[Block] /* = successors */,
      ListType[Value[Attribute]] /* = results */,
      ListType[Region] /* = regions */,
      DictType[String, Attribute], /* = dictProps */
      DictType[String, Attribute] /* = dictAttrs */
  ) => Operation
  def factory: FactoryType
  final def constructOp(
      operands: ListType[Value[Attribute]] = ListType(),
      successors: ListType[Block] = ListType(),
      results: ListType[Value[Attribute]] = ListType[Value[Attribute]](),
      regions: ListType[Region] = ListType[Region](),
      dictionaryProperties: DictType[String, Attribute] =
        DictType.empty[String, Attribute],
      dictionaryAttributes: DictType[String, Attribute] =
        DictType.empty[String, Attribute]
  ): Operation = factory(
    operands,
    successors,
    results,
    regions,
    dictionaryProperties,
    dictionaryAttributes
  )
}

trait DialectAttribute {
  def name: String
  type FactoryType = (Seq[Attribute]) => Attribute
  def factory: FactoryType = ???
  def parser[$: P]: P[Seq[Attribute]] =
    P(("<" ~ Type.rep(sep = ",") ~ ">").?).map(_.getOrElse(Seq()))
  def parse[$: P]: P[Attribute] =
    parser.map(factory(_))
}

final case class Dialect(
    val operations: Seq[DialectOperation],
    val attributes: Seq[DialectAttribute]
) {}

object IR {}

/*

import scala.quoted.*

trait MyObject {
  def name: String
}

trait MyTrait {
  type ReturnType <: MyObject

  inline def parseReturn(name: String):
    ReturnType = ${ MyTrait.impl[ReturnType]('name) }
}

object MyTrait {
  def impl[T <: MyObject: Type]
  (name: Expr[String])(using Quotes): Expr[T] = {
    import quotes.reflect.*

    // Generate code to instantiate the ReturnType with String argument
    val returnTypeSymbol = TypeRepr.of[T].typeSymbol
    val instance = New(TypeIdent(returnTypeSymbol))
      .select(returnTypeSymbol.primaryConstructor)
      .appliedToArgs(name)
      .asExprOf[T]

    instance
  }
}

 */
