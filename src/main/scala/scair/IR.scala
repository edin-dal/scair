package scair
import scala.collection.{immutable, mutable}
import scair.Parser._
import fastparse._

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
    arguments: Seq[Value[Attribute]] = Seq()
) {

  def verify(): Unit = {
    for (op <- operations) op.verify()
    for (arg <- arguments) arg.verify()
  }

  override def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }
}

case class Region(
    blocks: Seq[Block],
    parent: Option[Operation] = None
) {

  def verify(): Unit = {
    for (block <- blocks) block.verify()
  }
  override def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }
}

sealed abstract class Operation(
    val name: String,
    val operands: collection.mutable.ArrayBuffer[Value[Attribute]] =
      collection.mutable.ArrayBuffer(),
    val successors: collection.mutable.ArrayBuffer[Block] =
      collection.mutable.ArrayBuffer(),
    val results: Seq[Value[Attribute]] = Seq[Value[Attribute]](),
    val regions: Seq[Region] = Seq[Region](),
    val dictionaryProperties: immutable.Map[String, Attribute] =
      immutable.Map.empty[String, Attribute],
    val dictionaryAttributes: immutable.Map[String, Attribute] =
      immutable.Map.empty[String, Attribute]
) {

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
    override val operands: collection.mutable.ArrayBuffer[Value[Attribute]] =
      collection.mutable.ArrayBuffer(),
    override val successors: collection.mutable.ArrayBuffer[Block] =
      collection.mutable.ArrayBuffer(),
    override val results: Seq[Value[Attribute]] = Seq[Value[Attribute]](),
    override val regions: Seq[Region] = Seq[Region](),
    override val dictionaryProperties: immutable.Map[String, Attribute] =
      immutable.Map.empty[String, Attribute],
    override val dictionaryAttributes: immutable.Map[String, Attribute] =
      immutable.Map.empty[String, Attribute]
) extends Operation(name = name)

class RegisteredOperation(
    override val name: String,
    override val operands: collection.mutable.ArrayBuffer[Value[Attribute]] =
      collection.mutable.ArrayBuffer(),
    override val successors: collection.mutable.ArrayBuffer[Block] =
      collection.mutable.ArrayBuffer(),
    override val results: Seq[Value[Attribute]] = Seq[Value[Attribute]](),
    override val regions: Seq[Region] = Seq[Region](),
    override val dictionaryProperties: immutable.Map[String, Attribute] =
      immutable.Map.empty[String, Attribute],
    override val dictionaryAttributes: immutable.Map[String, Attribute] =
      immutable.Map.empty[String, Attribute]
) extends Operation(name = name)

trait DialectOperation {
  def name: String
  def parse[$: P](resNames: Seq[String], parser: Parser): P[Operation] =
    throw new Exception(
      s"No custom Parser implemented for Operation '${name}'"
    )
  type FactoryType = (
      collection.mutable.ArrayBuffer[Value[Attribute]] /* = operands */,
      collection.mutable.ArrayBuffer[Block] /* = successors */,
      Seq[Value[Attribute]] /* = results */,
      Seq[Region] /* = regions */,
      collection.immutable.Map[String, Attribute] /* = dictProps */,
      collection.immutable.Map[String, Attribute] /* = dictAttrs */
  ) => Operation
  def factory: FactoryType
  final def constructOp(
      operands: collection.mutable.ArrayBuffer[Value[Attribute]] =
        collection.mutable.ArrayBuffer(),
      successors: collection.mutable.ArrayBuffer[Block] =
        collection.mutable.ArrayBuffer(),
      results: Seq[Value[Attribute]] = Seq[Value[Attribute]](),
      regions: Seq[Region] = Seq[Region](),
      dictionaryProperties: immutable.Map[String, Attribute] =
        immutable.Map.empty[String, Attribute],
      dictionaryAttributes: immutable.Map[String, Attribute] =
        immutable.Map.empty[String, Attribute]
  ): Operation = factory(
    operands,
    successors,
    results,
    regions,
    dictionaryProperties,
    dictionaryProperties
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
