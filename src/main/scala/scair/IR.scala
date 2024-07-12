package scair
import scala.collection.{immutable, mutable}
import scair.Parser
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
    val parameters: Option[Attribute] | Attribute*
) extends Attribute

abstract class DataAttribute[D](
    override val name: String,
    val data: D
) extends Attribute {
  override def toString = data.toString
}

case class Value[+T](
    typ: T
) {
  override def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }
}

case class Block(
    operations: Seq[Operation] = Seq(),
    arguments: Seq[Value[Attribute]] = Seq()
) {
  override def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }
}

case class Region(
    blocks: Seq[Block],
    parent: Option[Operation] = None
) {
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

  def verify(): Unit = ()

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
) extends Operation(name = name) {}

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
) extends Operation(name = name) {}

trait DialectOperation {
  def name: String
  def parse(parser: Parser)(implicit
      parsingCtx: P[Any]
  ): Option[P[Operation]] = None
  def constructOp(
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
  ): Operation
}

trait DialectAttribute {
  def name: String
  def parseReturn[T <: Attribute]: => T
  def parse[$: P]: P[Attribute] = throw new Exception(
    s"No custom Parser implemented for Attribute '${name}'"
  )
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
