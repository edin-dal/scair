package scair.dialects.CMath.cmath

import fastparse._
import scair.dialects.builtin._
import scala.collection.immutable
import scala.collection.mutable
import scair.dialects.irdl.{Operand, OpResult}
import scair.Parser.{whitespace, Type}
import scair.{
  DictType,
  RegisteredOperation,
  Region,
  Block,
  Value,
  Attribute,
  TypeAttribute,
  ParametrizedAttribute,
  DataAttribute,
  DialectAttribute,
  DialectOperation,
  Dialect,
  Printer,
  AttrParser,
  Parser
}

//////////////////
// COMPLEX TYPE //
//////////////////

object ComplexType extends DialectAttribute {
  override def name: String = "cmath.complex"
  override def factory = ComplexType.apply
}

case class ComplexType(val body: Seq[Attribute])
    extends ParametrizedAttribute(
      name = "cmath.complex",
      parameters = body
    )
    with TypeAttribute {

  override def verify(): Unit =
    if (body.length != 1) {
      throw new Exception("TupleStream Tuple must contain 1 elements only.")
    } else
      body(0) match {
        case Float32Type =>
        case Float64Type =>
        case _ =>
          throw new Exception(
            "Complex type must be constructed with either 'f32' or 'f64' attribute."
          )
      }
}

////////////////////
// NORM OPERATION //
////////////////////

object Norm extends DialectOperation {
  override def name: String = "cmath.norm"
  override def factory: FactoryType = Norm.apply
}

case class Norm(
    override val operands: collection.mutable.ArrayBuffer[Value[Attribute]] =
      collection.mutable.ArrayBuffer(),
    override val successors: collection.mutable.ArrayBuffer[Block] =
      collection.mutable.ArrayBuffer(),
    override val results: Seq[Value[Attribute]] = Seq[Value[Attribute]](),
    override val regions: Seq[Region] = Seq[Region](),
    override val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends RegisteredOperation(name = "cmath.norm") {

  override def custom_verify(): Unit = (
    operands.length,
    successors.length,
    results.length,
    regions.length,
    dictionaryProperties.size,
    dictionaryAttributes.size
  ) match {
    case (1, 0, 1, 0, 0, 0) =>
    case _ =>
      throw new Exception(
        "Norm Operation must only contain 1 operand of 'complex' type, and 1 result of 'f32' or 'f64'."
      )
  }
}

///////////////////
// MUL OPERATION //
///////////////////

object Mul extends DialectOperation {
  override def name: String = "cmath.mul"
  override def factory: FactoryType = Mul.apply
}

case class Mul(
    override val operands: collection.mutable.ArrayBuffer[Value[Attribute]] =
      collection.mutable.ArrayBuffer(),
    override val successors: collection.mutable.ArrayBuffer[Block] =
      collection.mutable.ArrayBuffer(),
    override val results: Seq[Value[Attribute]] = Seq[Value[Attribute]](),
    override val regions: Seq[Region] = Seq[Region](),
    override val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends RegisteredOperation(name = "cmath.mul") {

  override def custom_verify(): Unit = (
    operands.length,
    successors.length,
    results.length,
    regions.length,
    dictionaryProperties.size,
    dictionaryAttributes.size
  ) match {
    case (2, 0, 1, 0, 0, 0) =>
    case _ =>
      throw new Exception(
        "Mul Operation must only contain 2 operands and 1 result of 'complex' type."
      )
  }
}

///////////
// CMATH //
///////////

val CMath: Dialect =
  new Dialect(
    operations = Seq(Norm, Mul),
    attributes = Seq(ComplexType)
  )
