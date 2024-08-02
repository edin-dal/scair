package scair.dialects.cmath

import fastparse._
import scair.dialects.builtin._
import scala.collection.immutable
import scair.dialects.irdl.{Operand, OpResult}
import scair.Parser.{whitespace, Type}
import scair.{
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
  override def parse[$: P]: P[Attribute] =
    P("<" ~ Type ~ ">").map(ComplexType(_))
}

case class ComplexType(val cmplxType: Attribute)
    extends ParametrizedAttribute(
      name = "cmath.complex",
      parameters = Seq(cmplxType)
    )
    with TypeAttribute {

  override def verify(): Unit = cmplxType match {
    case Float32Type =>
    case Float64Type =>
    case _ =>
      throw new Exception(
        "Complex type must be constructed with either 'f32' or 'f64' attribute."
      )
  }
  override def toString = s"${prefix}cmath.complex<$cmplxType>"
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
    override val dictionaryProperties: immutable.Map[String, Attribute] =
      immutable.Map.empty[String, Attribute],
    override val dictionaryAttributes: immutable.Map[String, Attribute] =
      immutable.Map.empty[String, Attribute]
) extends RegisteredOperation(name = "cmath.norm") {

  override def verify(): Unit = (
    operands.length,
    successors.length,
    results.length,
    regions.length,
    dictionaryProperties.size,
    dictionaryAttributes.size
  ) match {
    case (1, 0, 1, 0, 0, 0) =>
      operands(0).typ.verify()
      results(0).typ.verify()
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
    override val dictionaryProperties: immutable.Map[String, Attribute] =
      immutable.Map.empty[String, Attribute],
    override val dictionaryAttributes: immutable.Map[String, Attribute] =
      immutable.Map.empty[String, Attribute]
) extends RegisteredOperation(name = "cmath.mul") {

  override def verify(): Unit = (
    operands.length,
    successors.length,
    results.length,
    regions.length,
    dictionaryProperties.size,
    dictionaryAttributes.size
  ) match {
    case (2, 0, 1, 0, 0, 0) =>
      operands(0).typ.verify()
      operands(1).typ.verify()
      results(0).typ.verify()
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
