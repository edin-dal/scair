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
  AttrParser
}

//////////////////
// COMPLEX TYPE //
//////////////////

object ComplexType extends DialectAttribute {
  override def name: String = "cmath.complex"
  override def parse[$: P]: Some[P[Attribute]] =
    Some(P("<" ~ Type ~ ">").map(ComplexType(_)))
}

case class ComplexType(val cmplxType: Attribute)
    extends ParametrizedAttribute(
      name = "cmath.complex",
      parameters = cmplxType
    )
    with TypeAttribute {
  override def toString = s"${prefix}cmath.complex<$cmplxType>"
}

////////////////////
// NORM OPERATION //
////////////////////

object Norm extends DialectOperation {
  override def name: String = "cmath.norm"
  override def constructOp(
      operands: Seq[Value[Attribute]] = Seq(),
      successors: collection.mutable.ArrayBuffer[Block] =
        collection.mutable.ArrayBuffer(),
      results: Seq[Value[Attribute]] = Seq[Value[Attribute]](),
      regions: Seq[Region] = Seq[Region](),
      dictionaryProperties: immutable.Map[String, Attribute] =
        immutable.Map.empty[String, Attribute],
      dictionaryAttributes: immutable.Map[String, Attribute] =
        immutable.Map.empty[String, Attribute]
  ): Norm = Norm(
    operands,
    successors,
    results,
    regions,
    dictionaryProperties,
    dictionaryAttributes
  )
}

case class Norm(
    override val operands: Seq[Value[Attribute]] = Seq(),
    override val successors: collection.mutable.ArrayBuffer[Block] =
      collection.mutable.ArrayBuffer(),
    override val results: Seq[Value[Attribute]] = Seq[Value[Attribute]](),
    override val regions: Seq[Region] = Seq[Region](),
    override val dictionaryProperties: immutable.Map[String, Attribute] =
      immutable.Map.empty[String, Attribute],
    override val dictionaryAttributes: immutable.Map[String, Attribute] =
      immutable.Map.empty[String, Attribute]
) extends RegisteredOperation(name = "cmath.norm") {

  override def verify(): Unit = ()
}

///////////////////
// MUL OPERATION //
///////////////////

object Mul extends DialectOperation {
  override def name: String = "cmath.mul"
  override def constructOp(
      operands: Seq[Value[Attribute]] = Seq(),
      successors: collection.mutable.ArrayBuffer[Block] =
        collection.mutable.ArrayBuffer(),
      results: Seq[Value[Attribute]] = Seq[Value[Attribute]](),
      regions: Seq[Region] = Seq[Region](),
      dictionaryProperties: immutable.Map[String, Attribute] =
        immutable.Map.empty[String, Attribute],
      dictionaryAttributes: immutable.Map[String, Attribute] =
        immutable.Map.empty[String, Attribute]
  ): Mul = Mul(
    operands,
    successors,
    results,
    regions,
    dictionaryProperties,
    dictionaryAttributes
  )
}

case class Mul(
    override val operands: Seq[Value[Attribute]] = Seq(),
    override val successors: collection.mutable.ArrayBuffer[Block] =
      collection.mutable.ArrayBuffer(),
    override val results: Seq[Value[Attribute]] = Seq[Value[Attribute]](),
    override val regions: Seq[Region] = Seq[Region](),
    override val dictionaryProperties: immutable.Map[String, Attribute] =
      immutable.Map.empty[String, Attribute],
    override val dictionaryAttributes: immutable.Map[String, Attribute] =
      immutable.Map.empty[String, Attribute]
) extends RegisteredOperation(name = "cmath.mul") {

  override def verify(): Unit = ()
}

///////////
// CMATH //
///////////

val CMath: Dialect =
  new Dialect(
    operations = Seq(Norm, Mul),
    attributes = Seq(ComplexType)
  )

object CMathh {
  def main(args: Array[String]): Unit = {
    val res = parse("!cmath.complex<f32>", Type(_))
    println(res)
  }
}
