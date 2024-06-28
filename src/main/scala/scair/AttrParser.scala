package scair

import fastparse._
import scala.collection.mutable
import scala.util.{Try, Success, Failure}
import IR._
import Parser._
import Printer._

sealed trait Signedness
case object Signed extends Signedness
case object Unsigned extends Signedness
case object Signless extends Signedness

case object Float16Type extends TypeAttribute("builtin.f16") {
  override def toString = "f16"
}
case object Float32Type extends TypeAttribute("builtin.f32") {
  override def toString = "f32"
}
case object Float64Type extends TypeAttribute("builtin.f64") {
  override def toString = "f64"
}
case object Float80Type extends TypeAttribute("builtin.f80") {
  override def toString = "f80"
}
case object Float128Type extends TypeAttribute("builtin.f128") {
  override def toString = "f128"
}

case class IntegerType(val width: Int, val sign: Signedness)
    extends TypeAttribute("builtin.int_type") {
  override def toString = sign match {
    case Signless => s"i$width"
    case Signed   => s"si$width"
    case Unsigned => s"ui$width"
  }
}

case object IndexType extends TypeAttribute("builtin.index") {
  override def toString = "index"
}

case class ArrayAttribute(val attrValues: Seq[Attribute])
    extends Attribute("builtin.array_attribute") {
  override def toString =
    "[" + attrValues.map(x => x.toString).mkString(", ") + "]"
}

// shortened definition, does not include type information
case class StringAttribute(val stringLiteral: String)
    extends Attribute("builtin.string_attribute") {
  override def toString = "\"" + stringLiteral + "\""
}

case class RankedTensorType(
    val dimensionList: Seq[Int],
    val typ: Attribute,
    val encoding: Option[Attribute]
) extends TypeAttribute("builtin.ranked_tensor") {

  override def toString: String = {

    val shapeString =
      (dimensionList.map(x => if (x == -1) "?" else x.toString) :+ typ.toString)
        .mkString("x")

    val encodingString = encoding match {
      case Some(x) => x.toString
      case None    => ""
    }

    return s"tensor<${shapeString}${encodingString}>"
  }
}

case class UnrankedTensorType(val typ: Attribute)
    extends TypeAttribute("builtin.unranked_tensor") {
  override def toString = s"tensor<*x${typ.toString}>"
}

object AttrParser {

  ////////////////
  // FLOAT TYPE //
  ////////////////

  def Float16TypeP[$: P]: P[Attribute] = P("f16".!).map(_ => Float16Type)
  def Float32TypeP[$: P]: P[Attribute] = P("f32".!).map(_ => Float32Type)
  def Float64TypeP[$: P]: P[Attribute] = P("f64".!).map(_ => Float64Type)
  def Float80TypeP[$: P]: P[Attribute] = P("f80".!).map(_ => Float80Type)
  def Float128TypeP[$: P]: P[Attribute] = P("f128".!).map(_ => Float128Type)

  //////////////////
  // INTEGER TYPE //
  //////////////////

  def SignedIntegerTypeP[$: P]: P[Attribute] =
    P("si" ~~ CharIn("0-9").rep(1).!).map((intString: String) =>
      IntegerType(intString.toInt, Signed)
    )
  def UnsignedIntegerTypeP[$: P]: P[Attribute] =
    P("ui" ~~ CharIn("0-9").rep(1).!).map((intString: String) =>
      IntegerType(intString.toInt, Unsigned)
    )
  def SignlessIntegerTypeP[$: P]: P[Attribute] =
    P("i" ~~ CharIn("0-9").rep(1).!).map((intString: String) =>
      IntegerType(intString.toInt, Signless)
    )
  def IntegerTypeP[$: P]: P[Attribute] = P(
    SignedIntegerTypeP | UnsignedIntegerTypeP | SignlessIntegerTypeP
  )

  ////////////////
  // INDEX TYPE //
  ////////////////

  def IndexTypeP[$: P]: P[Attribute] = P("index".!).map(_ => IndexType)

  /////////////////////
  // ARRAY ATTRIBUTE //
  /////////////////////

  def ArrayAttributeP[$: P]: P[Attribute] = P(
    "[" ~ (AttributeValue
      .rep(sep = ","))
      .map((x: Seq[Attribute]) => ArrayAttribute(attrValues = x)) ~ "]"
  )

  //////////////////////
  // STRING ATTRIBUTE //
  //////////////////////

  def StringAttributeP[$: P]: P[Attribute] = P(
    Parser.StringLiteral.map((x: String) => StringAttribute(stringLiteral = x))
  ) // shortened definition omits typing information

  /////////////////
  // MEMREF TYPE //
  /////////////////

  // memref type
  // memref-type ::= 'memref' ~ '<' ~ ( ranked-memref-type | unranked-memref-type ) ~ '>'

  // ranked-memref-type ::= shape type (`,` layout-specification)? (`,` memory-space)?

  // 'Unranked' memref type
  // unranked-memref-type ::= '*' ~ 'x'.? ~ type ~ (`,` memory-space)?

  // layout-specification ::= attribute-value
  // memory-space ::= attribute-value
  // shape ::= ranked-shape | unranked-shape
  // ranked-shape ::= (dimension `x`)* type
  // unranked-shape ::= `*`x type
  // dimension ::= `?` | decimal-literal

  /////////////////
  // TENSOR TYPE //
  /////////////////

  // tensor-type           ::=   `tensor` `<` (ranked-tensor-type | unranked-tensor-type) `>`
  // ranked-tensor-type    ::=   dimension-list type (`,` encoding)?
  // unranked-tensor-type  ::=   `*` `x` type
  // dimension-list        ::=   (dimension `x`)*
  // dimension             ::=   `?` | decimal-literal
  // encoding              ::=   attribute-value

  def TensorTypeP[$: P]: P[Attribute] = P(
    "tensor" ~ "<" ~/ (UnrankedTensorTypeP | RankedTensorTypeP) ~ ">"
  )
  def RankedTensorTypeP[$: P]: P[Attribute] = P(
    DimensionList ~ Type ~ ("," ~ Encoding).?
  ).map((x: (Seq[Int], Attribute, Option[Attribute])) =>
    RankedTensorType(dimensionList = x._1, typ = x._2, encoding = x._3)
  )

  def UnrankedTensorTypeP[$: P]: P[Attribute] =
    P("*" ~ "x" ~ Type).map((x: Attribute) => UnrankedTensorType(typ = x))

  def DimensionList[$: P] = P((Dimension ~ "x").rep)

  def Dimension[$: P] = P("?".!.map(_ => -1) | DecimalLiteral)

  def Encoding[$: P] = P(AttributeValue)

  //////////////
  // BUILT IN //
  //////////////

  def BuiltIn[$: P]: P[Attribute] = P(
    Float16TypeP | Float32TypeP | Float64TypeP | Float80TypeP | Float128TypeP | IntegerTypeP | IndexTypeP | ArrayAttributeP | StringAttributeP | TensorTypeP
  )

  def main(args: Array[String]): Unit = {
    val Parsed.Success(x, y) = parse("f16", BuiltIn(_))
    println(x.getClass)
    println("Hello World!")
  }
}
