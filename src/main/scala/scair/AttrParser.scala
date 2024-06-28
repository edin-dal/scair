package scair

import fastparse._
import scala.collection.mutable
import scala.util.{Try, Success, Failure}
import IR._
import Parser._

import scair.dialects.builtin._
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
    "[" ~ (BuiltIn
      .rep(sep = ","))
      .map((x: Seq[Attribute]) => ArrayAttribute(attrValues = x)) ~ "]"
  )

  //////////////////////
  // STRING ATTRIBUTE //
  //////////////////////

  def StringAttributeP[$: P]: P[Attribute] = P(
    Parser.StringLiteral.map((x: String) => StringAttribute(stringLiteral = x))
  ) // shortened definition omits typing information

  //////////////////////////
  // DICTIONARY ATTRIBUTE //
  //////////////////////////

  def BuiltIn[$: P]: P[Attribute] = P(
    Float16TypeP | Float32TypeP | Float64TypeP | Float80TypeP | Float128TypeP | IntegerTypeP | IndexTypeP | ArrayAttributeP | StringAttributeP
  )

  def main(args: Array[String]): Unit = {
    val Parsed.Success(x, y) = parse("f16", BuiltIn(_))
    println(x.getClass)
    println("Hello World!")
  }
}
