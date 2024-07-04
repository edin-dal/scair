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

  // signed-integer-type    ::=  `si` [1-9][0-9]*
  // unsigned-integer-type  ::=  `ui` [1-9][0-9]*
  // signless-integer-type  ::=  `i`  [1-9][0-9]*
  // integer-type           ::=  signed-integer-type | unsigned-integer-type | signless-integer-type

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

  // array-attribute  ::=  `[` (attribute-value (`,` attribute-value)*)? `]`

  def ArrayAttributeP[$: P]: P[Attribute] = P(
    "[" ~ (AttributeValue
      .rep(sep = ","))
      .map((x: Seq[Attribute]) => ArrayAttribute(attrValues = x)) ~ "]"
  )

  //////////////////////
  // STRING ATTRIBUTE //
  //////////////////////

  // string-attribute ::= string-literal (`:` type)?

  def StringAttributeP[$: P]: P[Attribute] = P(
    Parser.StringLiteral.map((x: String) => StringAttribute(stringLiteral = x))
  ) // shortened definition omits typing information

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
  ).map((x: (ArrayAttribute[IntAttr], Attribute, Option[Attribute])) =>
    RankedTensorType(
      dimensionList = x._1,
      typ = x._2,
      encoding = x._3
    )
  )

  def UnrankedTensorTypeP[$: P]: P[Attribute] =
    P("*" ~ "x" ~ Type).map((x: Attribute) => UnrankedTensorType(typ = x))

  def DimensionList[$: P] =
    P((Dimension ~ "x").rep).map(x => ArrayAttribute(attrValues = x))

  def Dimension[$: P]: P[IntAttr] =
    P("?".!.map(_ => -1) | DecimalLiteral).map(x => IntAttr(x))

  def Encoding[$: P] = P(AttributeValue)

  //////////////
  // BUILT IN //
  //////////////

  def BuiltIn[$: P]: P[Attribute] = P(
    Float16TypeP | Float32TypeP | Float64TypeP | Float80TypeP | Float128TypeP | IntegerTypeP | IndexTypeP | ArrayAttributeP | StringAttributeP | TensorTypeP
  )
}

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
