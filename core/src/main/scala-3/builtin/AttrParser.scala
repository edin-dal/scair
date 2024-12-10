package scair

import fastparse.*
import scair.Parser.*
import scair.dialects.affine.AffineMapP
import scair.dialects.affine.AffineSetP
import scair.dialects.builtin.*
import scair.ir.*

import java.lang.Float.intBitsToFloat

class AttrParser(val ctx: MLContext) {

  def DialectAttribute[$: P]: P[Attribute] = P(
    "#" ~ PrettyDialectReferenceName.flatMap { (x: String, y: String) =>
      ctx.getAttribute(s"${x}.${y}") match {
        case Some(y) =>
          y.parse(this)
        case None =>
          throw new Exception(
            s"Type ${x} is not defined in any supported Dialect."
          )
      }
    }
  )

  def DialectType[$: P]: P[Attribute] = P(
    "!" ~ PrettyDialectReferenceName.flatMap { (x: String, y: String) =>
      ctx.getAttribute(s"${x}.${y}") match {
        case Some(y) =>
          y.parse(this)
        case None =>
          throw new Exception(
            s"Type ${x} is not defined in any supported Dialect."
          )
      }
    }
  )

  // [x] - attribute-entry ::= (bare-id | string-literal) `=` attribute-value
  // [x] - attribute-value ::= attribute-alias | dialect-attribute | builtin-attribute

  def AttributeEntry[$: P] = P(
    (BareId | StringLiteral) ~ "=" ~/ AttributeValue
  )

  def AttributeValue[$: P] = P(
    Type // AttrParser.BuiltIn | DialectAttribute // | AttributeAlias //
  )

  def Type[$: P] = P(
    (BuiltIn | DialectType | DialectAttribute)./
  ) // shortened definition TODO: finish...

  ////////////////
  // FLOAT TYPE //
  ////////////////

  def Float16TypeP[$: P]: P[FloatType] = P("f16".!).map(_ => Float16Type)
  def Float32TypeP[$: P]: P[FloatType] = P("f32".!).map(_ => Float32Type)
  def Float64TypeP[$: P]: P[FloatType] = P("f64".!).map(_ => Float64Type)
  def Float80TypeP[$: P]: P[FloatType] = P("f80".!).map(_ => Float80Type)
  def Float128TypeP[$: P]: P[FloatType] = P("f128".!).map(_ => Float128Type)
  def FloatTypeP[$: P]: P[FloatType] = P(
    Float16TypeP | Float32TypeP | Float64TypeP | Float80TypeP | Float128TypeP
  )

  //////////////
  // INT DATA //
  //////////////

  def IntDataP[$: P]: P[IntData] = P(IntegerLiteral).map(IntData(_))

  //////////////////
  // INTEGER TYPE //
  //////////////////

  // signed-integer-type    ::=  `si` [1-9][0-9]*
  // unsigned-integer-type  ::=  `ui` [1-9][0-9]*
  // signless-integer-type  ::=  `i`  [1-9][0-9]*
  // integer-type           ::=  signed-integer-type | unsigned-integer-type | signless-integer-type

  def SignedIntegerTypeP[$: P]: P[IntegerType] =
    P("si" ~~ DecimalLiteral).map((x: Long) => IntegerType(IntData(x), Signed))

  def UnsignedIntegerTypeP[$: P]: P[IntegerType] =
    P("ui" ~~ DecimalLiteral).map((x: Long) =>
      IntegerType(IntData(x), Unsigned)
    )

  def SignlessIntegerTypeP[$: P]: P[IntegerType] =
    P("i" ~~ DecimalLiteral).map((x: Long) => IntegerType(IntData(x), Signless))

  def IntegerTypeP[$: P]: P[IntegerType] = P(
    SignedIntegerTypeP | UnsignedIntegerTypeP | SignlessIntegerTypeP
  )

  //////////////////
  // INTEGER ATTR //
  //////////////////

  def IntegerAttrP[$: P]: P[IntegerAttr] =
    P(
      (IntDataP ~ (":" ~ (IntegerTypeP | IndexTypeP)).?).map((x, y) =>
        IntegerAttr(
          x,
          y match {
            case x: Some[IntegerType]    => x.get
            case y: Some[IndexType.type] => y.get
            case None                    => I64
          }
        )
      )
        | "true".map(_ => IntegerAttr(IntData(1), I1))
        | "false".map(_ => IntegerAttr(IntData(0), I1))
    )

  //////////////
  // INT DATA //
  //////////////

  def FloatDataP[$: P]: P[FloatData] = P(FloatLiteral).map(FloatData(_))

  //////////////////
  // FLOAT ATTR //
  //////////////////

  def FloatAttrP[$: P]: P[FloatAttr] =
    P(
      (FloatDataP ~ (":" ~ FloatTypeP).?).map((x, y) =>
        FloatAttr(
          x,
          y match {
            case Some(a) => a
            case None    => Float64Type
          }
        )
      ) | (HexadecimalLiteral ~ ":" ~ FloatTypeP).map((x, y) =>
        FloatAttr(FloatData(intBitsToFloat(x.intValue())), y)
      )
    )

  ////////////////
  // INDEX TYPE //
  ////////////////

  def IndexTypeP[$: P]: P[IndexType.type] = P("index".!).map(_ => IndexType)

  /////////////////////
  // ARRAY ATTRIBUTE //
  /////////////////////

  // array-attribute  ::=  `[` (attribute-value (`,` attribute-value)*)? `]`

  def ArrayAttributeP[$: P]: P[ArrayAttribute[Attribute]] = P(
    "[" ~ (AttributeValue
      .rep(sep = ","))
      .map((x: Seq[Attribute]) => ArrayAttribute(attrValues = x)) ~ "]"
  )

  /////////////////////
  // DENSE ARRAY ATTRIBUTE //
  /////////////////////

  // dense-array-attribute  ::=  `array` `<` (integer-type | float-type) (`:` tensor-literal)? `>`

  def DenseArrayAttributeP[$: P]: P[DenseArrayAttr] = P(
    "array<" ~ (((IntegerTypeP) ~ (":" ~ IntDataP.rep(sep = ",")).?.map(
      _.getOrElse(Seq())
    )).map((typ: IntegerType, x: Seq[IntData]) =>
      DenseArrayAttr(typ, x.map(IntegerAttr(_, typ)))
    ) | ((FloatTypeP) ~ (":" ~ FloatDataP.rep(sep = ",")).?.map(
      _.getOrElse(Seq())
    )).map((typ: FloatType, x: Seq[FloatData]) =>
      DenseArrayAttr(typ, x.map(FloatAttr(_, typ)))
    )) ~ ">"
  )

  //////////////////////
  // STRING ATTRIBUTE //
  //////////////////////

  // string-attribute ::= string-literal (`:` type)?

  def StringAttributeP[$: P]: P[StringData] = P(
    Parser.StringLiteral.map((x: String) => StringData(stringLiteral = x))
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

  def TensorTypeP[$: P]: P[TensorType] = P(
    "tensor" ~ "<" ~/ (UnrankedTensorTypeP | RankedTensorTypeP) ~ ">"
  )

  def RankedTensorTypeP[$: P]: P[TensorType] = P(
    DimensionList ~ Type ~ ("," ~ Encoding).?
  ).map((x: (ArrayAttribute[IntData], Attribute, Option[Attribute])) =>
    RankedTensorType(
      dimensionList = x._1,
      typ = x._2,
      encoding = x._3
    )
  )

  def UnrankedTensorTypeP[$: P]: P[TensorType] =
    P("*" ~ "x" ~ Type).map((x: Attribute) => UnrankedTensorType(typ = x))

  def DimensionList[$: P] =
    P((Dimension ~ "x").rep).map(x => ArrayAttribute(attrValues = x))

  def Dimension[$: P]: P[IntData] =
    P("?".!.map(_ => -1: Long) | DecimalLiteral).map(x => IntData(x))

  def Encoding[$: P] = P(AttributeValue)

  /////////////////
  // MEMREF TYPE //
  /////////////////

  // memref-type           ::=   `memref` `<` (ranked-memref-type | unranked-memref-type) `>`
  // ranked-memref-type    ::=   dimension-list type
  // unranked-memref-type  ::=   `*` `x` type
  // dimension-list        ::=   (dimension `x`)*
  // dimension             ::=   `?` | decimal-literal

  def MemrefTypeP[$: P]: P[MemrefType] = P(
    "memref" ~ "<" ~/ (UnrankedMemrefTypeP | RankedMemrefTypeP) ~ ">"
  )

  def RankedMemrefTypeP[$: P]: P[MemrefType] = P(
    DimensionList ~ Type ~ ("," ~ Encoding).?
  ).map((x: (ArrayAttribute[IntData], Attribute, Option[Attribute])) =>
    RankedMemrefType(
      dimensionList = x._1.data,
      typ = x._2
    )
  )

  def UnrankedMemrefTypeP[$: P]: P[MemrefType] =
    P("*" ~ "x" ~ Type).map((x: Attribute) => UnrankedMemrefType(typ = x))

  //////////////////////////
  // SYMBOL REF ATTRIBUTE //
  //////////////////////////

  def SymbolRefAttrP[$: P]: P[SymbolRefAttr] = P(
    SymbolRefId ~~ ("::" ~~ SymbolRefId).rep
  ).map((x: String, y: Seq[String]) =>
    SymbolRefAttr(
      StringData(x),
      y.map(z => StringData(z))
    )
  )

  //////////////////////////////
  // DenseIntOrFPElementsAttr //
  //////////////////////////////

  // TO-DO : it can also parse a vector type or a memref type
  // TO-DO : Figure out why it is throwing an error when you get rid of asInstanceOf...

  def DenseIntOrFPElementsAttrP[$: P]: P[DenseIntOrFPElementsAttr] =
    P("dense" ~ "<" ~ TensorLiteral ~ ">" ~ ":" ~ TensorTypeP).map((x, y) =>
      DenseIntOrFPElementsAttr(y, x.asInstanceOf[TensorLiteralArray])
    )

  def TensorLiteral[$: P]: P[TensorLiteralArray] =
    P(SingleTensorLiteral | EmptyTensorLiteral | MultipleTensorLiteral)

  def SingleTensorLiteral[$: P]: P[TensorLiteralArray] =
    P(FloatDataP | IntDataP).map(_ match {
      case (x: IntData) =>
        ArrayAttribute[IntegerAttr](Seq(IntegerAttr(x, I32)))
      case (y: FloatData) =>
        ArrayAttribute[FloatAttr](Seq(FloatAttr(y, Float32Type)))
    })

  def MultipleTensorLiteral[$: P]: P[TensorLiteralArray] =
    P(MultipleFloatTensorLiteral | MultipleIntTensorLiteral)

  def MultipleIntTensorLiteral[$: P]: P[ArrayAttribute[IntegerAttr]] =
    P("[" ~ IntDataP.rep(1, sep = ",") ~ "]").map((x: Seq[IntData]) =>
      ArrayAttribute[IntegerAttr](
        for (x1 <- x) yield IntegerAttr(x1, I32)
      )
    )

  def MultipleFloatTensorLiteral[$: P]: P[ArrayAttribute[FloatAttr]] =
    P("[" ~ FloatDataP.rep(1, sep = ",") ~ "]").map((y: Seq[FloatData]) =>
      ArrayAttribute[FloatAttr](
        for (y1 <- y) yield FloatAttr(y1, Float32Type)
      )
    )

  def EmptyTensorLiteral[$: P]: P[TensorLiteralArray] =
    P("[" ~ "]").map(_ => ArrayAttribute[IntegerAttr](Seq()))

  /////////////////////
  // AFFINE MAP ATTR //
  /////////////////////

  def AffineMapAttrP[$: P]: P[AffineMapAttr] =
    P("affine_map" ~ "<" ~ AffineMapP ~ ">").map(AffineMapAttr(_))

  /////////////////////
  // AFFINE SET ATTR //
  /////////////////////

  def AffineSetAttrP[$: P]: P[AffineSetAttr] =
    P("affine_set" ~ "<" ~ AffineSetP ~ ">").map(AffineSetAttr(_))

  //////////////
  // BUILT IN //
  //////////////

  ///////////////////
  // FUNCTION TYPE //
  ///////////////////

  def ParenTypeList[$: P] = P(
    "(" ~ Type.rep(sep = ",") ~ ")"
  )

  def FunctionType[$: P] = P(
    ParenTypeList ~ "->" ~ (ParenTypeList | Type.rep(exactly = 1))
  )

  def FunctionTypeP[$: P]: P[FunctionType] = P(
    FunctionType.map((inputs, outputs) =>
      scair.dialects.builtin.FunctionType(inputs, outputs)
    )
  )

  def BuiltIn[$: P]: P[Attribute] = P(
    FloatTypeP |
      IntegerTypeP |
      IndexTypeP |
      ArrayAttributeP |
      DenseArrayAttributeP |
      FunctionTypeP |
      StringAttributeP |
      TensorTypeP |
      MemrefTypeP |
      SymbolRefAttrP |
      FloatAttrP |
      IntegerAttrP |
      DenseIntOrFPElementsAttrP |
      AffineMapAttrP | AffineSetAttrP
  )

  def main(args: Array[String]): Unit = {
    val parsed =
      parse("dense<[10.0, 11.0]> : tensor<2xf32>", DenseIntOrFPElementsAttrP(_))
    println(parsed)
  }
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
