package scair

import scala.annotation.switch

import fastparse.*
import scair.Parser.*
import scair.dialects.affine.AffineMapP
import scair.dialects.affine.AffineSetP
import scair.dialects.builtin.*
import scair.ir.*

import java.lang.Float.intBitsToFloat
import scala.collection.mutable

// ░█████╗░ ████████╗ ████████╗ ██████╗░
// ██╔══██╗ ╚══██╔══╝ ╚══██╔══╝ ██╔══██╗
// ███████║ ░░░██║░░░ ░░░██║░░░ ██████╔╝
// ██╔══██║ ░░░██║░░░ ░░░██║░░░ ██╔══██╗
// ██║░░██║ ░░░██║░░░ ░░░██║░░░ ██║░░██║
// ╚═╝░░╚═╝ ░░░╚═╝░░░ ░░░╚═╝░░░ ╚═╝░░╚═╝

// ██████╗░ ░█████╗░ ██████╗░ ░██████╗ ███████╗ ██████╗░
// ██╔══██╗ ██╔══██╗ ██╔══██╗ ██╔════╝ ██╔════╝ ██╔══██╗
// ██████╔╝ ███████║ ██████╔╝ ╚█████╗░ █████╗░░ ██████╔╝
// ██╔═══╝░ ██╔══██║ ██╔══██╗ ░╚═══██╗ ██╔══╝░░ ██╔══██╗
// ██║░░░░░ ██║░░██║ ██║░░██║ ██████╔╝ ███████╗ ██║░░██║
// ╚═╝░░░░░ ╚═╝░░╚═╝ ╚═╝░░╚═╝ ╚═════╝░ ╚══════╝ ╚═╝░░╚═╝
//
// IE THE PARSER FOR BUILTIN DIALECT ATTRIBUTES

class AttrParser(
    val ctx: MLContext,
    val attributeAliases: mutable.Map[String, Attribute] = mutable.Map.empty,
    val typeAliases: mutable.Map[String, Attribute] = mutable.Map.empty
) {

  def DialectAttribute[$: P]: P[Attribute] = P(
    "#" ~~ PrettyDialectReferenceName.flatMapTry {
      (dialect: String, attrName: String) =>
        ctx.getAttribute(s"${dialect}.${attrName}") match {
          case Some(attr) =>
            attr.parse(this)
          case None =>
            throw new Exception(
              s"Attribute $dialect.$attrName is not defined in any supported Dialect."
            )
        }
    }
  )

  def DialectType[$: P]: P[Attribute] = P(
    "!" ~~ PrettyDialectReferenceName.flatMapTry {
      (dialect: String, attrName: String) =>
        ctx.getAttribute(s"${dialect}.${attrName}") match {
          case Some(attr) =>
            attr.parse(this)
          case None =>
            throw new Exception(
              s"Type $dialect.$attrName is not defined in any supported Dialect."
            )
        }
    }
  )

  // [x] - attribute-entry ::= (bare-id | string-literal) `=` attribute-value
  // [x] - attribute-value ::= attribute-alias | dialect-attribute | builtin-attribute

  def AttributeEntry[$: P] = P(
    (BareId | StringLiteral) ~ "=" ~/ (Attribute | Type)
  )

  def Attribute[$: P] = P(
    Type | BuiltinAttr | DialectAttribute | AttributeAlias // AttrParser.BuiltIn | DialectAttribute // | AttributeAlias //
  )

  def AttributeAlias[$: P] = P("#" ~~ AliasName).map((name: String) =>
    attributeAliases.getOrElse(
      name,
      throw new Exception(s"Attribute alias ${name} not defined.")
    )
  )

  def TypeAlias[$: P] = P("!" ~~ AliasName).map((name: String) =>
    typeAliases.getOrElse(
      name,
      throw new Exception(s"Type alias ${name} not defined.")
    )
  )

  def AttributeList[$: P] = Attribute.rep(sep = ",")

  def Type[$: P] = P(
    (BuiltinType | DialectType | TypeAlias)
  )

  def TypeList[$: P] = Type.rep(sep = ",")

  /*≡==--==≡≡≡≡==--=≡≡*\
  ||    FLOAT TYPE    ||
  \*≡==---==≡≡==---==≡*/

  def Float16TypeP[$: P]: P[Float16Type] = P("f16").map(_ => Float16Type())
  def Float32TypeP[$: P]: P[Float32Type] = P("f32").map(_ => Float32Type())
  def Float64TypeP[$: P]: P[Float64Type] = P("f64").map(_ => Float64Type())
  def Float80TypeP[$: P]: P[Float80Type] = P("f80").map(_ => Float80Type())
  def Float128TypeP[$: P]: P[Float128Type] = P("f128").map(_ => Float128Type())

  def FloatTypeP[$: P]: P[FloatType] = P(
    Float16TypeP | Float32TypeP | Float64TypeP | Float80TypeP | Float128TypeP
  )

  /*≡==--==≡≡≡≡==--=≡≡*\
  ||     INT DATA     ||
  \*≡==---==≡≡==---==≡*/

  def IntDataP[$: P]: P[IntData] = P(IntegerLiteral).map(IntData(_))

  /*≡==--==≡≡≡≡==--=≡≡*\
  ||   INTEGER TYPE   ||
  \*≡==---==≡≡==---==≡*/

  // signed-integer-type    ::=  `si` [1-9][0-9]*
  // unsigned-integer-type  ::=  `ui` [1-9][0-9]*
  // signless-integer-type  ::=  `i`  [1-9][0-9]*
  // integer-type           ::=  signed-integer-type | unsigned-integer-type | signless-integer-type

  def SignedIntegerTypeP[$: P]: P[IntegerType] =
    P("si" ~~ DecimalLiteral).map((x: BigInt) =>
      IntegerType(IntData(x), Signed)
    )

  def UnsignedIntegerTypeP[$: P]: P[IntegerType] =
    P("ui" ~~ DecimalLiteral).map((x: BigInt) =>
      IntegerType(IntData(x), Unsigned)
    )

  def SignlessIntegerTypeP[$: P]: P[IntegerType] =
    P("i" ~~ DecimalLiteral).map((x: BigInt) =>
      IntegerType(IntData(x), Signless)
    )

  def IntegerTypeP[$: P]: P[IntegerType] = P(
    (StringIn("i", "ui", "si").! ~~ DecimalLiteral).map((prefix: String, x: BigInt) =>
      prefix.head: @switch match
        case 'i'  => IntegerType(IntData(x), Signless)
        case 'u' => IntegerType(IntData(x), Unsigned)
        case 's' => IntegerType(IntData(x), Signed)
      )
  )

  /*≡==--==≡≡≡≡==--=≡≡*\
  ||   INTEGER ATTR   ||
  \*≡==---==≡≡==---==≡*/

  def IntegerAttrP[$: P]: P[IntegerAttr] =
    P(
      (IntDataP ~ (":" ~ (IntegerTypeP | IndexTypeP)).?).map((x, y) =>
        IntegerAttr(
          x,
          y match {
            case yy: Some[_] =>
              yy.get match
                case i: IntegerType => i
                case i: IndexType   => i
                case _              =>
                  throw new Exception(
                    s"Unreachable, fastparse's | is simply weakly typed."
                  )

            case None => I64
          }
        )
      )
        | "true".map(_ => IntegerAttr(IntData(1), I1))
        | "false".map(_ => IntegerAttr(IntData(0), I1))
    )

  /*≡==--==≡≡≡≡==--=≡≡*\
  ||    FLOAT DATA    ||
  \*≡==---==≡≡==---==≡*/

  def FloatDataP[$: P]: P[FloatData] = P(FloatLiteral).map(FloatData(_))

  /*≡==--==≡≡≡≡==--=≡≡*\
  ||    FLOAT ATTR    ||
  \*≡==---==≡≡==---==≡*/

  def FloatAttrP[$: P]: P[FloatAttr] =
    P(
      (FloatDataP ~ (":" ~ FloatTypeP).?).map((x, y) =>
        FloatAttr(
          x,
          y match {
            case Some(a) => a
            case None    => Float64Type()
          }
        )
      ) | (HexadecimalLiteral ~ ":" ~ FloatTypeP).map((x, y) =>
        FloatAttr(FloatData(intBitsToFloat(x.intValue())), y)
      )
    )

  /*≡==--==≡≡≡≡==--=≡≡*\
  ||    INDEX TYPE    ||
  \*≡==---==≡≡==---==≡*/

  def IndexTypeP[$: P]: P[IndexType] = P("index").map(_ => IndexType())

  /*≡==--==≡≡≡≡≡==--=≡≡*\
  ||  ARRAY ATTRIBUTE  ||
  \*≡==---==≡≡≡==---==≡*/

  // array-attribute  ::=  `[` (attribute-value (`,` attribute-value)*)? `]`

  def ArrayAttributeP[$: P]: P[ArrayAttribute[Attribute]] = P(
    "[" ~ AttributeList
      .map((x: Seq[Attribute]) => ArrayAttribute(attrValues = x)) ~ "]"
  )

  /*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
  || DICTIONARY ATTRIBUTE  ||
  \*≡==---==≡≡≡≡≡≡≡==---==≡*/

  def DictionaryAttributeP[$: P]: P[DictionaryAttr] = P(
    "{" ~ AttributeEntry.rep(sep = ",") ~ "}"
  ).map((entries: Seq[(String, Attribute)]) =>
    DictionaryAttr(entries.map((name, attr) => (name, attr)).toMap)
  )

  /*≡==--==≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
  ||  DENSE ARRAY ATTRIBUTE  ||
  \*≡==---==≡≡≡≡≡≡≡≡≡==---==≡*/

  // dense-array-attribute  ::=  `array` `<` (integer-type | float-type) (`:` tensor-literal)? `>`

  def DenseArrayAttributeP[$: P]: P[DenseArrayAttr] = P(
    "array<" ~ (((IntegerTypeP) ~ (":" ~ IntDataP.rep(sep = ",")).orElse(
      Seq()
    )).map((typ: IntegerType, x: Seq[IntData]) =>
      DenseArrayAttr(typ, x.map(IntegerAttr(_, typ)))
    ) | ((FloatTypeP) ~ (":" ~ FloatDataP.rep(sep = ",")).orElse(Seq())).map(
      (typ: FloatType, x: Seq[FloatData]) =>
        DenseArrayAttr(typ, x.map(FloatAttr(_, typ)))
    )) ~ ">"
  )

  /*≡==--==≡≡≡≡≡≡==--=≡≡*\
  ||  STRING ATTRIBUTE  ||
  \*≡==---==≡≡≡≡==---==≡*/

  // string-attribute ::= string-literal (`:` type)?

  def StringAttributeP[$: P]: P[StringData] = P(
    Parser.StringLiteral.map((x: String) => StringData(stringLiteral = x))
  ) // shortened definition omits typing information

  /*≡==--==≡≡≡==--=≡≡*\
  ||   TENSOR TYPE   ||
  \*≡==---==≡==---==≡*/

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
      shape = x._1,
      elementType = x._2,
      encoding = x._3
    )
  )

  def UnrankedTensorTypeP[$: P]: P[TensorType] =
    P("*" ~ "x" ~ Type).map((x: Attribute) =>
      UnrankedTensorType(elementType = x)
    )

  def DimensionList[$: P] =
    P((Dimension ~ "x").rep).map(x => ArrayAttribute(attrValues = x))

  def Dimension[$: P]: P[IntData] =
    P("?".map(_ => -1: BigInt) | DecimalLiteral).map(x => IntData(x))

  def Encoding[$: P] = P(Attribute)

  /*≡==--==≡≡≡≡≡==--=≡≡*\
  ||    MEMREF TYPE    ||
  \*≡==---==≡≡≡==---==≡*/

  // memref-type           ::=   `memref` `<` (ranked-memref-type | unranked-memref-type) `>`
  // ranked-memref-type    ::=   dimension-list type
  // unranked-memref-type  ::=   `*` `x` type
  // dimension-list        ::=   (dimension `x`)*
  // dimension             ::=   `?` | decimal-literal

  def MemrefTypeP[$: P]: P[MemrefType] = P(
    "memref" ~ "<" ~/ (UnrankedMemrefTypeP | RankedMemrefTypeP) ~ ">"
  )

  def RankedMemrefTypeP[$: P]: P[MemrefType] = P(
    DimensionList ~ Type
  ).map((x: (ArrayAttribute[IntData], Attribute)) =>
    RankedMemrefType(
      shape = x._1,
      elementType = x._2
    )
  )

  def UnrankedMemrefTypeP[$: P]: P[UnrankedMemrefType] =
    P("*" ~ "x" ~ Type).map((x: Attribute) =>
      UnrankedMemrefType(elementType = x)
    )

  def VectorDimensionList[$: P] =
    P(
      ((DecimalLiteral
        .map(x => (IntData(x), IntData(0))) | "[" ~ DecimalLiteral.map(x =>
        (IntData(x), IntData(1))
      ) ~ "]") ~ "x").rep(1).map(_.unzip)
    )

  def VectorTypeP[$: P]: P[VectorType] = P(
    "vector<" ~/ VectorDimensionList ~/ Type ~/ ">"
  ).map((shape: Seq[IntData], scalableDims: Seq[IntData], typ: Attribute) =>
    VectorType(
      shape = ArrayAttribute[IntData](shape),
      elementType = typ,
      scalableDims = ArrayAttribute[IntData](scalableDims)
    )
  )

  /*≡==--==≡≡≡≡≡≡≡==--=≡≡*\
  ||   SYMBOL REF ATTR   ||
  \*≡==---==≡≡≡≡≡==---==≡*/

  def SymbolRefAttrP[$: P]: P[SymbolRefAttr] = P(
    SymbolRefId ~~ ("::" ~~ SymbolRefId).rep
  ).map((x: String, y: Seq[String]) =>
    SymbolRefAttr(
      StringData(x),
      y.map(z => StringData(z))
    )
  )

  /*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
  ||   DenseIntOrFPElementsAttr   ||
  \*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

  // TO-DO : it can also parse a vector type or a memref type
  // TO-DO : Figure out why it is throwing an error when you get rid of asInstanceOf...

  def DenseIntOrFPElementsAttrP[$: P]: P[DenseIntOrFPElementsAttr] =
    P(
      "dense" ~ "<" ~ TensorLiteral ~ ">" ~ ":" ~ (TensorTypeP | MemrefTypeP | VectorTypeP)
    ).map((x, y) =>
      y match {
        case yy: (TensorType | MemrefType | VectorType) =>
          DenseIntOrFPElementsAttr(yy, x.asInstanceOf[TensorLiteralArray])
      }
      // DenseIntOrFPElementsAttr(y, x.asInstanceOf[TensorLiteralArray])
    )

  def TensorLiteral[$: P]: P[TensorLiteralArray] =
    P(SingleTensorLiteral | EmptyTensorLiteral | MultipleTensorLiteral)

  def SingleTensorLiteral[$: P]: P[TensorLiteralArray] =
    P(FloatDataP | IntDataP).map(_ match {
      case (x: IntData) =>
        ArrayAttribute[IntegerAttr](Seq(IntegerAttr(x, I32)))
      case (y: FloatData) =>
        ArrayAttribute[FloatAttr](Seq(FloatAttr(y, Float32Type())))
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
        for (y1 <- y) yield FloatAttr(y1, Float32Type())
      )
    )

  def EmptyTensorLiteral[$: P]: P[TensorLiteralArray] =
    P("[" ~ "]").map(_ => ArrayAttribute[IntegerAttr](Seq()))

  /*≡==--==≡≡≡≡≡≡≡==--=≡≡*\
  ||   AFFINE MAP ATTR   ||
  \*≡==---==≡≡≡≡≡==---==≡*/

  def AffineMapAttrP[$: P]: P[AffineMapAttr] =
    P("affine_map" ~ "<" ~ AffineMapP ~ ">").map(AffineMapAttr(_))

  /*≡==--==≡≡≡≡≡≡≡==--=≡≡*\
  ||   AFFINE SET ATTR   ||
  \*≡==---==≡≡≡≡≡==---==≡*/

  def AffineSetAttrP[$: P]: P[AffineSetAttr] =
    P("affine_set" ~ "<" ~ AffineSetP ~ ">").map(AffineSetAttr(_))

  /*≡==--==≡≡≡≡≡==--=≡≡*\
  ||   FUNCTION TYPE   ||
  \*≡==---==≡≡≡==---==≡*/

  def ParenTypeList[$: P] = P(
    "(" ~ Type.rep(sep = ",") ~ ")"
  )

  def FunctionType[$: P] = P(
    ParenTypeList ~ "->" ~/ (ParenTypeList | Type.map(Seq(_)))
  )

  def FunctionTypeP[$: P]: P[FunctionType] = P(
    FunctionType.map((inputs, outputs) =>
      scair.dialects.builtin.FunctionType(inputs, outputs)
    )
  )

  def BuiltinType[$: P]: P[Attribute] = P(
      IntegerTypeP |
      FloatTypeP |
      IndexTypeP |
      FunctionTypeP |
      TensorTypeP |
      MemrefTypeP |
      VectorTypeP
  )

  def BuiltinAttr[$: P]: P[Attribute] = P(
    ArrayAttributeP |
      DenseArrayAttributeP |
      StringAttributeP |
      SymbolRefAttrP |
      FloatAttrP |
      IntegerAttrP |
      DenseIntOrFPElementsAttrP |
      AffineMapAttrP |
      AffineSetAttrP
  )

}
