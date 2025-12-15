package scair.parse

import fastparse.*
import scair.*
import scair.dialects.builtin.*
import scair.dialects.builtin.VectorType
import scair.ir.*

import java.lang.Float.intBitsToFloat
import scala.annotation.tailrec

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

private def dialectAttributeP[$: P](using p: Parser): P[Attribute] =
  "#" ~~ prettyDialectReferenceNameP./.flatMapTry {
    (dialect: String, attrName: String) =>
      p.context.getAttrCompanion(s"$dialect.$attrName") match
        case Some(attr) =>
          attr.parse
        case None =>
          Fail(
            s"Attribute $dialect.$attrName is not defined in any supported Dialect."
          )
  }

private def dialectTypeP[$: P](using p: Parser): P[Attribute] =
  "!" ~~ prettyDialectReferenceNameP./.flatMapTry {
    (dialect: String, attrName: String) =>
      p.context.getAttrCompanion(s"$dialect.$attrName") match
        case Some(attr) =>
          attr.parse
        case None =>
          Fail(
            s"Type $dialect.$attrName is not defined in any supported Dialect."
          )
  }

// [x] - attribute-entry ::= (bare-id | string-literal) `=` attribute-value
// [x] - attribute-value ::= attribute-alias | dialect-attribute | builtin-attribute

private def attributeEntryP[$: P](using Parser) =
  (bareIdP | stringLiteralP) ~ "=" ~/ (attributeP)

def attributeP[$: P](using Parser) = P(
  typeP | builtinAttrP | dialectAttributeP | attributeAliasP
)

private def attributeAliasP[$: P](using p: Parser) =
  "#" ~~ aliasNameP.flatMap((name: String) =>
    p.attributeAliases.get(name) match
      case Some(attr) => Pass(attr)
      case None       =>
        Fail(s"Attribute alias $name not defined.")
  )

def attrOfOrP[A <: Attribute](default: A)(using
    Parser
)(using P[Any]) =
  attributeP.orElse(default).flatMap(_ match
    case attr: A => Pass(attr)
    case _       => Fail("Expected sumin, got sumin else"))

def attrOfP[A <: Attribute](using
    Parser
)(using P[Any]) =
  attributeP.flatMap(_ match
    case attr: A => Pass(attr)
    case _       => Fail("Expected sumin, got sumin else"))

def typeOfOrP[T <: TypeAttribute](default: T)(using
    Parser
)(using P[Any]) =
  typeP.orElse(default).flatMap(_ match
    case tpe: T => Pass(tpe)
    case _      => Fail("Expected sumin, got sumin else"))

def typeOfP[T <: TypeAttribute](using
    Parser
)(using P[Any]) =
  typeP.flatMap(_ match
    case tpe: T => Pass(tpe)
    case _      => Fail("Expected sumin, got sumin else"))

/*≡==--==≡≡≡==--=≡≡*\
||      TYPES      ||
\*≡==---==≡==---==≡*/

// [x] type ::= type-alias | dialect-type | builtin-type

// [x] type-list-no-parens ::=  type (`,` type)*
// [x] type-list-parens ::= `(` `)` | `(` type-list-no-parens `)`

// // This is a common way to refer to a value with a specified type.
// [ ] ssa-use-and-type ::= ssa-use `:` type
// [ ] ssa-use ::= value-use

// // Non-empty list of names and types.
// [ ] ssa-use-and-type-list ::= ssa-use-and-type (`,` ssa-use-and-type)*

// [x] function-type ::= (type | type-list-parens) `->` (type | type-list-parens)

def typeP[$: P](using Parser) =
  P(builtinTypeP | dialectTypeP | typeAliasP)

def typeListP[$: P](using Parser) = P(typeP.rep(sep = ","))

def parenTypeListP[$: P](using Parser) = P(
  "(" ~ typeListP ~ ")"
)

private def typeAliasP[$: P](using p: Parser) =
  "!" ~~ aliasNameP.flatMap((name: String) =>
    p.typeAliases.get(name) match
      case Some(attr) => Pass(attr)
      case None       => Fail(s"Type alias $name not defined.")
  )

/*≡==--==≡≡≡≡==--=≡≡*\
||    FLOAT TYPE    ||
\*≡==---==≡≡==---==≡*/

def float16TypeP[$: P](using Parser): P[Float16Type] =
  P("f16".map(_ => Float16Type()))

def float32TypeP[$: P](using Parser): P[Float32Type] =
  P("f32".map(_ => Float32Type()))

def float64TypeP[$: P](using Parser): P[Float64Type] =
  P("f64".map(_ => Float64Type()))

def float80TypeP[$: P](using Parser): P[Float80Type] =
  P("f80".map(_ => Float80Type()))

def float128TypeP[$: P](using Parser): P[Float128Type] =
  P("f128".map(_ => Float128Type()))

def floatTypeP[$: P](using Parser): P[FloatType] = P(
  float16TypeP | float32TypeP | float64TypeP | float80TypeP | float128TypeP
)

/*≡==--==≡≡≡≡==--=≡≡*\
||     INT DATA     ||
\*≡==---==≡≡==---==≡*/

def intDataP[$: P](using Parser): P[IntData] =
  P(integerLiteralP).map(IntData(_))

/*≡==--==≡≡≡≡==--=≡≡*\
||   INTEGER TYPE   ||
\*≡==---==≡≡==---==≡*/

// signed-integer-type    ::=  `si` [1-9][0-9]*
// unsigned-integer-type  ::=  `ui` [1-9][0-9]*
// signless-integer-type  ::=  `i`  [1-9][0-9]*
// integer-type           ::=  signed-integer-type | unsigned-integer-type | signless-integer-type

def integerTypeP[$: P](using Parser): P[IntegerType] = P(
  (("i".map(_ => Signless) | "si".map(_ => Signed) | "ui".map(_ => Unsigned)) ~~
    decimalLiteralP.map(IntData.apply))
    .map((sign, bits) => IntegerType.apply(bits, sign))
)

/*≡==--==≡≡≡≡==--=≡≡*\
||   INTEGER ATTR   ||
\*≡==---==≡≡==---==≡*/

def integerAttrP[$: P](using Parser): P[IntegerAttr] =
  P(
    (intDataP ~
      (":" ~ (integerTypeP | indexTypeP)
        .asInstanceOf[P[IntegerType | IndexType]]).orElse(I64))
      .map(IntegerAttr.apply) | "true".map(_ => IntegerAttr(IntData(1), I1)) |
      "false".map(_ => IntegerAttr(IntData(0), I1))
  )

/*≡==--==≡≡≡≡==--=≡≡*\
||    FLOAT DATA    ||
\*≡==---==≡≡==---==≡*/

def floatDataP[$: P](using Parser): P[FloatData] =
  P(floatLiteralP).map(FloatData(_))

/*≡==--==≡≡≡≡==--=≡≡*\
||    FLOAT ATTR    ||
\*≡==---==≡≡==---==≡*/

def floatAttrP[$: P](using Parser): P[FloatAttr] =
  P(
    (floatDataP ~ (":" ~ floatTypeP).orElse(Float64Type())).map((x, y) =>
      FloatAttr(
        x,
        y,
      )
    ) | (hexadecimalLiteralP ~ ":" ~ floatTypeP)
      .map((x, y) => FloatAttr(FloatData(intBitsToFloat(x.intValue())), y))
  )

/*≡==--==≡≡≡≡==--=≡≡*\
||    INDEX TYPE    ||
\*≡==---==≡≡==---==≡*/

def indexTypeP[$: P](using Parser): P[IndexType] =
  P("index".map(_ => IndexType()))
/*≡==--==≡≡≡≡≡≡==--=≡≡*\
||    COMPLEX TYPE    ||
\*≡==---==≡≡≡≡==---==≡*/

def complexTypeP[$: P](using Parser): P[ComplexType] =
  P("complex<" ~ (indexTypeP | integerTypeP | floatTypeP) ~ ">")
    .map((tpe: TypeAttribute) =>
      ComplexType(tpe.asInstanceOf[IntegerType | IndexType | FloatType])
    )

/*≡==--==≡≡≡≡≡==--=≡≡*\
||  ARRAY ATTRIBUTE  ||
\*≡==---==≡≡≡==---==≡*/

// array-attribute  ::=  `[` (attribute-value (`,` attribute-value)*)? `]`

def arrayAttributeP[$: P](using Parser): P[ArrayAttribute[Attribute]] = P(
  "[" ~ attributeP.rep(sep = ",")
    .map((x: Seq[Attribute]) => ArrayAttribute(attrValues = x)) ~ "]"
)

/*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
|| DICTIONARY ATTRIBUTE  ||
\*≡==---==≡≡≡≡≡≡≡==---==≡*/

def dictionaryAttributeP[$: P](using Parser): P[DictionaryAttr] =
  attributeDictionaryP
    .map(
      DictionaryAttr.apply
    )

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||  DENSE ARRAY ATTRIBUTE  ||
\*≡==---==≡≡≡≡≡≡≡≡≡==---==≡*/

// dense-array-attribute  ::=  `array` `<` (integer-type | float-type) (`:` tensor-literal)? `>`

def denseArrayAttributeP[$: P](using Parser): P[DenseArrayAttr] = P(
  "array<" ~
    (((integerTypeP) ~ (":" ~ intDataP.rep(sep = ",")).orElse(
      Seq()
    )).map((typ: IntegerType, x: Seq[IntData]) =>
      DenseArrayAttr(typ, x.map(IntegerAttr(_, typ)))
    ) | ((floatTypeP) ~ (":" ~ floatDataP.rep(sep = ",")).orElse(Seq()))
      .map((typ: FloatType, x: Seq[FloatData]) =>
        DenseArrayAttr(typ, x.map(FloatAttr(_, typ)))
      )) ~ ">"
)

/*≡==--==≡≡≡≡≡≡==--=≡≡*\
||  STRING ATTRIBUTE  ||
\*≡==---==≡≡≡≡==---==≡*/

// string-attribute ::= string-literal (`:` type)?

def stringAttributeP[$: P](using Parser): P[StringData] = P(
  stringLiteralP.map(StringData.apply)
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

def tensorTypeP[$: P](using Parser): P[TensorType] = P(
  "tensor" ~ "<" ~/ (unrankedTensorTypeP | rankedTensorTypeP) ~ ">"
)

def rankedTensorTypeP[$: P](using Parser): P[TensorType] = P(
  dimensionListP ~ typeP ~ ("," ~ encodingP).?
).map((x: (ArrayAttribute[IntData], Attribute, Option[Attribute])) =>
  RankedTensorType(
    shape = x._1,
    elementType = x._2,
    encoding = x._3,
  )
)

def unrankedTensorTypeP[$: P](using Parser): P[TensorType] =
  P("*" ~ "x" ~ typeP)
    .map((x: Attribute) => UnrankedTensorType(elementType = x))

def dimensionListP[$: P](using Parser) =
  P((dimensionP ~ "x").rep).map(x => ArrayAttribute(attrValues = x))

def dimensionP[$: P](using Parser): P[IntData] =
  P("?".map(_ => -1: BigInt) | decimalLiteralP).map(x => IntData(x))

def encodingP[$: P](using Parser) = P(attributeP)

/*≡==--==≡≡≡≡≡==--=≡≡*\
||    MEMREF TYPE    ||
\*≡==---==≡≡≡==---==≡*/

// memref-type           ::=   `memref` `<` (ranked-memref-type | unranked-memref-type) `>`
// ranked-memref-type    ::=   dimension-list type
// unranked-memref-type  ::=   `*` `x` type
// dimension-list        ::=   (dimension `x`)*
// dimension             ::=   `?` | decimal-literal

def memrefTypeP[$: P](using Parser): P[MemrefType] = P(
  "memref" ~ "<" ~/ (unrankedMemrefTypeP | rankedMemrefTypeP) ~ ">"
)

def rankedMemrefTypeP[$: P](using Parser): P[MemrefType] = P(
  dimensionListP ~ typeP
).map((x: (ArrayAttribute[IntData], Attribute)) =>
  RankedMemrefType(
    shape = x._1,
    elementType = x._2,
  )
)

def unrankedMemrefTypeP[$: P](using Parser): P[UnrankedMemrefType] =
  P("*" ~ "x" ~ typeP)
    .map((x: Attribute) => UnrankedMemrefType(elementType = x))

def vectorDimensionListP[$: P](using Parser) =
  P(
    ((decimalLiteralP.map(x => (IntData(x), IntData(0))) |
      "[" ~ decimalLiteralP.map(x => (IntData(x), IntData(1))) ~ "]") ~ "x")
      .rep(1).map(_.unzip)
  )

def vectorTypeP[$: P](using Parser): P[VectorType] = P(
  "vector<" ~/ vectorDimensionListP ~/ typeP ~/ ">"
).map((shape: Seq[IntData], scalableDims: Seq[IntData], typ: Attribute) =>
  VectorType(
    shape = ArrayAttribute[IntData](shape),
    elementType = typ,
    scalableDims = ArrayAttribute[IntData](scalableDims),
  )
)

/*≡==--==≡≡≡≡≡≡≡==--=≡≡*\
||   SYMBOL REF ATTR   ||
\*≡==---==≡≡≡≡≡==---==≡*/

def symbolRefAttrP[$: P](using Parser): P[SymbolRefAttr] = P(
  symbolRefIdP ~~ ("::" ~~ symbolRefIdP).repX
).map((x: String, y: Seq[String]) =>
  SymbolRefAttr(
    StringData(x),
    y.map(StringData.apply),
  )
)

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||   DenseIntOrFPElementsAttr   ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

// TO-DO : it can also parse a vector type or a memref type
// TO-DO : Figure out why it is throwing an error when you get rid of asInstanceOf...

def denseIntOrFPElementsAttrP[$: P](using
    Parser
): P[DenseIntOrFPElementsAttr] =
  P(
    "dense" ~ "<" ~ tensorLiteralP ~ ">" ~ ":" ~
      (tensorTypeP | memrefTypeP | vectorTypeP)
  ).map((x, y) => DenseIntOrFPElementsAttr(y, x))

def tensorLiteralP[$: P](using Parser): P[TensorLiteralArray] =
  P(singleTensorLiteralP | emptyTensorLiteralP | multipleTensorLiteralP)

def singleTensorLiteralP[$: P](using Parser): P[TensorLiteralArray] =
  P(floatDataP | intDataP)
    .map(_ match
      case (x: IntData) =>
        ArrayAttribute[IntegerAttr](Seq(IntegerAttr(x, I32)))
      case (y: FloatData) =>
        ArrayAttribute[FloatAttr](Seq(FloatAttr(y, Float32Type()))))

def multipleTensorLiteralP[$: P](using Parser): P[TensorLiteralArray] =
  P(multipleFloatTensorLiteralP | multipleIntTensorLiteralP)

def multipleIntTensorLiteralP[$: P](using
    Parser
): P[ArrayAttribute[IntegerAttr]] =
  P("[" ~ intDataP.rep(1, sep = ",") ~ "]").map((x: Seq[IntData]) =>
    ArrayAttribute[IntegerAttr](
      for (x1 <- x) yield IntegerAttr(x1, I32)
    )
  )

def multipleFloatTensorLiteralP[$: P](using
    Parser
): P[ArrayAttribute[FloatAttr]] =
  P("[" ~ floatDataP.rep(1, sep = ",") ~ "]").map((y: Seq[FloatData]) =>
    ArrayAttribute[FloatAttr](
      for (y1 <- y) yield FloatAttr(y1, Float32Type())
    )
  )

def emptyTensorLiteralP[$: P](using Parser): P[TensorLiteralArray] =
  P("[" ~ "]").map(_ => ArrayAttribute[IntegerAttr](Seq()))

/*≡==--==≡≡≡≡≡≡≡==--=≡≡*\
||   AFFINE MAP ATTR   ||
\*≡==---==≡≡≡≡≡==---==≡*/

def affineMapAttrP[$: P](using Parser): P[AffineMapAttr] =
  P("affine_map<" ~/ affineMapP ~ ">").map(AffineMapAttr(_))

/*≡==--==≡≡≡≡≡≡≡==--=≡≡*\
||   AFFINE SET ATTR   ||
\*≡==---==≡≡≡≡≡==---==≡*/

def affineSetAttrP[$: P](using Parser): P[AffineSetAttr] =
  P("affine_set<" ~/ affineSetP ~ ">").map(AffineSetAttr(_))

/*≡==--==≡≡≡≡≡==--=≡≡*\
||   FUNCTION TYPE   ||
\*≡==---==≡≡≡==---==≡*/

def functionTypeP[$: P](using Parser): P[FunctionType] = P(
  (parenTypeListP ~ "->" ~/ (parenTypeListP | typeP.map(Seq(_))))
    .map(FunctionType.apply)
)

private def builtinTypeP[$: P](using Parser): P[Attribute] =
  floatTypeP | integerTypeP | indexTypeP | complexTypeP | functionTypeP |
    tensorTypeP | memrefTypeP | vectorTypeP

private def builtinAttrP[$: P](using Parser): P[Attribute] =
  arrayAttributeP | denseArrayAttributeP | stringAttributeP | symbolRefAttrP |
    floatAttrP | integerAttrP | denseIntOrFPElementsAttrP | affineMapAttrP |
    affineSetAttrP
