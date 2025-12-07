package scair.parse

import fastparse.*
import scair.*
import scair.dialects.builtin.*
import scair.dialects.builtin.VectorType
import scair.ir.*

import java.lang.Float.intBitsToFloat
import scala.annotation.switch
import scala.annotation.tailrec
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
// IE THE PARSER FOR BUILTIN DIALECT ATTRIBUTES

/** Whitespace syntax that supports // line-comments, *without* /* */ comments,
  * as is the case in the MLIR Language Spec.
  *
  * It's litteraly fastparse's JavaWhitespace with the /* */ states just erased
  * :)
  */
given whitespace: Whitespace = new Whitespace:
  def apply(ctx: P[?]) =
    val input = ctx.input
    val startIndex = ctx.index
    @tailrec def rec(current: Int, state: Int): ParsingRun[Unit] =
      if !input.isReachable(current) then
        if state == 0 || state == 1 then ctx.freshSuccessUnit(current)
        else ctx.freshSuccessUnit(current - 1)
      else
        val currentChar = input(current)
        (state: @switch) match
          case 0 =>
            (currentChar: @switch) match
              case ' ' | '\t' | '\n' | '\r' => rec(current + 1, state)
              case '/'                      => rec(current + 1, state = 2)
              case _                        => ctx.freshSuccessUnit(current)
          case 1 =>
            rec(current + 1, state = if currentChar == '\n' then 0 else state)
          case 2 =>
            (currentChar: @switch) match
              case '/' => rec(current + 1, state = 1)
              case _   => ctx.freshSuccessUnit(current - 1)
    rec(current = ctx.index, state = 0)

class AttrParser(
    private[parse] val context: MLContext,
    private[parse] val attributeAliases: mutable.Map[String, Attribute] =
      mutable.Map.empty,
    private[parse] val typeAliases: mutable.Map[String, Attribute] =
      mutable.Map.empty
)

def DialectAttribute[$: P](using p: AttrParser): P[Attribute] = P(
  "#" ~~ PrettyDialectReferenceName.flatMapTry {
    (dialect: String, attrName: String) =>
      p.context.getAttrCompanion(s"${dialect}.${attrName}") match
        case Some(attr) =>
          attr.parse
        case None =>
          Fail(
            s"Attribute $dialect.$attrName is not defined in any supported Dialect."
          )
  }
)

def DialectType[$: P](using p: AttrParser): P[Attribute] = P(
  "!" ~~ PrettyDialectReferenceName.flatMapTry {
    (dialect: String, attrName: String) =>
      p.context.getAttrCompanion(s"${dialect}.${attrName}") match
        case Some(attr) =>
          attr.parse
        case None =>
          Fail(
            s"Type $dialect.$attrName is not defined in any supported Dialect."
          )
  }
)

// [x] - attribute-entry ::= (bare-id | string-literal) `=` attribute-value
// [x] - attribute-value ::= attribute-alias | dialect-attribute | builtin-attribute

def AttributeEntry[$: P](using AttrParser) = P(
  (BareId | StringLiteral) ~ "=" ~/ (AttributeP | TypeP)
)

def AttributeP[$: P](using AttrParser) = P(
  TypeP | BuiltinAttr | DialectAttribute | AttributeAlias // AttrBuiltIn | DialectAttribute // | AttributeAlias //
)

def AttributeAlias[$: P](using p: AttrParser) = P(
  "#" ~~ AliasName.flatMap((name: String) =>
    p.attributeAliases.get(name) match
      case Some(attr) => Pass(attr)
      case None       =>
        Fail(s"Attribute alias ${name} not defined.")
  )
)

def TypeAlias[$: P](using p: AttrParser) = P(
  "!" ~~ AliasName.flatMap((name: String) =>
    p.typeAliases.get(name) match
      case Some(attr) => Pass(attr)
      case None       => Fail(s"Type alias ${name} not defined.")
  )
)

def AttributeList[$: P](using AttrParser) = AttributeP.rep(sep = ",")

def TypeP[$: P](using AttrParser) = P(
  (BuiltinType | DialectType | TypeAlias)
)

def TypeList[$: P](using AttrParser) = TypeP.rep(sep = ",")

/*≡==--==≡≡≡≡==--=≡≡*\
||    FLOAT TYPE    ||
\*≡==---==≡≡==---==≡*/

def Float16TypeP[$: P](using AttrParser): P[Float16Type] =
  P("f16").map(_ => Float16Type())

def Float32TypeP[$: P](using AttrParser): P[Float32Type] =
  P("f32").map(_ => Float32Type())

def Float64TypeP[$: P](using AttrParser): P[Float64Type] =
  P("f64").map(_ => Float64Type())

def Float80TypeP[$: P](using AttrParser): P[Float80Type] =
  P("f80").map(_ => Float80Type())

def Float128TypeP[$: P](using AttrParser): P[Float128Type] =
  P("f128").map(_ => Float128Type())

def FloatTypeP[$: P](using AttrParser): P[FloatType] = P(
  Float16TypeP | Float32TypeP | Float64TypeP | Float80TypeP | Float128TypeP
)

/*≡==--==≡≡≡≡==--=≡≡*\
||     INT DATA     ||
\*≡==---==≡≡==---==≡*/

def IntDataP[$: P](using AttrParser): P[IntData] =
  P(IntegerLiteral).map(IntData(_))

/*≡==--==≡≡≡≡==--=≡≡*\
||   INTEGER TYPE   ||
\*≡==---==≡≡==---==≡*/

// signed-integer-type    ::=  `si` [1-9][0-9]*
// unsigned-integer-type  ::=  `ui` [1-9][0-9]*
// signless-integer-type  ::=  `i`  [1-9][0-9]*
// integer-type           ::=  signed-integer-type | unsigned-integer-type | signless-integer-type

def IntegerTypeP[$: P](using AttrParser): P[IntegerType] = P(
  (("i".map(_ => Signless) | "si".map(_ => Signed) | "ui".map(_ =>
    Unsigned
  )) ~~ DecimalLiteral.map(IntData.apply)).map((sign, bits) =>
    IntegerType.apply(bits, sign)
  )
)

/*≡==--==≡≡≡≡==--=≡≡*\
||   INTEGER ATTR   ||
\*≡==---==≡≡==---==≡*/

def IntegerAttrP[$: P](using AttrParser): P[IntegerAttr] =
  P(
    (IntDataP ~ (":" ~ (IntegerTypeP | IndexTypeP)
      .asInstanceOf[P[IntegerType | IndexType]]).orElse(I64))
      .map(IntegerAttr.apply)
      | "true".map(_ => IntegerAttr(IntData(1), I1))
      | "false".map(_ => IntegerAttr(IntData(0), I1))
  )

/*≡==--==≡≡≡≡==--=≡≡*\
||    FLOAT DATA    ||
\*≡==---==≡≡==---==≡*/

def FloatDataP[$: P](using AttrParser): P[FloatData] =
  P(FloatLiteral).map(FloatData(_))

/*≡==--==≡≡≡≡==--=≡≡*\
||    FLOAT ATTR    ||
\*≡==---==≡≡==---==≡*/

def FloatAttrP[$: P](using AttrParser): P[FloatAttr] =
  P(
    (FloatDataP ~ (":" ~ FloatTypeP).orElse(Float64Type())).map((x, y) =>
      FloatAttr(
        x,
        y
      )
    ) | (HexadecimalLiteral ~ ":" ~ FloatTypeP).map((x, y) =>
      FloatAttr(FloatData(intBitsToFloat(x.intValue())), y)
    )
  )

/*≡==--==≡≡≡≡==--=≡≡*\
||    INDEX TYPE    ||
\*≡==---==≡≡==---==≡*/

inline def IndexTypeP[$: P](using AttrParser): P[IndexType] =
  "index".map(_ => IndexType())
/*≡==--==≡≡≡≡≡≡==--=≡≡*\
||    COMPLEX TYPE    ||
\*≡==---==≡≡≡≡==---==≡*/

inline def ComplexTypeP[$: P](using AttrParser): P[ComplexType] =
  ("complex<" ~ (IndexTypeP | IntegerTypeP | FloatTypeP) ~ ">").map(
    (tpe: TypeAttribute) =>
      ComplexType(tpe.asInstanceOf[IntegerType | IndexType | FloatType])
  )

/*≡==--==≡≡≡≡≡==--=≡≡*\
||  ARRAY ATTRIBUTE  ||
\*≡==---==≡≡≡==---==≡*/

// array-attribute  ::=  `[` (attribute-value (`,` attribute-value)*)? `]`

def ArrayAttributeP[$: P](using AttrParser): P[ArrayAttribute[Attribute]] = P(
  "[" ~ AttributeList
    .map((x: Seq[Attribute]) => ArrayAttribute(attrValues = x)) ~ "]"
)

/*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
|| DICTIONARY ATTRIBUTE  ||
\*≡==---==≡≡≡≡≡≡≡==---==≡*/

def DictionaryAttributeP[$: P](using AttrParser): P[DictionaryAttr] = P(
  "{" ~ AttributeEntry.rep(sep = ",") ~ "}"
).map((entries: Seq[(String, Attribute)]) =>
  DictionaryAttr(entries.map((name, attr) => (name, attr)).toMap)
)

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||  DENSE ARRAY ATTRIBUTE  ||
\*≡==---==≡≡≡≡≡≡≡≡≡==---==≡*/

// dense-array-attribute  ::=  `array` `<` (integer-type | float-type) (`:` tensor-literal)? `>`

def DenseArrayAttributeP[$: P](using AttrParser): P[DenseArrayAttr] = P(
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

def StringAttributeP[$: P](using AttrParser): P[StringData] = P(
  StringLiteral.map((x: String) => StringData(stringLiteral = x))
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

def TensorTypeP[$: P](using AttrParser): P[TensorType] = P(
  "tensor" ~ "<" ~/ (UnrankedTensorTypeP | RankedTensorTypeP) ~ ">"
)

def RankedTensorTypeP[$: P](using AttrParser): P[TensorType] = P(
  DimensionList ~ TypeP ~ ("," ~ Encoding).?
).map((x: (ArrayAttribute[IntData], Attribute, Option[Attribute])) =>
  RankedTensorType(
    shape = x._1,
    elementType = x._2,
    encoding = x._3
  )
)

def UnrankedTensorTypeP[$: P](using AttrParser): P[TensorType] =
  P("*" ~ "x" ~ TypeP).map((x: Attribute) =>
    UnrankedTensorType(elementType = x)
  )

def DimensionList[$: P](using AttrParser) =
  P((Dimension ~ "x").rep).map(x => ArrayAttribute(attrValues = x))

def Dimension[$: P](using AttrParser): P[IntData] =
  P("?".map(_ => -1: BigInt) | DecimalLiteral).map(x => IntData(x))

def Encoding[$: P](using AttrParser) = P(AttributeP)

/*≡==--==≡≡≡≡≡==--=≡≡*\
||    MEMREF TYPE    ||
\*≡==---==≡≡≡==---==≡*/

// memref-type           ::=   `memref` `<` (ranked-memref-type | unranked-memref-type) `>`
// ranked-memref-type    ::=   dimension-list type
// unranked-memref-type  ::=   `*` `x` type
// dimension-list        ::=   (dimension `x`)*
// dimension             ::=   `?` | decimal-literal

def MemrefTypeP[$: P](using AttrParser): P[MemrefType] = P(
  "memref" ~ "<" ~/ (UnrankedMemrefTypeP | RankedMemrefTypeP) ~ ">"
)

def RankedMemrefTypeP[$: P](using AttrParser): P[MemrefType] = P(
  DimensionList ~ TypeP
).map((x: (ArrayAttribute[IntData], Attribute)) =>
  RankedMemrefType(
    shape = x._1,
    elementType = x._2
  )
)

def UnrankedMemrefTypeP[$: P](using AttrParser): P[UnrankedMemrefType] =
  P("*" ~ "x" ~ TypeP).map((x: Attribute) =>
    UnrankedMemrefType(elementType = x)
  )

def VectorDimensionList[$: P](using AttrParser) =
  P(
    ((DecimalLiteral
      .map(x => (IntData(x), IntData(0))) | "[" ~ DecimalLiteral.map(x =>
      (IntData(x), IntData(1))
    ) ~ "]") ~ "x").rep(1).map(_.unzip)
  )

def VectorTypeP[$: P](using AttrParser): P[VectorType] = P(
  "vector<" ~/ VectorDimensionList ~/ TypeP ~/ ">"
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

def SymbolRefAttrP[$: P](using AttrParser): P[SymbolRefAttr] = P(
  SymbolRefId ~~ ("::" ~~ SymbolRefId).repX
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

def DenseIntOrFPElementsAttrP[$: P](using
    AttrParser
): P[DenseIntOrFPElementsAttr] =
  P(
    "dense" ~ "<" ~ TensorLiteral ~ ">" ~ ":" ~ (TensorTypeP | MemrefTypeP | VectorTypeP)
  ).map((x, y) => DenseIntOrFPElementsAttr(y, x))

def TensorLiteral[$: P](using AttrParser): P[TensorLiteralArray] =
  P(SingleTensorLiteral | EmptyTensorLiteral | MultipleTensorLiteral)

def SingleTensorLiteral[$: P](using AttrParser): P[TensorLiteralArray] =
  P(FloatDataP | IntDataP).map(_ match
    case (x: IntData) =>
      ArrayAttribute[IntegerAttr](Seq(IntegerAttr(x, I32)))
    case (y: FloatData) =>
      ArrayAttribute[FloatAttr](Seq(FloatAttr(y, Float32Type()))))

def MultipleTensorLiteral[$: P](using AttrParser): P[TensorLiteralArray] =
  P(MultipleFloatTensorLiteral | MultipleIntTensorLiteral)

def MultipleIntTensorLiteral[$: P](using
    AttrParser
): P[ArrayAttribute[IntegerAttr]] =
  P("[" ~ IntDataP.rep(1, sep = ",") ~ "]").map((x: Seq[IntData]) =>
    ArrayAttribute[IntegerAttr](
      for (x1 <- x) yield IntegerAttr(x1, I32)
    )
  )

def MultipleFloatTensorLiteral[$: P](using
    AttrParser
): P[ArrayAttribute[FloatAttr]] =
  P("[" ~ FloatDataP.rep(1, sep = ",") ~ "]").map((y: Seq[FloatData]) =>
    ArrayAttribute[FloatAttr](
      for (y1 <- y) yield FloatAttr(y1, Float32Type())
    )
  )

def EmptyTensorLiteral[$: P](using AttrParser): P[TensorLiteralArray] =
  P("[" ~ "]").map(_ => ArrayAttribute[IntegerAttr](Seq()))

/*≡==--==≡≡≡≡≡≡≡==--=≡≡*\
||   AFFINE MAP ATTR   ||
\*≡==---==≡≡≡≡≡==---==≡*/

def AffineMapAttrP[$: P](using AttrParser): P[AffineMapAttr] =
  P("affine_map<" ~ AffineMapP ~ ">").map(AffineMapAttr(_))

/*≡==--==≡≡≡≡≡≡≡==--=≡≡*\
||   AFFINE SET ATTR   ||
\*≡==---==≡≡≡≡≡==---==≡*/

def AffineSetAttrP[$: P](using AttrParser): P[AffineSetAttr] =
  P("affine_set<" ~ AffineSetP ~ ">").map(AffineSetAttr(_))

/*≡==--==≡≡≡≡≡==--=≡≡*\
||   FUNCTION TYPE   ||
\*≡==---==≡≡≡==---==≡*/

def ParenTypeList[$: P](using AttrParser) = P(
  "(" ~ TypeP.rep(sep = ",") ~ ")"
)

def FunctionTypeP[$: P](using AttrParser): P[FunctionType] = P(
  (ParenTypeList ~ "->" ~/ (ParenTypeList | TypeP.map(Seq(_))))
    .map(FunctionType.apply)
)

def BuiltinType[$: P](using AttrParser): P[Attribute] = P(
  FloatTypeP |
    IntegerTypeP |
    IndexTypeP |
    ComplexTypeP |
    FunctionTypeP |
    TensorTypeP |
    MemrefTypeP |
    VectorTypeP
)

def BuiltinAttr[$: P](using AttrParser): P[Attribute] = P(
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
