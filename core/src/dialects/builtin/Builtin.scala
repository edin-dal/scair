package scair.dialects.builtin

import fastparse.*
import scair.Printer
import scair.clair.*
import scair.ir.*
import scair.parse.*
import scair.utils.*

// ██████╗░ ██╗░░░██╗ ██╗ ██╗░░░░░ ████████╗ ██╗ ███╗░░██╗
// ██╔══██╗ ██║░░░██║ ██║ ██║░░░░░ ╚══██╔══╝ ██║ ████╗░██║
// ██████╦╝ ██║░░░██║ ██║ ██║░░░░░ ░░░██║░░░ ██║ ██╔██╗██║
// ██╔══██╗ ██║░░░██║ ██║ ██║░░░░░ ░░░██║░░░ ██║ ██║╚████║
// ██████╦╝ ╚██████╔╝ ██║ ███████╗ ░░░██║░░░ ██║ ██║░╚███║
// ╚═════╝░ ░╚═════╝░ ╚═╝ ╚══════╝ ░░░╚═╝░░░ ╚═╝ ╚═╝░░╚══╝

// ██████╗░ ██╗ ░█████╗░ ██╗░░░░░ ███████╗ ░█████╗░ ████████╗
// ██╔══██╗ ██║ ██╔══██╗ ██║░░░░░ ██╔════╝ ██╔══██╗ ╚══██╔══╝
// ██║░░██║ ██║ ███████║ ██║░░░░░ █████╗░░ ██║░░╚═╝ ░░░██║░░░
// ██║░░██║ ██║ ██╔══██║ ██║░░░░░ ██╔══╝░░ ██║░░██╗ ░░░██║░░░
// ██████╔╝ ██║ ██║░░██║ ███████╗ ███████╗ ╚█████╔╝ ░░░██║░░░
// ╚═════╝░ ╚═╝ ╚═╝░░╚═╝ ╚══════╝ ╚══════╝ ░╚════╝░ ░░░╚═╝░░░

val I1 = IntegerType(IntData(1), Signless)
val I32 = IntegerType(IntData(32), Signless)
val I64 = IntegerType(IntData(64), Signless)

/*≡==--==≡≡≡≡==--=≡≡*\
||    SIGNEDNESS   ||
\*≡==---==≡≡==---==≡*/

sealed abstract class Signedness(override val name: String, val dat: String)
    extends DataAttribute[String](name, dat):
  override def customPrint(p: Printer) = p.print(dat)

case object Signed extends Signedness("signed", "si")
case object Unsigned extends Signedness("unsigned", "ui")
case object Signless extends Signedness("signless", "i")

/*≡==--==≡≡≡≡==--=≡≡*\
||    FLOAT TYPE    ||
\*≡==---==≡≡==---==≡*/

sealed abstract class FloatType extends TypeAttribute

final case class Float16Type() extends FloatType with DerivedAttribute["f16"]
    derives AttrDefs:
  override def customPrint(p: Printer) = p.print(name)

final case class Float32Type() extends FloatType with DerivedAttribute["f32"]
    derives AttrDefs:
  override def customPrint(p: Printer) = p.print(name)

final case class Float64Type() extends FloatType with DerivedAttribute["f64"]
    derives AttrDefs:
  override def customPrint(p: Printer) = p.print(name)

final case class Float80Type() extends FloatType with DerivedAttribute["f80"]
    derives AttrDefs:
  override def customPrint(p: Printer) = p.print(name)

final case class Float128Type() extends FloatType with DerivedAttribute["f128"]
    derives AttrDefs:
  override def customPrint(p: Printer) = p.print(name)

/*≡==--==≡≡≡≡==--=≡≡*\
||     INT DATA     ||
\*≡==---==≡≡==---==≡*/

final case class IntData(value: BigInt)
    extends DataAttribute[BigInt]("builtin.int_attr", value)
    derives TransparentData:
  override def customPrint(p: Printer) = p.print(value.toString)

/*≡==--==≡≡≡≡==--=≡≡*\
||  INTEGER TYPE    ||
\*≡==---==≡≡==---==≡*/

final case class IntegerType(width: IntData, sign: Signedness)
    extends TypeAttribute
    with DerivedAttribute["builtin.int_type"] derives AttrDefs:

  override def customPrint(p: Printer) =
    p.print(sign)
    p.print(width)

/*≡==--==≡≡≡≡==--=≡≡*\
|| INTEGER ATTRIBUTE ||
\*≡==---==≡≡==---==≡*/

case class IntegerAttr(
    value: IntData,
    typ: IntegerType | IndexType = I64,
) extends DerivedAttribute["builtin.integer_attr"] derives AttrDefs:

  infix def +(that: IntegerAttr): IntegerAttr =
    if this.typ != that.typ then
      throw new Exception(
        s"Cannot add IntegerAttrs of different types: ${this.typ} and ${that.typ}"
      )
    // TODO: Make it correct
    IntegerAttr(IntData(this.value.value + that.value.value), this.typ)

  infix def -(that: IntegerAttr): IntegerAttr =
    if this.typ != that.typ then
      throw new Exception(
        s"Cannot add IntegerAttrs of different types: ${this.typ} and ${that.typ}"
      )
    // TODO: Make it correct
    IntegerAttr(IntData(this.value.value - that.value.value), this.typ)

  infix def *(that: IntegerAttr): IntegerAttr =
    if this.typ != that.typ then
      throw new Exception(
        s"Cannot multiply IntegerAttrs of different types: ${this
            .typ} and ${that.typ}"
      )
    // TODO: Make it correct
    IntegerAttr(IntData(this.value.value * that.value.value), this.typ)

  override def customPrint(p: Printer) = (value, typ) match
    case (IntData(1), IntegerType(IntData(1), Signless)) => p.print("true")
    case (IntData(0), IntegerType(IntData(1), Signless)) => p.print("false")
    case (_, IntegerType(IntData(64), Signless))         => p.print(value)
    case (_, _) => p.print(value, " : ", typ)

/*≡==--==≡≡≡≡==--=≡≡*\
||    FLOAT DATA    ||
\*≡==---==≡≡==---==≡*/

final case class FloatData(value: Double)
    extends DataAttribute[Double]("builtin.float_data", value)
    derives TransparentData:
  override def customPrint(p: Printer) = p.print(value.toString)

/*≡==--==≡≡≡≡==--=≡≡*\
||  FLOAT ATTRIBUTE ||
\*≡==---==≡≡==---==≡*/

final case class FloatAttr(value: FloatData, typ: FloatType)
    extends DerivedAttribute["builtin.float_attr"] derives AttrDefs:

  override def customPrint(p: Printer) =
    p.print(value, " : ", typ)

/*≡==--==≡≡≡≡==--=≡≡*\
||   INDEX TYPE     ||
\*≡==---==≡≡==---==≡*/

final case class IndexType()
    extends DerivedAttribute["builtin.index"]
    with TypeAttribute derives AttrDefs:
  override def customPrint(p: Printer) = p.print("index")

final case class ComplexType(
    tpe: IntegerType | IndexType | FloatType
) extends DerivedAttribute["builtin.complex"] derives AttrDefs:

  override def customPrint(p: Printer) = p.print("complex<", tpe, ">")

/*≡==--==≡≡≡≡==--=≡≡*\
|| ARRAY ATTRIBUTE  ||
\*≡==---==≡≡==---==≡*/

final case class ArrayAttribute[D <: Attribute](attrValues: Seq[D])
    extends DataAttribute[Seq[D]]("builtin.array_attr", attrValues):

  override def customPrint(p: Printer) =
    p.printList(attrValues, "[", ", ", "]")

/*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
|| DICTIONARY ATTRIBUTE  ||
\*≡==---==≡≡≡≡≡≡≡==---==≡*/

final case class DictionaryAttr(entries: Map[String, Attribute])
    extends DataAttribute[Map[String, Attribute]](
      "builtin.dict_attr",
      entries,
    ):

  override def customPrint(p: Printer) =
    p.printAttrDict(entries)

/*≡==--==≡≡≡≡==--=≡≡*\
|| STRING ATTRIBUTE ||
\*≡==---==≡≡==---==≡*/
final case class StringData(stringLiteral: String)
    extends DataAttribute("builtin.string", stringLiteral)
    derives TransparentData:

  override def customPrint(p: Printer) =
    //       ("\\" ~~ (
    //   "n"  ~~ Pass("\n")
    // | "t"  ~~ Pass("\t")
    // | "\\" ~~ Pass("\\")
    // | "\"" ~~ Pass("\"")
    p.print(
      "\"",
      stringLiteral.flatMap((c: Char) =>
        c match
          case '\n' => "\\n"
          case '\t' => "\\t"
          case '\\' => "\\\\"
          case '"'  => "\\\""
          case _    => c.toString()
      ),
      "\"",
    )

/*≡==--==≡≡≡≡==--=≡≡*\
||   SHAPED TYPE    ||
\*≡==---==≡≡==---==≡*/

trait ShapedType extends TypeAttribute:
  def getNumDims: Int
  def getShape: Seq[Long]
  def elementCount: Long = getShape.product

/*≡==--==≡≡≡≡==--=≡≡*\
||   TENSOR TYPE    ||
\*≡==---==≡≡==---==≡*/
trait ContainerType extends ParametrizedAttribute, TypeAttribute:
  def elementType: Attribute

sealed trait TensorType extends ContainerType

case class RankedTensorType(
    elementType: Attribute,
    shape: ArrayAttribute[IntData],
    encoding: Option[Attribute] = None,
) extends TensorType,
      ShapedType:

  override def name: String = "builtin.ranked_tensor"

  override def parameters: Seq[Attribute | Seq[Attribute]] =
    shape +: elementType +: encoding.toSeq

  override def getNumDims = shape.attrValues.length
  override def getShape = shape.attrValues.map(_.data.toLong)

  override def customPrint(p: Printer) =
    p.print("tensor<")
    shape.attrValues.foreach(s =>
      s match
        case IntData(-1) => p.print("?")
        case d           => p.print(d)
      p.print("x")
    )
    p.print(elementType)
    if encoding.isDefined then p.print(", ", encoding)
    p.print(">")

final case class UnrankedTensorType(elementType: Attribute)
    extends DerivedAttribute["builtin.unranked_tensor"]
    with TensorType derives AttrDefs:

  override def customPrint(p: Printer) =
    p.print("tensor<*x", elementType, ">")

/*≡==--==≡≡≡≡==--=≡≡*\
||   MEMREF TYPE    ||
\*≡==---==≡≡==---==≡*/

sealed trait MemrefType
    extends ParametrizedAttribute,
      TypeAttribute,
      ContainerType

final case class RankedMemrefType(
    elementType: Attribute,
    shape: ArrayAttribute[IntData],
    encoding: Option[Attribute] = None,
) extends MemrefType,
      ShapedType:

  override def name: String = "builtin.ranked_memref"

  override def parameters: Seq[Attribute | Seq[Attribute]] =
    shape +: elementType +: encoding.toSeq

  override def getNumDims = shape.attrValues.length
  override def getShape = shape.attrValues.map(_.data.toLong)

  override def customPrint(p: Printer) =
    p.print("memref<")
    shape.attrValues.foreach(s =>
      s match
        case IntData(-1) => p.print("?")
        case d           => p.print(d)
      p.print("x")
    )

    p.print(elementType, ">")

final case class UnrankedMemrefType(elementType: Attribute)
    extends DerivedAttribute["builtin.unranked_memref"]
    with MemrefType derives AttrDefs:

  override def customPrint(p: Printer) =
    p.print("tensor<*x", elementType, ">")

/*≡==--==≡≡≡≡==--=≡≡*\
||   VECTOR TYPE    ||
\*≡==---==≡≡==---==≡*/

final case class VectorType(
    elementType: Attribute,
    shape: ArrayAttribute[IntData],
    scalableDims: ArrayAttribute[IntData],
) extends DerivedAttribute["builtin.vector_type"]
    with ShapedType
    with ContainerType derives AttrDefs:

  override def getNumDims = shape.attrValues.length
  override def getShape = shape.attrValues.map(_.data.toLong)

  override def customPrint(p: Printer): Unit =

    p.print("vector<")
    p.printListF(
      shape zip scalableDims,
      (size, scalable) =>
        if scalable.data != 0 then p.print("[", size, "]")
        else p.print(size),
      sep = "x",
    )
    p.print("x", elementType, ">")

/*≡==--==≡≡≡≡==--=≡≡*\
|| SYMBOL REF ATTR  ||
\*≡==---==≡≡==---==≡*/

final case class SymbolRefAttr(
    rootRef: StringData,
    nestedRefs: Seq[StringData] = Seq(),
) extends ParametrizedAttribute:

  override def name: String = "builtin.symbol_ref"

  override def parameters: Seq[Attribute | Seq[Attribute]] =
    Seq(rootRef, nestedRefs)

  override def customPrint(p: Printer) =
    p.printListF(
      rootRef +: nestedRefs,
      ref => p.print("@", ref.data),
      sep = "::",
    )

/*≡==--==≡≡≡≡==--=≡≡*\
|| DenseArrayAttr   ||
\*≡==---==≡≡==---==≡*/

final case class DenseArrayAttr(
    typ: IntegerType | FloatType,
    data: Seq[IntegerAttr] | Seq[FloatAttr],
) extends ParametrizedAttribute
    with Seq[Attribute]:

  override def name: String = "builtin.dense_array"
  override def parameters: Seq[Attribute | Seq[Attribute]] = Seq(typ, data)

  override def customVerify(): OK[Unit] =
    if !data.forall(_ match
        case IntegerAttr(_, eltyp) => eltyp == typ
        case FloatAttr(_, eltyp)   => eltyp == typ)
    then Err("Element types do not match the dense array type")
    else OK()

  override def customPrint(p: Printer) =
    p.print("array<", typ)
    if data.nonEmpty then p.print(": ")
    p.printListF(
      data,
      {
        case IntegerAttr(value, _) => p.print(value)
        case FloatAttr(value, _)   => p.print(value)
      },
    )
    p.print(">")

  // Seq methods
  def apply(idx: Int): Attribute = data.apply(idx)

  def length: Int = data.length

  def iterator: Iterator[Attribute] = data.iterator

/*≡==--==≡≡≡≡==--=≡≡*\
||  FunctionType    ||
\*≡==---==≡≡==---==≡*/

final case class FunctionType(
    inputs: Seq[Attribute] = Seq.empty,
    outputs: Seq[Attribute] = Seq.empty,
) extends ParametrizedAttribute
    with TypeAttribute:

  override def name: String = "builtin.function_type"

  override def parameters: Seq[Attribute | Seq[Attribute]] =
    Seq(inputs, outputs)

  override def customPrint(p: Printer) =
    p.print("(")
    p.printList(inputs)
    p.print(") -> ")
    outputs match
      case Seq(single) => p.print(single)
      case s           => p.printList(s, "(", ", ", ")")

/*≡==--==≡≡≡≡==--=≡≡*\
|| DenseIntOrFPAttr ||
\*≡==---==≡≡==---==≡*/

type TensorLiteralArray =
  ArrayAttribute[IntegerAttr] | ArrayAttribute[FloatAttr]

final case class DenseIntOrFPElementsAttr(
    typ: ContainerType,
    data: TensorLiteralArray,
) extends DerivedAttribute["builtin.dense"] derives AttrDefs:

  def elementType = typ.elementType

  override def customVerify(): OK[Unit] =
    val tpe = elementType match
      case it: IntegerType => OK(it)
      case ft: FloatType   => OK(ft)
      case _               =>
        Err(
          s"DenseIntOrFPElementsAttr element type must be IntegerType or FloatType, got: $elementType"
        )

    data.attrValues.foldLeft[OK[Any]](
      tpe
    )((acc, elt) =>
      acc.map(tpe =>
        elt match
          case IntegerAttr(_, etyp) =>
            if tpe == etyp then acc
            else
              Err(
                s"DenseIntOrFPElementsAttr data element type $etyp does not match expected type $tpe"
              )
          case FloatAttr(_, etyp) =>
            if tpe == etyp then acc
            else
              Err(
                s"DenseIntOrFPElementsAttr data element type $etyp does not match expected type $tpe"
              )
          case _ =>
            Err(
              s"DenseIntOrFPElementsAttr data element must be IntegerAttr or FloatAttr, got: $elt"
            )
      )
    ).map(_ => ())

  override def customPrint(p: Printer) =
    val values = data.attrValues(0) match
      case x: IntegerAttr =>
        for (a <- data.attrValues) yield a.asInstanceOf[IntegerAttr].value
      case y: FloatAttr =>
        for (a <- data.attrValues) yield a.asInstanceOf[FloatAttr].value
    p.print("dense<")
    values match
      case Seq(single) => p.print(single)
      case s           => p.printList(s, "[", ", ", "]")
    p.print("> : ")
    p.print(typ)

/*≡==--==≡≡≡≡==--=≡≡*\
||  AFFINE MAP ATTR ||
\*≡==---==≡≡==---==≡*/

final case class AffineMapAttr(affineMap: AffineMap)
    extends DataAttribute[AffineMap]("builtin.affine_map", affineMap)
    with AliasedAttribute("map") derives TransparentData:

  override def customPrint(p: Printer) =
    p.print("affine_map<", affineMap.toString, ">")

/*≡==--==≡≡≡≡==--=≡≡*\
||  AFFINE SET ATTR ||
\*≡==---==≡≡==---==≡*/
// note: in mlir terms this is called an IntegerSetAttr

final case class AffineSetAttr(affineSet: AffineSet)
    extends DataAttribute[AffineSet]("builtin.affine_set", affineSet)
    with AliasedAttribute("set") derives TransparentData:

  override def customPrint(p: Printer) =
    p.print("affine_set<", affineSet.toString, ">")

/*≡==--==≡≡≡≡==--=≡≡*\
||   OPERATIONS    ||
\*≡==---==≡≡==---==≡*/

// ==------== //
//  ModuleOp  //
// ==------== //

given OperationCustomParser[ModuleOp]:

  // ==--- Custom Parsing ---== //
  def parse[$: P](
      resNames: Seq[String]
  )(using Parser): P[ModuleOp] =
    P(
      regionP()
    ).map(ModuleOp.apply)

  // ==----------------------== //

case class ModuleOp(
    body: Region
) extends DerivedOperation["builtin.module"]
    with SymbolTable derives OpDefs:

  override def customPrint(
      p: Printer
  ) =
    p.print("builtin.module ", regions(0))

case class UnrealizedConversionCastOp(
    inputs: Seq[Value[Attribute]] = Seq(),
    outputs: Seq[Result[Attribute]] = Seq(),
) extends DerivedOperation["builtin.unrealized_conversion_cast"] derives OpDefs

val BuiltinDialect =
  summonDialect[EmptyTuple, (ModuleOp, UnrealizedConversionCastOp)]
