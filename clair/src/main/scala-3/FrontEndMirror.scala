package scair.clair.mirrored

import scala.deriving._
import scala.compiletime._
import scair.scairdl.irdef._
import scair.scairdl.constraints._
import Variadicity._
import scala.reflect._
import scair.ir._
import scair.dialects.builtin._
import scair.scairdl.irdef.{AttrEscapeHatch, OpEscapeHatch}

/*≡≡=---=≡≡≡≡≡≡≡≡≡=---=≡≡*\
||   DIFFERENT CLASSES   ||
\*≡==----=≡≡≡≡≡≡≡=----==≡*/

abstract class DialectFE
trait OperationFE extends DialectFE
trait AttributeFE extends DialectFE
trait TypeAttributeFE extends AttributeFE

sealed abstract class AnyAttribute extends TypeAttribute

abstract class Input[T <: Attribute | AttributeFE]
case class Operand[T <: Attribute | AttributeFE]() extends Input[T]
case class Result[T <: Attribute | AttributeFE]() extends Input[T]
case class Region() extends Input[Nothing]
case class Successor() extends Input[Nothing]
case class Property[T <: Attribute | AttributeFE]() extends Input[T]
case class Attr[T <: Attribute | AttributeFE]() extends Input[T]

abstract class Variadic[T]

case class ConstraintRef(attr_name: String) extends IRDLConstraint {

  override def verify(
      that_attr: Attribute,
      constraint_ctx: ConstraintContext
  ): Unit = ()

  override def toString = s"BaseAttr[${attr_name}]()"
}

/*≡≡=---=≡≡≡≡≡≡=---=≡≡*\
||   ERROR HANDLING   ||
\*≡==----=≡≡≡≡=----==≡*/

object ErrorMessages {
  val invalidOpInput =
    "You can only pass in Operand, Result, Region, Successor, Property or Attr to the Operation definition"
  val invalidVariadicOpInput =
    "Variadicity is supported only for Operand, Result, Region or Successor."
}

/*≡≡=---=≡≡≡≡≡≡=---=≡≡*\
||    MIRROR LOGIC    ||
\*≡==----=≡≡≡≡=----==≡*/

/** Given a type, produces a corresponding constraint.
  *
  * @return
  *   IRDLConstraint
  */
inline def constraintFromAttr[T <: Attribute: ClassTag]: IRDLConstraint = {

  inline erasedValue[T] match
    case _: AnyAttribute => AnyAttr
    case _: Attribute    => BaseAttr[T]()
    case _ =>
      throw new Exception(
        "The Clair front-end currently supports only AnyAttr and BaseAttr Constraints"
      )
}

/** Given a AttributeFE, produces a corresponding constraint reference.
  *
  * @return
  *   IRDLConstraint
  */
inline def constraintFromAttrFE[T <: AttributeFE](using
    m: Mirror.ProductOf[T]
): IRDLConstraint = {
  val attr_name = constValue[m.MirroredLabel]
  ConstraintRef(attr_name)
}

/** Instantiates a ClassTag for the given type T. This is necessary as some
  * constraints deal with ClassTags.
  *
  * @return
  *   An IRDLConstraint given a type T.
  */
inline def getConstraint[T <: Attribute | AttributeFE]: IRDLConstraint = {
  inline erasedValue[T] match

    case _: Attribute =>
      type RefT = T & Attribute // absorbtion => A & (A | B) == A
      constraintFromAttr[RefT](using summonInline[ClassTag[RefT]])

    case _: AttributeFE =>
      type RefT = T & AttributeFE // absorbtion => A & (A | B) == A
      constraintFromAttrFE[RefT](using summonInline[Mirror.ProductOf[RefT]])
}

inline def inputVariadicity[Elem] = inline erasedValue[Elem] match
  case _: Variadic[t] => Variadic
  case _              => Single

type unwrappedInput[Elem] = Elem match
  case Variadic[t] => t
  case _           => Elem

/** Produces an OpInput to OperationDef given a definition of a Type.
  *
  * @return
  *   Input to OperationDef, either: OperandDef, ResultDef, RegionDef,
  *   SuccessorDef, OpPropertyDef, OpAttributeDef
  */
inline def getDefInput[Elem]: String => OpInput = {
  inline erasedValue[unwrappedInput[Elem]] match
    case _: Operand[t] =>
      (name: String) =>
        OperandDef(
          id = name,
          getConstraint[t],
          inputVariadicity[Elem]
        )
    case _: Result[t] =>
      (name: String) =>
        ResultDef(
          id = name,
          getConstraint[t],
          inputVariadicity[Elem]
        )
    case _: Region =>
      (name: String) =>
        RegionDef(
          id = name,
          inputVariadicity[Elem]
        )
    case _: Successor =>
      (name: String) =>
        SuccessorDef(
          id = name,
          inputVariadicity[Elem]
        )
    case _: Property[t] =>
      (name: String) =>
        OpPropertyDef(
          id = name,
          getConstraint[t]
        )
    case _: Attr[t] =>
      (name: String) =>
        OpAttributeDef(
          id = name,
          getConstraint[t]
        )
    case _ =>
      throw new Exception("aiowdjaowidjawoidjowij")
}

/** Loops through a Tuple of Input definitions and produces a List of inputs to
  * OperationDef.
  *
  * @return
  *   Lambda that produces an input to OperationDef, given a string
  */
inline def summonInput[Elems <: Tuple]: List[String => OpInput] = {

  inline erasedValue[Elems] match
    case _: (elem *: elems) => getDefInput[elem] :: summonInput[elems]
    case _: EmptyTuple      => Nil
}

/** Translates a Tuple of string types into a list of strings.
  *
  * @return
  *   Tuple of String types
  */
inline def stringifyLabels[Elems <: Tuple]: List[String] = {

  inline erasedValue[Elems] match
    case _: (elem *: elems) =>
      constValue[elem].asInstanceOf[String] :: stringifyLabels[elems]
    case _: EmptyTuple => Nil
}

/** Generates a OperationDef given param m.
  *
  * @param m
  *   \- Mirror Product of an dialect enum case.
  * @return
  *   Lambda that produces an Operadtion Def given a dialect name.
  */
inline def getDef[T](dialect_name: String)(using
    m: Mirror.ProductOf[T]
): OperationDef | AttributeDef = {

  val defname = constValue[m.MirroredLabel]
  val paramLabels = stringifyLabels[m.MirroredElemLabels]

  inline erasedValue[T] match
    case _: OperationFE =>
      val inputs = summonInput[m.MirroredElemTypes]

      val operands: ListType[OperandDef] = ListType()
      val results: ListType[ResultDef] = ListType()
      val regions: ListType[RegionDef] = ListType()
      val successors: ListType[SuccessorDef] = ListType()
      val opProperty: ListType[OpPropertyDef] = ListType()
      val opAttribute: ListType[OpAttributeDef] = ListType()

      for ((name, input) <- paramLabels zip inputs) yield input(name) match {
        case a: OperandDef     => operands += a
        case b: ResultDef      => results += b
        case c: RegionDef      => regions += c
        case d: SuccessorDef   => successors += d
        case e: OpPropertyDef  => opProperty += e
        case f: OpAttributeDef => opAttribute += f
      }

      OperationDef(
        dialect_name + "." + defname.toLowerCase,
        defname,
        operands.toSeq,
        results.toSeq,
        regions.toSeq,
        successors.toSeq,
        opProperty.toSeq,
        opAttribute.toSeq
      )

    case _: AttributeFE =>
      val inputs = summonInput[m.MirroredElemTypes]
      val operands: ListType[OperandDef] = ListType()

      val typee = inline erasedValue[T] match
        case _: TypeAttributeFE => 1
        case _                  => 0

      for ((name, input) <- paramLabels zip inputs) yield input(name) match {
        case a: OperandDef => operands += a
        case _ =>
          throw new Exception("Attributes only accept Operands.")
      }

      AttributeDef(
        dialect_name + "." + defname.toLowerCase,
        defname,
        operands.toSeq,
        typee
      )

    case _ =>
      throw new Exception("OperationFE or AttributeFE definition expected.")

}

/** Instantiates a Mirror Product for the given element.
  *
  * @return
  *   Lambda that produces an OperadtionDef given a string of dialect name.
  */
inline def summonDef[Elem](
    dialect_name: String
): OperationDef | AttributeDef = {
  getDef[Elem](dialect_name)(using summonInline[Mirror.ProductOf[Elem]])
}

/** Generates a list of OperationDef given enum cases.
  *
  * @param dialect_name
  */
inline def summonDialectDefs[Prods <: Tuple](
    dialect_name: String
): Seq[OperationDef | AttributeDef] = {

  inline erasedValue[Prods] match
    case _: (prod *: prods) =>
      summonDef[prod](dialect_name) +: summonDialectDefs[prods](dialect_name)

    case _: EmptyTuple => Seq.empty
}

/** Generates the DialectDef object from the enum definition.
  *
  * @param m
  *   \- Sum Mirror of a given dialect
  */
inline def summonDialect[Prods <: Tuple](
    dialect_name: String,
    opHatches: Seq[OpEscapeHatch[_]] = Seq(),
    attrHatches: Seq[AttrEscapeHatch[_]] = Seq()
): DialectDef = {
  val defs =
    summonDialectDefs[Prods](dialect_name.toLowerCase)

  val opDefs = defs.collect { case op: OperationDef => op }
  val attrDefs = defs.collect { case attr: AttributeDef => attr }

  DialectDef(
    dialect_name,
    opDefs,
    attrDefs,
    opHatches,
    attrHatches
  )
}

/*≡≡=---=≡≡≡=---=≡≡*\
||     TESTING     ||
\*≡==----=≡=----==≡*/

// object FrontEnd {

//   // inline def regionindent[T]: String = {
//   //   constValue[T].asInstanceOf[Int].toString
//   // }
//   import scair.ir.{DataAttribute, AttributeObject}

//   object SampleData extends AttributeObject {
//     override def name: String = "sample"
//   }

//   case class SampleData(val d: String)
//       extends DataAttribute[String]("sample", d)

//   enum CMath extends DialectFE:

//     case Complex(
//         e1: Operand[IntegerAttr]
//     ) extends CMath with AttributeFE

//     case ComplexType(
//         e1: Operand[IntegerAttr]
//     ) extends CMath with TypeAttributeFE

//     case Norm(
//         e1: Variadic[Operand[IntegerAttr]],
//         e2: Result[AnyAttribute],
//         e3: Region
//     ) extends CMath with OperationFE

//     case Mul(
//         e1: Variadic[Operand[Complex]],
//         e2: Result[AnyAttribute]
//     ) extends CMath with OperationFE

//   object CMath {
//     val opHatches = Seq()
//     val attrHatches = Seq(new AttrEscapeHatch[SampleData])
//     val generator = summonDialect[CMath](opHatches, attrHatches)
//   }

//   def main(args: Array[String]): Unit = {
//     println(CMath.generator.print(0))
//     println(new AttrEscapeHatch[SampleData].importt)
//   }

// }
