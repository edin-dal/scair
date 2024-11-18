package scair.clair.mirrored

import scala.deriving._
import scala.compiletime._
import scair.scairdl.irdef._
import scair.scairdl.constraints._
import Variadicity._
import scala.reflect._
import scair.ir._
import scair.dialects.builtin._

/*≡≡=---=≡≡≡≡≡≡≡≡≡=---=≡≡*\
||   DIFFERENT CLASSES   ||
\*≡==----=≡≡≡≡≡≡≡=----==≡*/

abstract class DialectOperation
abstract class DialectAttribute

sealed abstract class AnyAttribute extends TypeAttribute

abstract class Input[T <: Attribute]
case class Operand[T <: Attribute]() extends Input[T]
case class Result[T <: Attribute]() extends Input[T]
case class Region() extends Input[Nothing]
case class Successor() extends Input[Nothing]
case class Property[T <: Attribute]() extends Input[T]
case class Attr[T <: Attribute]() extends Input[T]

/*≡≡=---=≡≡≡≡≡≡=---=≡≡*\
||    MIRROR LOGIC    ||
\*≡==----=≡≡≡≡=----==≡*/

/** Given a type, produces a corresponding constraint.
  *
  * @return
  *   IRDLConstraint
  */
inline def indent[T <: Attribute: ClassTag]: IRDLConstraint = {

  inline erasedValue[T] match
    case _: AnyAttribute => AnyAttr
    case _: Attribute    => BaseAttr[T]()
    case _ =>
      throw new Exception(
        "The Clair front-end currently supports only AnyAttr Constraint"
      )
}

/** Instantiates a ClassTag for the given type T. This is necessary as some
  * constraints deal with ClassTags.
  *
  * @return
  *   An IRDLConstraint given a type T.
  */
inline def getConstraint[T <: Attribute]: IRDLConstraint = {
  indent[T](using summonInline[ClassTag[T]])
}

/** Produces an OpInput to OperationDef given a definition of a Type.
  *
  * @return
  *   Input to OperationDef, either: OperandDef, ResultDef, RegionDef,
  *   SuccessorDef, OpPropertyDef, OpAttributeDef
  */
inline def getOpInput[Elem]: String => OpInput = {

  inline erasedValue[Elem] match
    case _: Operand[t] =>
      (name: String) =>
        OperandDef(
          id = name,
          getConstraint[t],
          Single
        )
    case _: Result[t] =>
      (name: String) =>
        ResultDef(
          id = name,
          getConstraint[t],
          Single
        )
    case _: Region =>
      (name: String) =>
        RegionDef(
          id = name,
          Single
        )
    case _: Successor =>
      (name: String) =>
        SuccessorDef(
          id = name,
          Single
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
      throw new Exception(
        "You can only pass in Operand, Result, Region, Successor, Property or Attr to the Operation definition"
      )
}

/** Loops through a Tuple of Input definitions and produces a List of inputs to
  * OperationDef.
  *
  * @return
  *   Lambda that produces an input to OperationDef, given a string
  */
inline def summonInput[Elems <: Tuple]: List[String => OpInput] = {

  inline erasedValue[Elems] match
    case _: (elem *: elems) => getOpInput[elem] :: summonInput[elems]
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
inline def getOpDef[T](using
    m: Mirror.ProductOf[T]
): String => OperationDef = {

  val opName = constValue[m.MirroredLabel]
  val paramLabels = stringifyLabels[m.MirroredElemLabels]

  inline erasedValue[T] match
    case _: DialectOperation =>
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

      (dialect: String) =>
        OperationDef(
          dialect + "." + opName.toLowerCase,
          opName,
          operands.toSeq,
          results.toSeq,
          regions.toSeq,
          successors.toSeq,
          opProperty.toSeq,
          opAttribute.toSeq
        )

    case _ =>
      throw new Exception("Operation definition expected.")

}

/** Generates a AttributeDef given param m.
  *
  * @param m
  *   \- Mirror Product of an dialect enum case.
  * @return
  *   Lambda that produces an AttributeDef given a dialect name.
  */
inline def getAttrDef[T](using
    m: Mirror.ProductOf[T]
): String => AttributeDef = {

  val attrName = constValue[m.MirroredLabel]
  val paramLabels = stringifyLabels[m.MirroredElemLabels]

  inline erasedValue[T] match
    case _: DialectAttribute =>
      val inputs = summonInput[m.MirroredElemTypes]

      val operands: ListType[OperandDef] = ListType()

      for ((name, input) <- paramLabels zip inputs) yield input(name) match {
        case a: OperandDef => operands += a
        case _ =>
          throw new Exception("Attributes only accept Operands.")
      }

      (dialect: String) =>
        AttributeDef(
          dialect + "." + attrName.toLowerCase,
          attrName,
          operands.toSeq,
          0
        )

    case _ =>
      println(attrName)
      throw new Exception("Attr definition expected.")
}

/** Instantiates a Mirror Product for the given element.
  *
  * @return
  *   Lambda that produces an OperadtionDef given a string of dialect name.
  */
inline def summonOpDef[Elem]: String => OperationDef = {
  getOpDef[Elem](using summonInline[Mirror.ProductOf[Elem]])
}

/** Instantiates a Mirror Product for the given element.
  *
  * @return
  *   Lambda that produces an AttributeDef given a string of dialect name.
  */
inline def summonAttrDef[Elem]: String => AttributeDef = {
  getAttrDef[Elem](using summonInline[Mirror.ProductOf[Elem]])
}

/** Generates a list of OperationDef given enum cases.
  *
  * @param dialect_name
  */
inline def summonDialectOps[Prods <: Tuple](
    dialect_name: String
): ListType[OperationDef] = {

  inline erasedValue[Prods] match
    case _: (prod *: prods) =>
      summonOpDef[prod](dialect_name) +: summonDialectOps[prods](dialect_name)
    case _: EmptyTuple => ListType.empty
}

/** Generates a list of AttributeDef given enum cases.
  *
  * @param dialect_name
  */
inline def summonDialectAttrs[Prods <: Tuple](
    dialect_name: String
): ListType[AttributeDef] = {

  inline erasedValue[Prods] match
    case _: (prod *: prods) =>
      summonAttrDef[prod](dialect_name) +: summonDialectAttrs[prods](
        dialect_name
      )
    case _: EmptyTuple => ListType.empty
}

/** Generates the DialectDef object from the enum definition.
  *
  * @param m
  *   \- Sum Mirror of a given dialect
  */
inline def summonDialect[T1 <: DialectOperation, T2 <: DialectAttribute](using
    m1: Mirror.SumOf[T1],
    m2: Mirror.SumOf[T2]
): DialectDef = {

  val dialect_name = constValue[m1.MirroredLabel].toLowerCase
  val opsDefs = summonDialectOps[m1.MirroredElemTypes](dialect_name)
  val attrDefs = summonDialectAttrs[m2.MirroredElemTypes](dialect_name)

  DialectDef(
    dialect_name,
    opsDefs,
    attrDefs
  )
}

/*≡≡=---=≡≡≡=---=≡≡*\
||     TESTING     ||
\*≡==----=≡=----==≡*/

object FrontEnd {

  // inline def regionindent[T]: String = {
  //   constValue[T].asInstanceOf[Int].toString
  // }

  enum CMathAttr extends DialectAttribute:

    case Complex(
        e1: Operand[IntegerAttr]
    )

  enum CMath extends DialectOperation:

    case Norm(
        e1: Operand[IntegerAttr],
        e2: Result[AnyAttribute],
        e3: Region
    )
    case Mul[Operation](
        e1: Operand[IntegerAttr],
        e2: Result[AnyAttribute]
    )

  object CMath {
    val generator = summonDialect[CMath, CMathAttr]
  }

  def main(args: Array[String]): Unit = {
    println(CMath.generator.print(0))
  }
}
