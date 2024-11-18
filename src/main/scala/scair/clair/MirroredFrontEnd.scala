package scair.clair.mirrored

import scala.deriving._
import scala.compiletime._
import scair.scairdl.irdef._
import scair.scairdl.constraints._
import Variadicity._

/*≡≡=---=≡≡≡≡≡≡≡≡≡=---=≡≡*\
||   DIFFERENT CLASSES   ||
\*≡==----=≡≡≡≡≡≡≡=----==≡*/

abstract class Dialect

abstract class Input[T]
case class Operand[T]() extends Input[T]
case class Result[T]() extends Input[T]
case class Region() extends Input[Nothing]
case class Successor() extends Input[Nothing]
case class Property[T]() extends Input[T]
case class Attribute[T]() extends Input[T]

/*≡≡=---=≡≡≡≡≡≡=---=≡≡*\
||    MIRROR LOGIC    ||
\*≡==----=≡≡≡≡=----==≡*/

/** Given a type, produces a corresponding constraint.
  *
  * @return
  *   IRDLConstraint
  */
inline def indent[T]: IRDLConstraint = {
  inline erasedValue[T] match
    case AnyAttr => AnyAttr
    case _ =>
      throw new Exception(
        "The Clair front-end currently supports only AnyAttr Constraint"
      )
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
          indent[t],
          Single
        )
    case _: Result[t] =>
      (name: String) =>
        ResultDef(
          id = name,
          indent[t],
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
          indent[t]
        )
    case _: Attribute[t] =>
      (name: String) =>
        OpAttributeDef(
          id = name,
          indent[t]
        )
    case _ =>
      throw new Exception(
        "You can only pass in Operand, Result, Region, Successor, Property or Attribute to the Operation definition"
      )
}

/** Loops through a Tuple of Input definitions and produces a List of inputs to
  * OperationDef.
  *
  * @return
  *   Lambda that produces an input to OperationDef, given a string
  */
inline def summonOpInput[Elems <: Tuple]: List[String => OpInput] = {
  inline erasedValue[Elems] match
    case _: (elem *: elems) => getOpInput[elem] :: summonOpInput[elems]
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
    case _: Dialect =>
      val inputs = summonOpInput[m.MirroredElemTypes]

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
      throw new Exception("bruh, don't even try this.")
}

/** Instantiates a Mirror Product for the given element.
  *
  * @return
  *   Lambda that produces an Operadtion Def given a dialect name.
  */
inline def summonOpDef[Elem]: String => OperationDef = {
  getOpDef[Elem](using summonInline[Mirror.ProductOf[Elem]])
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

/** Generates the DialectDef object from the enum definition.
  *
  * @param m
  *   \- Sum Mirror of a given dialect
  */
inline def summonDialect[T <: Dialect](using
    m: Mirror.SumOf[T]
): DialectDef = {
  val dialect_name = constValue[m.MirroredLabel].toLowerCase
  val elems = summonDialectOps[m.MirroredElemTypes](dialect_name)

  DialectDef(
    dialect_name,
    elems,
    ListType()
  )
}

/*≡≡=---=≡≡≡=---=≡≡*\
||     TESTING     ||
\*≡==----=≡=----==≡*/

object FrontEnd {

  // inline def regionindent[T]: String = {
  //   constValue[T].asInstanceOf[Int].toString
  // }

  enum CMath extends Dialect:
    case Norm(
        e1: Operand[AnyAttr.type],
        e2: Result[AnyAttr.type],
        e3: Region
    )
    case Mul(
        e1: Operand[AnyAttr.type],
        e2: Result[AnyAttr.type]
    )

  object CMath {
    val generator = summonDialect[CMath]
  }

  def main(args: Array[String]): Unit = {
    println(CMath.generator.print(0))
  }
}
