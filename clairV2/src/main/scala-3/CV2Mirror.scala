package scair.clairV2.mirrored

import scair.clairV2.codegen.*
import scair.clairV2.macros.*
import scair.ir.*

import scala.compiletime._
import scala.deriving._

import scala.Tuple.Zip
import scala.collection.View.Empty

// ░█████╗░ ██╗░░░░░ ░█████╗░ ██╗ ██████╗░ ██╗░░░██╗ ██████╗░
// ██╔══██╗ ██║░░░░░ ██╔══██╗ ██║ ██╔══██╗ ██║░░░██║ ╚════██╗
// ██║░░╚═╝ ██║░░░░░ ███████║ ██║ ██████╔╝ ╚██╗░██╔╝ ░░███╔═╝
// ██║░░██╗ ██║░░░░░ ██╔══██║ ██║ ██╔══██╗ ░╚████╔╝░ ██╔══╝░░
// ╚█████╔╝ ███████╗ ██║░░██║ ██║ ██║░░██║ ░░╚██╔╝░░ ███████╗
// ░╚════╝░ ╚══════╝ ╚═╝░░╚═╝ ╚═╝ ╚═╝░░╚═╝ ░░░╚═╝░░░ ╚══════╝

// ███╗░░░███╗ ██╗ ██████╗░ ██████╗░ ░█████╗░ ██████╗░
// ████╗░████║ ██║ ██╔══██╗ ██╔══██╗ ██╔══██╗ ██╔══██╗
// ██╔████╔██║ ██║ ██████╔╝ ██████╔╝ ██║░░██║ ██████╔╝
// ██║╚██╔╝██║ ██║ ██╔══██╗ ██╔══██╗ ██║░░██║ ██╔══██╗
// ██║░╚═╝░██║ ██║ ██║░░██║ ██║░░██║ ╚█████╔╝ ██║░░██║
// ╚═╝░░░░░╚═╝ ╚═╝ ╚═╝░░╚═╝ ╚═╝░░╚═╝ ░╚════╝░ ╚═╝░░╚═╝

/*≡≡=---=≡≡≡≡≡≡=---=≡≡*\
||    MIRROR LOGIC    ||
\*≡==----=≡≡≡≡=----==≡*/

// currently not supported, but will soon be :)
inline def inputVariadicity[Elem] = inline erasedValue[Elem] match
  case _: Variadic[t] => Variadicity.Variadic
  case _              => Variadicity.Single

/** Produces an OpInput to OperationDef given a definition of a Type.
  *
  * @return
  *   Input to OperationDef, either: OperandDef, ResultDef, RegionDef,
  *   SuccessorDef, OpPropertyDef, OpAttributeDef
  */
inline def getDefInput[Label, Elem]: OpInput = {

  val name = inline erasedValue[Label] match
    case _: String => constValue[Label].asInstanceOf[String]
    case _         => throw new Exception("Internal error!")
  inline erasedValue[Elem] match
    case _: Result[t] =>
      ResultDef(
        id = name,
        typeString = typeToString[t],
        inputVariadicity[Elem]
      )
    case _: Operand[t] =>
      OperandDef(
        id = name,
        typeString = typeToString[t],
        inputVariadicity[Elem]
      )
    case _: Region =>
      RegionDef(
        id = name,
        inputVariadicity[Elem]
      )
    case _: Successor =>
      SuccessorDef(
        id = name,
        inputVariadicity[Elem]
      )
    case _: Property[t] =>
      OpPropertyDef(
        id = name,
        typeString = typeToString[t]
      )
    case _: Attr[t] =>
      OpAttributeDef(
        id = name,
        typeString = typeToString[t]
      )
    case shennanigan =>
      throw new Exception(f"Unsupported shennaigans with field $name")
}

/** Loops through a Tuple of Input definitions and produces a List of inputs to
  * OperationDef.
  *
  * @return
  *   Lambda that produces an input to OperationDef, given a string
  */
inline def summonInput[Labels <: Tuple, Elems <: Tuple]: List[OpInput] = {

  inline erasedValue[(Labels, Elems)] match
    case _: ((label *: labels, elem *: elems)) =>
      getDefInput[label, elem] :: summonInput[labels, elems]
    case _: (EmptyTuple, EmptyTuple) => Nil
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

inline def getMLIRName[T] = inline erasedValue[T] match
  case _: MLIRName[name] => constValue[name]
  case _ =>
    throw new Exception(
      "Expected this type to extend MLIRName with a constant type-parameter."
    )

/** Generates a OperationDef given param m.
  *
  * @param m
  *   \- Mirror Product of an dialect enum case.
  * @return
  *   Lambda that produces an Operadtion Def given a dialect name.
  */
inline def getDef[T](using
    m: Mirror.ProductOf[T]
): OperationDef = {

  val defname = constValue[m.MirroredLabel]
  val paramLabels = stringifyLabels[m.MirroredElemLabels]

  val inputs = summonInput[m.MirroredElemLabels, m.MirroredElemTypes]

  val operands: ListType[OperandDef] = ListType()
  val results: ListType[ResultDef] = ListType()
  val regions: ListType[RegionDef] = ListType()
  val successors: ListType[SuccessorDef] = ListType()
  val opProperty: ListType[OpPropertyDef] = ListType()
  val opAttribute: ListType[OpAttributeDef] = ListType()
  var assembly_format: Option[String] = None

  for (input <- inputs) yield input match {
    case a: OperandDef     => operands += a
    case b: ResultDef      => results += b
    case c: RegionDef      => regions += c
    case d: SuccessorDef   => successors += d
    case e: OpPropertyDef  => opProperty += e
    case f: OpAttributeDef => opAttribute += f
    case _                 => throw new Exception("Internal error!")
  }

  val name = getMLIRName[T]

  OperationDef(
    name,
    defname,
    operands.toSeq,
    results.toSeq,
    regions.toSeq,
    successors.toSeq,
    opProperty.toSeq,
    opAttribute.toSeq,
    assembly_format
  )

}

/** Generates a list of OperationDef given enum cases.
  *
  * @param dialect_name
  */
inline def summonOperationDefs[Prods <: Tuple]: Seq[OperationDef] = {

  inline erasedValue[Prods] match
    case _: (prod *: prods) =>
      getDef[prod](using
        summonInline[Mirror.ProductOf[prod]]
      ) +: summonOperationDefs[prods]

    case _: EmptyTuple => Seq.empty
}
