package scair.clairV2.mirrored

import scair.clairV2.codegen.*
import scair.clairV2.macros.*
import scair.ir.*

import scala.compiletime.*
import scala.deriving.*

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
inline def getDefInput[Elem]: String => OpInput = {
  inline erasedValue[Elem] match
    case _: Operand[t] =>
      (name: String) =>
        OperandDef(
          id = name,
          typeString = typeToString[t],
          inputVariadicity[Elem]
        )
    case _: Result[t] =>
      (name: String) =>
        ResultDef(
          id = name,
          typeString = typeToString[t],
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
          typeString = typeToString[t]
        )
    case _: Attr[t] =>
      (name: String) =>
        OpAttributeDef(
          id = name,
          typeString = typeToString[t]
        )
    case shennanigan =>
      (name: String) =>
        throw new Exception(f"Unsupported shennaigans with field $name")
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
inline def getDef[T](dialectName: String)(using
    m: Mirror.ProductOf[T]
): String => OperationDef = {

  val defname = constValue[m.MirroredLabel]
  val paramLabels = stringifyLabels[m.MirroredElemLabels]

  inline erasedValue[T] match
    case _: ADTOperation =>
      val inputs = summonInput[m.MirroredElemTypes]

      val operands: ListType[OperandDef] = ListType()
      val results: ListType[ResultDef] = ListType()
      val regions: ListType[RegionDef] = ListType()
      val successors: ListType[SuccessorDef] = ListType()
      val opProperty: ListType[OpPropertyDef] = ListType()
      val opAttribute: ListType[OpAttributeDef] = ListType()
      var assembly_format: Option[String] = None

      for ((name, input) <- paramLabels zip inputs) yield input(name) match {
        case a: OperandDef     => operands += a
        case b: ResultDef      => results += b
        case c: RegionDef      => regions += c
        case d: SuccessorDef   => successors += d
        case e: OpPropertyDef  => opProperty += e
        case f: OpAttributeDef => opAttribute += f
        case _                 => throw new Exception("Internal error!")
      }

      (packageName: String) =>
        OperationDef(
          dialectName,
          defname.toLowerCase,
          defname,
          packageName,
          operands.toSeq,
          results.toSeq,
          regions.toSeq,
          successors.toSeq,
          opProperty.toSeq,
          opAttribute.toSeq,
          assembly_format
        )

    case _ =>
      throw new Exception("ADTOperation definition expected.")

}

/** Instantiates a Mirror Product for the given element.
  *
  * @return
  *   Lambda that produces an OperadtionDef given a string of dialect name.
  */
inline def summonDef[Elem](dialectName: String): OperationDef = {
  val opDef = getDef[Elem](dialectName)(using summonInline[Mirror.ProductOf[Elem]])

  val packageName = getClassPath[Elem]
  val withoutLastSegment =
    packageName.substring(0, packageName.lastIndexOf('.'))
  val finalPkgName = withoutLastSegment.replace("$", "")

  opDef(finalPkgName)
}

/** Generates a list of OperationDef given enum cases.
  *
  * @param dialect_name
  */
inline def summonOperationDefs[Prods <: Tuple](dialectName: String): Seq[OperationDef] = {

  inline erasedValue[Prods] match
    case _: (prod *: prods) =>
      summonDef[prod](dialectName) +: summonOperationDefs[prods](dialectName)

    case _: EmptyTuple => Seq.empty
}

/** Generates the DialectDef object from the enum definition.
  *
  * @param m
  *   \- Sum Mirror of a given dialect
  */
inline def summonMLIROps[Prods <: Tuple](dialectName: String): MLIROpDef = {

  val defs = summonOperationDefs[Prods](dialectName)

  MLIROpDef(
    dialectName,
    defs
  )
}
