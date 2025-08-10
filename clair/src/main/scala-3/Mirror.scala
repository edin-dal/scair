package scair.clair.mirrored

import scair.clair.codegen.*
import scair.clair.macros.*
import scair.ir.*

import scala.deriving.*
import scala.quoted.*

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

def getDefVariadicityAndType[Elem: Type](using Quotes): (Variadicity, Type[?]) =
  Type.of[Elem] match
    case '[Option[t]] =>
      (Variadicity.Optional, Type.of[t])
    case '[Seq[t]] =>
      (Variadicity.Variadic, Type.of[t])
    case t =>
      (Variadicity.Single, t)

/** Produces an OpInput to OperationDef given a definition of a Type.
  *
  * @return
  *   Input to OperationDef, either: OperandDef, ResultDef, RegionDef,
  *   SuccessorDef, OpPropertyDef
  */
def getDefInput[Label: Type, Elem: Type](using Quotes): OpInputDef = {
  import quotes.reflect._
  val name = Type.of[Label] match
    case '[String] =>
      Type.valueOfConstant[Label].get.asInstanceOf[String]
  val (variadicity, elem) = getDefVariadicityAndType[Elem]
  elem match
    case '[Result[t]] =>
      ResultDef(
        name = name,
        tpe = Type.of[t],
        variadicity
      )
    case '[Operand[t]] =>
      OperandDef(
        name = name,
        tpe = Type.of[t],
        variadicity
      )
    case '[Region] =>
      RegionDef(
        name = name,
        variadicity
      )
    case '[Successor] =>
      SuccessorDef(
        name = name,
        variadicity
      )
    case '[Attribute] =>
      OpPropertyDef(
        name = name,
        tpe = elem,
        variadicity == Variadicity.Optional
      )
    case _: Type[?] =>
      report.errorAndAbort(
        s"Field ${Type.show[Label]} : ${Type.show[Elem]} is unsupported for MLIR derivation."
      )
}

/** Loops through a Tuple of Input definitions and produces a List of inputs to
  * OperationDef.
  *
  * @return
  *   Lambda that produces an input to OperationDef, given a string
  */
def summonInput[Labels: Type, Elems: Type](using Quotes): List[OpInputDef] = {

  Type.of[(Labels, Elems)] match
    case '[(label *: labels, elem *: elems)] =>
      getDefInput[label, elem] :: summonInput[labels, elems]
    case '[(EmptyTuple, EmptyTuple)] => Nil
}

def getAttrDef[Label: Type, Elem: Type](using
    Quotes
): AttributeParamDef = {
  val name = Type.of[Label] match
    case '[String] =>
      Type.valueOfConstant[Label].get.asInstanceOf[String]

  Type.of[Elem] match
    case '[Attribute] =>
      AttributeParamDef(
        name = name,
        tpe = Type.of[Elem]
      )
    case _ =>
      throw new Exception(
        "Expected this type to be an Attribute"
      )
}

def summonAttrDefs[Labels: Type, Elems: Type](using
    Quotes
): List[AttributeParamDef] = {

  Type.of[(Labels, Elems)] match
    case '[(label *: labels, elem *: elems)] =>
      getAttrDef[label, elem] :: summonAttrDefs[labels, elems]
    case '[(EmptyTuple, EmptyTuple)] => Nil
}

/** Translates a Tuple of string types into a list of strings.
  *
  * @return
  *   Tuple of String types
  */
def stringifyLabels[Elems: Type](using Quotes): List[String] = {

  Type.of[Elems] match
    case '[elem *: elems] =>
      Type
        .valueOfConstant[elem]
        .get
        .asInstanceOf[String] :: stringifyLabels[elems]
    case '[EmptyTuple] => Nil
}

def getDefImpl[T: Type](using quotes: Quotes): OperationDef =
  import quotes.reflect._

  val m = Expr.summon[Mirror.ProductOf[T]].get
  m match
    case '{
          $m: Mirror.ProductOf[T] {
            type MirroredLabel = label; type MirroredElemLabels = elemLabels;
            type MirroredElemTypes = elemTypes
          }
        } =>
      val defname = Type.valueOfConstant[label].get.asInstanceOf[String]

      val paramLabels = stringifyLabels[elemLabels]
      val name = Type.of[T] match
        case '[DerivedOperation[name, _]] =>
          Type.valueOfConstant[name].get
        case _ =>
          report.errorAndAbort(
            s"${Type.show[T]} should extend DerivedOperation to derive DerivedOperationCompanion.",
            TypeRepr.of[T].typeSymbol.pos.get
          )

      val inputs = Type.of[(elemLabels, elemTypes)] match
        case '[(Tuple, Tuple)] => summonInput[elemLabels, elemTypes]
      val opDef = OperationDef(
        name = name,
        className = defname,
        operands = inputs.collect { case a: OperandDef => a },
        results = inputs.collect { case a: ResultDef => a },
        regions = inputs.collect { case a: RegionDef => a },
        successors = inputs.collect { case a: SuccessorDef => a },
        properties = inputs.collect { case a: OpPropertyDef => a },
        assembly_format = None
      )
      val format = Type.of[T] match
        case '[AssemblyFormat[format]] =>
          Some(parseAssemblyFormat(Type.valueOfConstant[format].get, opDef))
        case _ => None
      opDef.copy(assembly_format = format)

def getAttrDefImpl[T: Type](using quotes: Quotes): AttributeDef = {
  import quotes.reflect._

  val m = Expr.summon[Mirror.ProductOf[T]].get
  m match
    case '{
          $m: Mirror.ProductOf[T] {
            type MirroredLabel = label; type MirroredElemLabels = elemLabels;
            type MirroredElemTypes = elemTypes
          }
        } =>
      val defname = Type.valueOfConstant[label].get

      val paramLabels = stringifyLabels[elemLabels]

      val name = Type.of[T] match
        case '[DerivedAttribute[name, _]] =>
          Type.valueOfConstant[name].get
        case _ =>
          report.errorAndAbort(
            s"${Type.show[T]} should extend DerivedAttribute to derive DerivedAttributeCompanion.",
            TypeRepr.of[T].typeSymbol.pos.get
          )

      val attributeDefs = summonAttrDefs[elemLabels, elemTypes]

      AttributeDef(
        name = name,
        attributes = attributeDefs
      )
}
