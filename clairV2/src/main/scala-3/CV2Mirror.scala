package scair.clairV2.mirrored

import scair.clairV2.codegen.*
import scair.clairV2.macros.*
import scair.ir.*

import scala.compiletime._
import scala.deriving._

import scala.Tuple.Zip
import scala.collection.View.Empty
import scala.quoted._

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

/** Produces an OpInput to OperationDef given a definition of a Type.
  *
  * @return
  *   Input to OperationDef, either: OperandDef, ResultDef, RegionDef,
  *   SuccessorDef, OpPropertyDef, OpAttributeDef
  */
def getDefInput[Label: Type, Elem: Type](using Quotes): OpInputDef = {
  val name = Type.of[Label] match
    case '[String] =>
      Type.valueOfConstant[Label].get.asInstanceOf[String]

  Type.of[Elem] match
    case '[Variadic[Result[t]]] =>
      ResultDef(
        name = name,
        tpe = Type.of[t],
        Variadicity.Variadic
      )
    case '[Variadic[Operand[t]]] =>
      OperandDef(
        name = name,
        tpe = Type.of[t],
        Variadicity.Variadic
      )
    case '[Result[t]] =>
      ResultDef(
        name = name,
        tpe = Type.of[t],
        Variadicity.Single
      )
    case '[Operand[t]] =>
      OperandDef(
        name = name,
        tpe = Type.of[t],
        Variadicity.Single
      )
    case '[Region] =>
      RegionDef(
        name = name,
        Variadicity.Single
      )
    case '[Successor] =>
      SuccessorDef(
        name = name,
        Variadicity.Single
      )
    case '[Property[t]] =>
      OpPropertyDef(
        name = name,
        tpe = Type.of[t]
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

inline def getMLIRName[T] = inline erasedValue[T] match
  case _: MLIRName[name] => constValue[name]
  case _ =>
    throw new Exception(
      "Expected this type to extend MLIRName with a constant type-parameter."
    )

def getDefImpl[T: Type](using quotes: Quotes): OperationDef =

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
        case '[MLIRName[name]] =>
          Type.valueOfConstant[name].get.asInstanceOf[String]
      val assemblyFormat = Type.of[T] match
        case '[MLIRFormat[format]] =>
          val assemblyFormatString =
            Type.valueOfConstant[format].get.asInstanceOf[String]
          Some(Parseassemblyformat(assemblyFormatString))
        case _ => None

      val inputs = Type.of[(elemLabels, elemTypes)] match
        case _: Type[(Tuple, Tuple)] => summonInput[elemLabels, elemTypes]
      val e = OperationDef(
        name = name,
        className = defname,
        operands = inputs.collect { case a: OperandDef => a },
        results = inputs.collect { case a: ResultDef => a },
        regions = inputs.collect { case a: RegionDef => a },
        successors = inputs.collect { case a: SuccessorDef => a },
        properties = inputs.collect { case a: OpPropertyDef => a },
        assembly_format = assemblyFormat
      )
      e
