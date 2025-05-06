package scair.clair.mirrored

import scair.clair.codegen.*
import scair.clair.macros.*
import scair.ir.*

import scala.deriving.*
import scala.quoted.*
import scair.clair.constraint.*

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

def cookConstrainedType[C <: Constraint[_] : Type](using Quotes) = 
  import quotes.reflect.*
  Type.of[C] match
    case '[Constraint[bound]] =>
      val symbol = TypeRepr.of[C].termSymbol
      throw new NotImplementedError(
        s"What: ${TypeRepr.of[C].show}, ${Expr.summon[C]}, ${TypeRepr.of[C].termSymbol}, ${symbol.isDefinedInCurrentRun}"
      )
    case _ => throw new Exception(
      s"Expected this type to be a Constraint"
    )

/*≡≡=---=≡≡≡≡≡≡=---=≡≡*\
||    MIRROR LOGIC    ||
\*≡==----=≡≡≡≡=----==≡*/

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

  Type.of[Elem] match
    case '[Seq[Result[t]]] =>
      ResultDef(
        name = name,
        tpe = Type.of[t],
        Variadicity.Variadic
      )
    case '[Seq[Operand[t]]] =>
      Type.of[t] match
        // case '[Constrained[tt]] => throw new Exception(
        //   s"What: ${Type.of[tt]}"
        // )
        case _ =>
      
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
      Type.of[t] match
        case '[Constrained[cstr]] =>
          cookConstrainedType[cstr]
        case _ =>
          println(s"$name:${TypeTree.of[t].show},${TypeRepr.of[t].show}")
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
    case '[t] if TypeRepr.of[t] <:< TypeRepr.of[Attribute] =>
      OpPropertyDef(
        name = name,
        tpe = Type.of[t]
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
          Type.valueOfConstant[name].get.asInstanceOf[String]
        case _ =>
          report.errorAndAbort(
            s"${Type.show[T]} should extend DerivedOperation to derive DerivedOperationCompanion.",
            TypeRepr.of[T].typeSymbol.pos.get
          )

      val inputs = Type.of[(elemLabels, elemTypes)] match
        case '[(Tuple, Tuple)] => summonInput[elemLabels, elemTypes]
      val e = OperationDef(
        name = name,
        className = defname,
        operands = inputs.collect { case a: OperandDef => a },
        results = inputs.collect { case a: ResultDef => a },
        regions = inputs.collect { case a: RegionDef => a },
        successors = inputs.collect { case a: SuccessorDef => a },
        properties = inputs.collect { case a: OpPropertyDef => a },
        assembly_format = None
      )
      e

def getAttrDefImpl[T: Type](using quotes: Quotes): AttributeDef = {
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

      val attributeDefs = Type.of[(elemLabels, elemTypes)] match
        case _: Type[(Tuple, Tuple)] => summonAttrDefs[elemLabels, elemTypes]

      AttributeDef(
        name = name,
        attributes = attributeDefs
      )
}

inline def getNameDefBlaBla[T] = ${ getNameDefBlaBlaImpl[T] }

def getNameDefBlaBlaImpl[T: Type](using quotes: Quotes): Expr[String] =
  Expr(getDefImpl[T].name)
