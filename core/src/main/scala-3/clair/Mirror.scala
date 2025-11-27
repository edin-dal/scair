package scair.clair.mirrored

import fastparse.*
import scair.AttrParser
import scair.Parser
import scair.clair.codegen.*
import scair.clair.macros.*
import scair.core.constraints.*
import scair.ir.*

import scala.deriving.*
import scala.quoted.*
import scala.quoted.Type

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

def getTypeConstraint(tpe: Type[?])(using Quotes) =
  import quotes.reflect.*
  val op = TypeRepr.of[!>]
  TypeRepr.of(using tpe) match
    case AppliedType(op, List(attr, constraint)) =>
      constraint.asType match
        case '[type t <: Constraint; `t`] =>
          Expr.summon[ConstraintImpl[t]] match
            case Some(i) => Some(i)
            case None    =>
              Implicits.search(TypeRepr.of[ConstraintImpl[t]]) match
                case s: ImplicitSearchSuccess =>
                  Some(s.tree.asExprOf[ConstraintImpl[t]])
                case f: ImplicitSearchFailure =>
                  report.errorAndAbort(
                    s"Could not find an implementation for constraint ${Type.show[t]}:\n${f.explanation}"
                  )
    case _ =>
      None

def getDefType(elem: Type[?])(using Quotes) =
  elem match
    case '[Result[t]] =>
      Type.of[t]
    case '[Operand[t]] =>
      Type.of[t]
    case '[Region] =>
      Type.of[Attribute]
    case '[Successor] =>
      Type.of[Attribute]
    case t @ '[Attribute] =>
      t

def getDefVariadicityAndType[Elem: Type](using Quotes): (Variadicity, Type[?]) =
  Type.of[Elem] match
    // This first case is to catch Attributes that would also implement Seq or Option.
    case '[type t <: Attribute; `t`] =>
      (Variadicity.Single, Type.of[t])
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
def getDefInput[Label: Type, Elem: Type](using Quotes): OpInputDef =
  import quotes.reflect.*
  val name = Type.of[Label] match
    case '[String] =>
      Type.valueOfConstant[Label].get.asInstanceOf[String]
  val (variadicity, elem) = getDefVariadicityAndType[Elem]
  val (tpe) = getDefType(elem)
  val constraint = getTypeConstraint(tpe)

  elem match
    case '[Value[t]] if TypeRepr.of[Result[t]] =:= TypeRepr.of(using elem) =>
      ResultDef(
        name = name,
        tpe = tpe,
        variadicity,
        constraint
      )
    case '[Value[t]] =>
      OperandDef(
        name = name,
        tpe = tpe,
        variadicity,
        constraint
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
      variadicity match
        case Variadicity.Variadic =>
          report.errorAndAbort(
            s"Variadic properties are not supported; use ArrayAttribute[${Type
                .show(using elem)}] instead."
          )
        case v @ (Variadicity.Single | Variadicity.Optional) =>
          OpPropertyDef(
            name = name,
            tpe = tpe,
            v,
            constraint
          )
    case _: Type[?] =>
      report.errorAndAbort(
        s"Field ${Type.show[Label]} : ${Type.show[Elem]} is unsupported for MLIR derivation."
      )

/** Loops through a Tuple of Input definitions and produces a List of inputs to
  * OperationDef.
  *
  * @return
  *   Lambda that produces an input to OperationDef, given a string
  */
def summonInput[Labels: Type, Elems: Type](using Quotes): List[OpInputDef] =

  Type.of[(Labels, Elems)] match
    case '[(label *: labels, elem *: elems)] =>
      getDefInput[label, elem] :: summonInput[labels, elems]
    case '[(EmptyTuple, EmptyTuple)] => Nil

def getAttrDef[Label: Type, Elem: Type](using
    Quotes
): AttributeParamDef =
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

def summonAttrDefs[Labels: Type, Elems: Type](using
    Quotes
): List[AttributeParamDef] =

  Type.of[(Labels, Elems)] match
    case '[(label *: labels, elem *: elems)] =>
      getAttrDef[label, elem] :: summonAttrDefs[labels, elems]
    case '[(EmptyTuple, EmptyTuple)] => Nil

/** Translates a Tuple of string types into a list of strings.
  *
  * @return
  *   Tuple of String types
  */
def stringifyLabels[Elems: Type](using Quotes): List[String] =

  Type.of[Elems] match
    case '[elem *: elems] =>
      Type
        .valueOfConstant[elem]
        .get
        .asInstanceOf[String] :: stringifyLabels[elems]
    case '[EmptyTuple] => Nil

def getDefImpl[T <: Operation: Type](using quotes: Quotes): OperationDef =
  import quotes.reflect.*

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
        case '[DerivedOperation[name, T]] =>
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

def getCompanion[T: Type](using quotes: Quotes) =
  import quotes.reflect.*
  TypeRepr.of[T].typeSymbol.companionModule

def getOpCustomParse[T <: Operation: Type](
    p: Expr[Parser],
    resNames: Expr[Seq[String]]
)(using
    quotes: Quotes
) =
  import quotes.reflect.*

  val comp = getCompanion(using Type.of[T])
  val sig = TypeRepr
    .of[OperationCompanion[T]]
    .typeSymbol
    .declaredMethod("parse")
    .head
    .signature
  comp.methodMember("parse").filter(_.signature == sig) match
    case Seq(m) =>
      val callTerm = Select
        .unique(Ref(comp), m.name)
        .appliedToType(TypeRepr.of[Any])
        .appliedTo(p.asTerm, resNames.asTerm)
        .etaExpand(comp)
        .asExprOf[P[Any] => P[T]]
      Some('{ (ctx: P[Any]) ?=> ${ callTerm }(ctx) })
    case Seq() =>
      None
    case d: Seq[?] =>
      report.errorAndAbort(
        s"Multiple companion parse methods not supported at this point."
      )

def getAttrCustomParse[T: Type](p: Expr[AttrParser], ctx: Expr[P[Any]])(using
    quotes: Quotes
) =
  import quotes.reflect.*

  val comp = getCompanion(using Type.of[T])
  val sig = TypeRepr
    .of[AttributeCompanion]
    .typeSymbol
    .declaredMethod("parse")
    .head
    .signature
  comp.methodMember("parse").filter(_.signature == sig) match
    case Seq(m) =>
      val callTerm = Select
        .unique(Ref(comp), m.name)
        .appliedToType(TypeRepr.of[Any])
        .appliedTo(p.asTerm)
        .appliedTo(ctx.asTerm)
        .etaExpand(comp)
        .asExprOf[P[T]]
      Some(callTerm)
    case Seq() =>
      None
    case d: Seq[?] =>
      report.errorAndAbort(
        s"Multiple companion parse methods not supported at this point."
      )

def getAttrDefImpl[T: Type](using quotes: Quotes): AttributeDef =
  import quotes.reflect.*

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
        case '[DerivedAttribute[name, ?]] =>
          Type.valueOfConstant[name].get
        case _ =>
          report.errorAndAbort(
            s"${Type.show[T]} should extend DerivedAttribute.DerivedOperation to derive DerivedAttributeCompanion.",
            TypeRepr.of[T].typeSymbol.pos.get
          )

      val attributeDefs = summonAttrDefs[elemLabels, elemTypes]

      AttributeDef(
        name = name,
        attributes = attributeDefs
      )
