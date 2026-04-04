package scair.clair.macros

import fastparse.*
import scair.clair.*
import scair.constraints.*
import scair.ir.*
import scair.parse.Parser

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
                  report
                    .errorAndAbort(
                      s"Could not find an implementation for constraint ${Type
                          .show[t]}:\n${f.explanation}"
                    )
    case _ =>
      None

def normalizeDefElem(elem: Type[?])(using
    Quotes
): (Variadicity, Type[?]) =
  elem match
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
def getInputDef(label: String, elem: Type[?])(using
    defaults: Map[String, Expr[Any]]
)(using Quotes): OpInputDef =
  import quotes.reflect.*
  val (variadicity, constructType) = normalizeDefElem(elem)

  constructType match
    case '[Value[t]]
        if TypeRepr.of[Result[t]] =:= TypeRepr.of(using constructType) =>
      val tpe = Type.of[t]
      ResultDef(
        name = label,
        tpe = tpe,
        variadicity,
        getTypeConstraint(tpe),
      )
    case '[Value[t]] =>
      val tpe = Type.of[t]
      OperandDef(
        name = label,
        tpe = tpe,
        variadicity,
        getTypeConstraint(tpe),
      )
    case '[Region] =>
      RegionDef(
        name = label,
        variadicity,
      )
    case '[Successor] =>
      SuccessorDef(
        name = label,
        variadicity,
      )
    case tpe @ '[Attribute] =>
      variadicity match
        case Variadicity.Variadic =>
          report
            .errorAndAbort(
              s"Variadic properties are not supported; use ArrayAttribute[${Type
                  .show(using elem)}] instead."
            )
        case v @ (Variadicity.Single | Variadicity.Optional) =>
          OpPropertyDef(
            name = label,
            tpe = tpe,
            v,
            getTypeConstraint(tpe),
            defaults.get(label),
          )
    case _: Type[?] =>
      report.errorAndAbort(
        s"Field $label : ${Type.show(using elem)} is unsupported for MLIR derivation."
      )

/** Loops through a Tuple of Input definitions and produces a List of inputs to
  * OperationDef.
  *
  * @return
  *   Lambda that produces an input to OperationDef, given a string
  */
def getInputDefs(params: Seq[(name: String, tpe: Type[?])])(using
    defaults: Map[String, Expr[Any]]
)(using Quotes): Seq[OpInputDef] =

  params.map(p => getInputDef(p.name, p.tpe))

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
        tpe = Type.of[Elem],
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
      Type.valueOfConstant[elem].get.asInstanceOf[String] ::
        stringifyLabels[elems]
    case '[EmptyTuple] => Nil

/** For a case class T, returns a dictionary mapping field names to expressions
  * representing the default value of that field, if defined.
  */
def defaultExprs[T: Type](using Quotes): Map[String, Expr[Any]] =
  import quotes.reflect.*

  val sym = TypeRepr.of[T].typeSymbol
  val companion = sym.companionModule
  val fields = sym.caseFields

  fields.zipWithIndex.flatMap { (f, i) =>
    val fieldName = f.name
    // Scala generates those methods on the companion object of case classes for default parameters values.
    val methodName =
      s"$$lessinit$$greater$$default$$${i + 1}"
    companion.declaredMethod(methodName) match
      case List(m) =>
        // Select the method off the companion object reference
        val defaultValueSymbol = Ref(companion).select(m)
        val typeArgs = TypeRepr.of[T].typeArgs
        val typeApplied = defaultValueSymbol.appliedToTypes(typeArgs)
        Some(fieldName -> typeApplied.asExpr)
      case _ =>
        None

  }.toMap

def getDefImpl[T <: Operation: Type](using quotes: Quotes): OperationDef =
  import quotes.reflect.*

  val typerepr = TypeRepr.of[T]
  val defname = typerepr.typeSymbol.name
  val fields = typerepr.typeSymbol.caseFields

  val params = fields
    .map(f => (name = f.name, tpe = typerepr.memberType(f).asType))

  val name = Type.of[T] match
    case '[DerivedOperation[name]] =>
      Type.valueOfConstant[name].get
    case _ =>
      report.errorAndAbort(
        s"${Type.show[T]} should extend DerivedOperation to derive DerivedOperationCompanion.",
        TypeRepr.of[T].typeSymbol.pos.get,
      )
  val defaults = defaultExprs[T]
  val inputs = getInputDefs(params)(using defaults)
  val opDef = OperationDef(
    name = name,
    className = defname,
    operands = inputs.collect { case a: OperandDef => a },
    results = inputs.collect { case a: ResultDef => a },
    regions = inputs.collect { case a: RegionDef => a },
    successors = inputs.collect { case a: SuccessorDef => a },
    properties = inputs.collect { case a: OpPropertyDef => a },
    assemblyFormat = None,
  )
  val format = Type.of[T] match
    case '[AssemblyFormat[format]] =>
      Some(parseAssemblyFormat(Type.valueOfConstant[format].get, opDef))
    case _ => None
  opDef.copy(assemblyFormat = format)

def getCompanion[T: Type](using quotes: Quotes) =
  import quotes.reflect.*
  TypeRepr.of[T].typeSymbol.companionModule

def getOpCustomParse[T <: Operation: Type](
    p: Expr[Parser],
    resNames: Expr[Seq[String]],
)(using
    quotes: Quotes
) =
  Expr.summon[OperationCustomParser[T]].map(parser =>
    '{ (ctx: P[Any]) ?=> $parser.parse($resNames)(using ctx, $p) }
  )

def getAttrCustomParse[T <: Attribute: Type](
    p: Expr[Parser],
    ctx: Expr[P[Any]],
)(using
    quotes: Quotes
) = Expr.summon[AttributeCustomParser[T]]
  .map(parser => '{ $parser.parse(using $ctx, $p) })

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
        case '[DerivedAttribute[name]] =>
          Type.valueOfConstant[name].get
        case _ =>
          report.errorAndAbort(
            s"${Type.show[T]} should extend DerivedAttribute.DerivedOperation to derive AttrDefs.",
            TypeRepr.of[T].typeSymbol.pos.get,
          )

      val attributeDefs = summonAttrDefs[elemLabels, elemTypes]

      AttributeDef(
        name = name,
        attributes = attributeDefs,
      )
