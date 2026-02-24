package scair.clair.mirrored

import fastparse.*
import scair.clair.codegen.*
import scair.clair.macros.*
import scair.core.constraints.*
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

/** Extracts type constraint information from a type if it has the form `!>`
  * (constrained type operator).
  *
  * @param tpe
  *   The type to check for constraints.
  * @return
  *   Some(constraint implementation) if the type is constrained, None
  *   otherwise.
  */
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

/** Gets the definition type from an element type. For operands and results,
  * extracts the inner attribute type. For regions and successors, returns
  * Attribute.
  *
  * @param elem
  *   The element type to extract from.
  * @return
  *   The extracted type.
  */
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

/** Gets the variadicity and type from an element. Handles Option (optional),
  * Seq (variadic), and direct types (single).
  *
  * @tparam Elem
  *   The element type to analyze.
  * @return
  *   A tuple of (Variadicity, Type) representing the element's characteristics.
  */
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

/** Produces an OpInputDef from a label (field name) and element type pair.
  * Dispatches to the appropriate definition type based on the element's actual
  * type.
  *
  * @tparam Label
  *   The field label type (must be a String constant).
  * @tparam Elem
  *   The field element type.
  * @return
  *   The appropriate OpInputDef (OperandDef, ResultDef, RegionDef,
  *   SuccessorDef, or OpPropertyDef).
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
        constraint,
      )
    case '[Value[t]] =>
      OperandDef(
        name = name,
        tpe = tpe,
        variadicity,
        constraint,
      )
    case '[Region] =>
      RegionDef(
        name = name,
        variadicity,
      )
    case '[Successor] =>
      SuccessorDef(
        name = name,
        variadicity,
      )
    case '[Attribute] =>
      variadicity match
        case Variadicity.Variadic =>
          report
            .errorAndAbort(
              s"Variadic properties are not supported; use ArrayAttribute[${Type
                  .show(using elem)}] instead."
            )
        case v @ (Variadicity.Single | Variadicity.Optional) =>
          OpPropertyDef(
            name = name,
            tpe = tpe,
            v,
            constraint,
          )
    case _: Type[?] =>
      report.errorAndAbort(
        s"Field ${Type.show[Label]} : ${Type.show[Elem]} is unsupported for MLIR derivation."
      )

/** Loops through a Tuple of label-element pairs and produces a List of
  * OpInputDefs.
  *
  * @tparam Labels
  *   Tuple of label types (String constants).
  * @tparam Elems
  *   Tuple of element types.
  * @return
  *   List of OpInputDefs representing all fields.
  */
def summonInput[Labels: Type, Elems: Type](using Quotes): List[OpInputDef] =

  Type.of[(Labels, Elems)] match
    case '[(label *: labels, elem *: elems)] =>
      getDefInput[label, elem] :: summonInput[labels, elems]
    case '[(EmptyTuple, EmptyTuple)] => Nil

/** Produces an AttributeParamDef from a label and element type pair. Used for
  * attribute parameters.
  *
  * @tparam Label
  *   The parameter label type (must be a String constant).
  * @tparam Elem
  *   The parameter element type (must be an Attribute).
  * @return
  *   An AttributeParamDef for this parameter.
  * @throws Exception
  *   If Elem is not an Attribute subtype.
  */
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

/** Loops through a Tuple of label-element pairs for attributes and produces a
  * List of AttributeParamDefs.
  *
  * @tparam Labels
  *   Tuple of label types (String constants).
  * @tparam Elems
  *   Tuple of element types (must be Attributes).
  * @return
  *   List of AttributeParamDefs representing all attribute parameters.
  */
def summonAttrDefs[Labels: Type, Elems: Type](using
    Quotes
): List[AttributeParamDef] =

  Type.of[(Labels, Elems)] match
    case '[(label *: labels, elem *: elems)] =>
      getAttrDef[label, elem] :: summonAttrDefs[labels, elems]
    case '[(EmptyTuple, EmptyTuple)] => Nil

/** Translates a Tuple of string constant types into a list of strings.
  *
  * @tparam Elems
  *   Tuple of String constant types.
  * @return
  *   List of the actual string values.
  */
def stringifyLabels[Elems: Type](using Quotes): List[String] =

  Type.of[Elems] match
    case '[elem *: elems] =>
      Type.valueOfConstant[elem].get.asInstanceOf[String] ::
        stringifyLabels[elems]
    case '[EmptyTuple] => Nil

/** Derives an OperationDef from an Operation type using Scala 3's Mirror API.
  * Extracts all field information including operands, results, regions,
  * successors, and properties.
  *
  * @tparam T
  *   The Operation type to derive from (must extend DerivedOperation).
  * @return
  *   The derived OperationDef.
  */
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
            TypeRepr.of[T].typeSymbol.pos.get,
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
        assemblyFormat = None,
      )
      val format = Type.of[T] match
        case '[AssemblyFormat[format]] =>
          Some(parseAssemblyFormat(Type.valueOfConstant[format].get, opDef))
        case _ => None
      opDef.copy(assemblyFormat = format)

/** Gets the companion module symbol for a type.
  *
  * @tparam T
  *   The type to get the companion for.
  * @return
  *   The companion module symbol.
  */
def getCompanion[T: Type](using quotes: Quotes) =
  import quotes.reflect.*
  TypeRepr.of[T].typeSymbol.companionModule

/** Looks up a custom parser for an operation type if one is defined via
  * OperationCustomParser.
  *
  * @tparam T
  *   The operation type.
  * @param p
  *   The Parser expression.
  * @param resNames
  *   The result names expression.
  * @return
  *   Some(parser expression) if a custom parser is defined, None otherwise.
  */
def getOpCustomParse[T <: Operation: Type](
    p: Expr[Parser],
    resNames: Expr[Seq[String]],
)(using
    quotes: Quotes
) =
  Expr.summon[OperationCustomParser[T]].map(parser =>
    '{ (ctx: P[Any]) ?=> $parser.parse($resNames)(using ctx, $p) }
  )

/** Looks up a custom parser for an attribute type if one is defined via
  * AttributeCustomParser.
  *
  * @tparam T
  *   The attribute type.
  * @param p
  *   The Parser expression.
  * @param ctx
  *   The parsing context expression.
  * @return
  *   Some(parser expression) if a custom parser is defined, None otherwise.
  */
def getAttrCustomParse[T <: Attribute: Type](
    p: Expr[Parser],
    ctx: Expr[P[Any]],
)(using
    quotes: Quotes
) = Expr.summon[AttributeCustomParser[T]]
  .map(parser => '{ $parser.parse(using $ctx, $p) })

/** Derives an AttributeDef from an Attribute type using Scala 3's Mirror API.
  * Extracts all parameter information.
  *
  * @tparam T
  *   The Attribute type to derive from (must extend DerivedAttribute).
  * @return
  *   The derived AttributeDef.
  */
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
            TypeRepr.of[T].typeSymbol.pos.get,
          )

      val attributeDefs = summonAttrDefs[elemLabels, elemTypes]

      AttributeDef(
        name = name,
        attributes = attributeDefs,
      )
