package scair.clair.macros

import fastparse.*
import scair.clair.*
import scair.ir.*
import scair.parse.Parser

import scala.deriving.*
import scala.quoted.*

def getAttrDef[Label: Type, Elem: Type](using
    Quotes
): AttributeParamDef =
  val name = Type.of[Label] match
    case '[String] =>
      Type.valueOfConstant[Label].get.asInstanceOf[String]

  Type.of[Elem] match
    case '[type t <: Attribute; `t`] =>
      AttributeParamDef(
        name = name,
        tpe = Type.of[t],
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
