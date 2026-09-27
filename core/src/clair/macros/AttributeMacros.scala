package scair.clair.macros

import fastparse.*
import scair.clair.*
import scair.ir.*
import scair.parse.*

import scala.quoted.*

def getAttrConstructor[T: Type](
    attrDef: AttributeDef,
    attributes: Expr[Seq[Attribute]],
)(using
    Quotes
): Expr[T] =
  import quotes.reflect.*

  val lengthCheck = Type.of[T] match
    case '[type t <: Attribute; `t`] =>
      '{
        if ${ Expr(attrDef.attributes.length) } != $attributes.length then
          throw new Exception(
            s"Number of attributes ${${ Expr(attrDef.attributes.length) }} does not match the number of provided attributes ${$attributes
                .length}"
          )
      }
    case _ =>
      report
        .errorAndAbort(
          s"Type ${Type.show[T]} needs to be a subtype of Attribute"
        )

  val defs = attrDef.attributes

  val extractedConstructs =
    (defs.zipWithIndex.map((d, i) => '{ ${ attributes }(${ Expr(i) }) }) zip
      defs).map { (a, d) =>
      // expected type of the attribute
      val tpe = d.tpe
      tpe match
        case '[t] =>
          '{
            if !${ a }.isInstanceOf[t] then
              throw Exception(
                s"Expected ${${ Expr(d.name) }} to be of type ${${
                    Expr(Type.show[t])
                  }}, got ${${ a }}"
              )
            ${ a }.asInstanceOf[t]
          }
    }

  val args = (extractedConstructs zip attrDef.attributes)
    .map((e, d) => NamedArg(d.name, e.asTerm))

  val constructorCall = Apply(
    Select(New(TypeTree.of[T]), TypeRepr.of[T].typeSymbol.primaryConstructor),
    List.from(args),
  ).asExprOf[T]

  '{
    $lengthCheck
    $constructorCall
  }

def ADTFlatAttrInputMacro[Def <: AttributeDef: Type](
    attrInputDefs: Seq[AttributeParamDef],
    adtAttrExpr: Expr[?],
)(using Quotes): Expr[Seq[Attribute]] =
  Expr
    .ofList(
      attrInputDefs.map(d => selectMember[Attribute](adtAttrExpr, d.name))
    )

def parametersMacro(
    attrDef: AttributeDef,
    adtAttrExpr: Expr[?],
)(using Quotes): Expr[Seq[Attribute]] =
  ADTFlatAttrInputMacro(attrDef.attributes, adtAttrExpr)

def deriveAttrDefs[T <: Attribute: Type](using
    Quotes
): Expr[AttrDefs[T]] =

  val attrDef = getAttrDefImpl[T]

  '{
    new AttrDefs[T]:
      override def name: String = ${ Expr(attrDef.name) }
      override def parse[$: P as ctx](using p: Parser): P[T] = ${
        getAttrCustomParse[T]('{ p }, '{ ctx })
          .getOrElse(
            '{
              given Whitespace = scair.parse.whitespace
              given Parser = p
              ("<" ~/ attributeP.rep(sep = ",") ~ ">").orElse(Seq())
                .map(x => ${ getAttrConstructor[T](attrDef, '{ x }) })
            }
          )
      }
      def parameters(attr: T): Seq[Attribute] = ${
        parametersMacro(attrDef, '{ attr })
      }
  }
