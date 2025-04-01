package scair.clair.testAttrMacros

import fastparse.*
import scala.quoted.*
import scair.AttrParser
import scair.Parser.*
import scair.ir.Attribute
import scair.clair.mirrored._
import scair.clair.codegen._
import scair.ir.*
import scala.compiletime._
import scala.deriving.Mirror

def getConstructor[T: Type](
    attrDef: AttributeDef,
    attributes: Expr[Seq[Attribute]]
)(using
    Quotes
): Expr[T] = {
  import quotes.reflect._

  if !(TypeRepr.of[T] <:< TypeRepr.of[Attribute]) then
    throw new Exception(
      s"Type ${Type.show[T]} needs to be a subtype of Attribute"
    )

  '{
    if ${ Expr(attrDef.attributes.length) } != $attributes.length then
      throw new Exception(
        s"Number of attributes ${${ Expr(attrDef.attributes.length) }} does not match the number of provided attributes ${$attributes.length}"
      )
  }

  val defs = attrDef.attributes

  val verifiedConstructs = (defs.zipWithIndex.map((d, i) =>
    '{ ${ attributes }(${ Expr(i) }) }
  ) zip defs)
    .map { (a, d) =>
      // expected type of the attribute
      val tpe = d.tpe
      tpe match
        case '[t] =>
          '{
            if (!${ a }.isInstanceOf[t & Attribute]) then
              throw Exception(
                s"Expected ${${ Expr(d.name) }} to be of type ${${
                    Expr(Type.show[t])
                  }}, got ${${ a }}"
              )
            ${ a }.asInstanceOf[t & Attribute]
          }
    }

  val args = (verifiedConstructs zip attrDef.attributes)
    .map((e, d) => NamedArg(d.name, e.asTerm))

  Apply(
    Select(New(TypeTree.of[T]), TypeRepr.of[T].typeSymbol.primaryConstructor),
    List.from(args)
  ).asExprOf[T]
}

object AttributeTrait {

  inline def derived[T]: AttributeTrait[T] = ${ derivedImpl[T] }

  def derivedImpl[T: Type](using Quotes): Expr[AttributeTrait[T]] =

    val attrDef = getAttrDefImpl[T]

    '{
      new AttributeTrait[T] {
        override def name: String = ${ Expr(attrDef.name) }
        override def parse[$: P](p: AttrParser): P[T] = P(
          ("<" ~/ p.Type.rep(sep = ",") ~ ">")
        ).orElse(Seq())
          .map(x => ${ getConstructor[T](attrDef, '{ x }) })
      }
    }

}

trait AttributeTrait[T] extends AttributeTraitI[T] {
  extension (op: T) override def AttributeTrait = this
}

inline def summonAttributeTraits[T <: Tuple]: Seq[AttributeTrait[_]] =
  inline erasedValue[T] match
    case _: (t *: ts) =>
      // slight workaround on that &: TODO -> get rid of it ;)
      AttributeTrait.derived[t] +: summonAttributeTraits[ts]
    case _: EmptyTuple => Seq()
