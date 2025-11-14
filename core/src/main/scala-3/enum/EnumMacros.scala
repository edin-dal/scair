package scair.eenum.macros

import scala.quoted.*
import scair.ir.Attribute
import scair.dialects.builtin.{IntegerAttr, IntData}

inline def enumFromOrdinalFunction[E <: scala.reflect.Enum]: Int => E =
  ${ enumFromOrdinalFunc[E] }

def enumFromPropertyOption[A <: scala.reflect.Enum: Type](
    list: Expr[Map[String, Attribute]],
    propName: String
)(using Quotes): Expr[Option[A]] =
  val typeName = Type.of[A].toString()
  '{
    val value: Option[Attribute] = $list.get(${ Expr(propName) })
    value.map {
      case prop @ IntegerAttr(IntData(i), _) => $enumFromOrdinalFunc(i.toInt)
      case _                                 =>
        throw new IllegalArgumentException(
          s"Type mismatch for enum property \"${${ Expr(propName) }}\": " +
            s"expected IntegerAttr, but found ${value.getClass}"
        )
    }
  }

def enumFromProperty[A <: scala.reflect.Enum: Type](
    list: Expr[Map[String, Attribute]],
    propName: String
)(using Quotes): Expr[A] =
  val typeName = Type.of[A].toString()
  '{
    val value: Option[Attribute] = $list.get(${ Expr(propName) })
    value match
      case None =>
        throw new IllegalArgumentException(
          s"Missing required property \"${${ Expr(propName) }}\" of type ${${
              Expr(typeName)
            }}"
        )
      case Some(prop: A) => prop
      case Some(_)       =>
        throw new IllegalArgumentException(
          s"Type mismatch for property \"${${ Expr(propName) }}\": " +
            s"expected ${${ Expr(typeName) }}, " +
            s"but found ${value.getClass.getSimpleName}"
        )
  }

def enumFromOrdinalFunc[E <: scala.reflect.Enum: Type](using
    Quotes
): Expr[Int => E] =
  import quotes.reflect.*

  val tpe = TypeRepr.of[E]
  val symbol = tpe.typeSymbol

  val companion = symbol.companionModule

  val fromOrdSym = companion.methodMember("fromOrdinal").head

  val companionRef = Ref(companion)
  '{ (x: Int) =>
    ${ Select(companionRef, fromOrdSym).appliedTo('{ x }.asTerm).asExprOf[E] }
  }
