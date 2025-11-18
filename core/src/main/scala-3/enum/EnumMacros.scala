package scair.enums.macros

import scair.dialects.builtin.IntData
import scair.dialects.builtin.IntegerAttr
import scair.ir.Attribute

import scala.quoted.*

// ███████╗ ███╗░░██╗ ██╗░░░██╗ ███╗░░░███╗
// ██╔════╝ ████╗░██║ ██║░░░██║ ████╗░████║
// █████╗░░ ██╔██╗██║ ██║░░░██║ ██╔████╔██║
// ██╔══╝░░ ██║╚████║ ██║░░░██║ ██║╚██╔╝██║
// ███████╗ ██║░╚███║ ╚██████╔╝ ██║░╚═╝░██║
// ╚══════╝ ╚═╝░░╚══╝ ░╚═════╝░ ╚═╝░░░░░╚═╝
//
// ███╗░░░███╗ ░█████╗░ ░█████╗░ ██████╗░ ░█████╗░ ░██████╗
// ████╗░████║ ██╔══██╗ ██╔══██╗ ██╔══██╗ ██╔══██╗ ██╔════╝
// ██╔████╔██║ ███████║ ██║░░╚═╝ ██████╔╝ ██║░░██║ ╚█████╗░
// ██║╚██╔╝██║ ██╔══██║ ██║░░██╗ ██╔══██╗ ██║░░██║ ░╚═══██╗
// ██║░╚═╝░██║ ██║░░██║ ╚█████╔╝ ██║░░██║ ╚█████╔╝ ██████╔╝
// ╚═╝░░░░░╚═╝ ╚═╝░░╚═╝ ░╚════╝░ ╚═╝░░╚═╝ ░╚════╝░ ╚═════╝░

/** Generates code to convert an IntegerAttr value to an Optional enum property
  * argument.
  *
  * @param list
  * @param propName
  * @return
  */
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

/** Generates code to convert an IntegerAttr value to a required enum property
  * argument.
  *
  * @param list
  * @param propName
  * @return
  */
def enumFromProperty[A <: scala.reflect.Enum: Type](
    list: Expr[Map[String, Attribute]],
    propName: String
)(using Quotes): Expr[A] =
  import quotes.reflect.*
  val typeName = TypeRepr.of[A].show
  '{
    val value: Option[Attribute] = $list.get(${ Expr(propName) })
    value match
      case None =>
        throw new IllegalArgumentException(
          s"Missing required property \"${${ Expr(propName) }}\" of type ${${
              Expr(typeName)
            }}"
        )
      case Some(prop @ IntegerAttr(IntData(i), _)) =>
        $enumFromOrdinalFunc(i.toInt)
      case Some(_) =>
        throw new IllegalArgumentException(
          s"Type mismatch for property \"${${ Expr(propName) }}\": " +
            s"expected ${${ Expr(typeName) }}, " +
            s"but found ${value.getClass.getSimpleName}"
        )
  }

/** Retrieves a given Enum case's companion object and returns an expression of
  * ``fromOrdinal`` method as a function.
  *
  * @return
  *   Expr[Int => E]
  */
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
