package scair.enums

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
    propName: String,
)(using Quotes): Expr[Option[A]] =
  val typeName = Type.of[A].toString()
  '{
    val value: Option[Attribute] = $list.get(${ Expr(propName) })
    value.map {
      case prop: A                           => prop
      case prop @ IntegerAttr(IntData(i), _) => $enumFromOrdinalFunc(i.toInt)
      case prop                              =>
        throw new IllegalArgumentException(
          s"Type mismatch for enum property \"${${ Expr(propName) }}\": " +
            s"expected IntegerAttr, but found ${prop.getClass}"
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
    propName: String,
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
      case Some(IntegerAttr(IntData(i), _)) =>
        $enumFromOrdinalFunc(i.toInt)
      case Some(i: A)  => i
      case Some(value) =>
        throw new IllegalArgumentException(
          s"Type mismatch for property \"${${ Expr(propName) }}\": " +
            s"expected ${${ Expr(typeName) }}, " +
            s"but found ${value.getClass()}"
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

/** Retrieves a given Enum's companion object and returns an expression of its
  * synthetic ``values`` array.
  *
  * @return
  *   Expr[Array[E]]
  */
def enumValuesExpr[E <: scala.reflect.Enum: Type](using
    Quotes
): Expr[Array[E]] =
  import quotes.reflect.*

  val symbol = TypeRepr.of[E].typeSymbol
  val companion = symbol.companionModule

  companion.methodMember("values") match
    case valuesSym :: _ =>
      Ref(companion).select(valuesSym).asExprOf[Array[E]]
    case Nil =>
      report
        .errorAndAbort(
          s"${Type.show[E]} has no `values` member on its companion object; " +
            "it must be a Scala 3 `enum` (and not one of its cases)."
        )

/** All of an Enum's cases, as an Array. */
inline def enumValues[E <: scala.reflect.Enum]: Array[E] = ${
  enumValuesExpr[E]
}
