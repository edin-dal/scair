package scair.clair.macros

import scala.quoted.*

/** Small helper to select a member of an expression.
  * @param obj
  *   The object to select the member from.
  * @param name
  *   The name of the member to select.
  */
def selectMember[T: Type](obj: Expr[?], name: String)(using
    Quotes
): Expr[T] =
  import quotes.reflect.*

  Select.unique(obj.asTerm, name).asExprOf[T]

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
