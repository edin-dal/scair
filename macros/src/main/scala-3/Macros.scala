package scair.macros

import scala.quoted.*

extension (expr: Expr[?])

  def member[T: Type](name: String)(using Quotes): Expr[T] =
    import quotes.reflect.*
    Select.unique(expr.asTerm, name).asExprOf[T]
