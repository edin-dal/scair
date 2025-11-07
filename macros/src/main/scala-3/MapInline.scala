package scair.macros

import scala.quoted.*

def mapInlineApply[A: Type, B: Type](using
    Quotes
)(a: quotes.reflect.Term, b: Expr[A => B])(using Quotes) =
  import quotes.reflect.*
  Expr.betaReduce('{ $b(${ a.asExprOf[A] }) }).asTerm

def mapInlineCase[A: Type, B: Type](using
    Quotes
)(a: quotes.reflect.CaseDef, b: Expr[A => B]) =
  import quotes.reflect.*
  a match
    case CaseDef(pat, guard, rhs) =>
      CaseDef(pat, guard, mapInlineTerm(rhs, b))

def mapInlineTerm[A: Type, B: Type](using
    Quotes
)(a: quotes.reflect.Term, b: Expr[A => B]): quotes.reflect.Term =
  import quotes.reflect.*
  a match
    case Block(statements, term) =>
      Block(statements, mapInlineTerm(term, b))
    case Inlined(what, defs, term) =>
      Inlined(what, defs, mapInlineTerm(term, b))
    case If(cond, thent, elset) =>
      If(cond, mapInlineTerm(thent, b), mapInlineTerm(elset, b))
    case Match(selector, cases) =>
      Match(selector, cases.map(mapInlineCase(_, b)))
    // TODO: this might be a brittle case as-is
    case Typed(term, tpt) =>
      mapInlineTerm(term, b)
    case Return(term, from) =>
      Return(mapInlineTerm(term, b), from)
    case Try(body, catches, finallizer) =>
      Try(
        mapInlineTerm(body, b),
        catches.map(mapInlineCase(_, b)),
        finallizer
      )

    // Was unsure about going inside Closure?
    case _ =>
      mapInlineApply(a, b)

def mapInlineImpl[A: Type, B: Type](a: Expr[A], b: Expr[A => B])(using
    Quotes
): Expr[B] =
  import quotes.reflect.*

  Expr.betaReduce(mapInlineTerm(a.asTerm, b).asExprOf[B])

transparent inline def mapInline[A, B](inline a: A, inline b: A => B): B =
  ${ mapInlineImpl('a, 'b) }

inline def code(inline expr: Any): String =
  ${ codeImpl('expr) }

def codeImpl(expr: Expr[Any])(using Quotes): Expr[String] =
  Expr(expr.show)
