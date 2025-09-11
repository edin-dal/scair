package scair.core.constraints

import scair.ir.*

import scala.quoted.*

def eqAttrImpl[To <: Attribute: Type](using Quotes) = {
  import quotes.reflect._
  val t = TypeRepr.of[To].simplified
  val ref = t match
    case tr: TermRef => Ref(tr.termSymbol).asExprOf[To]
    case _           =>
      report.errorAndAbort(
        s"got ${t.show}:\n${t}"
      )
  ref
}
