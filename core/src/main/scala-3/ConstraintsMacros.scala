package scair.core.constraints

import scala.quoted._
import scair.ir._

def eqAttrImpl[To <: Attribute: Type](using Quotes) = {
  import quotes.reflect._
  println(s"owner: ${Symbol.spliceOwner}:${Symbol.spliceOwner.pos.map(p =>
      p.sourceFile.toString + ":" + p.startLine.toString + ":" + p.endLine.toString
    )}")
  val t = TypeRepr.of[To].simplified
  val ref = t match
    case tr: TermRef => Ref(tr.termSymbol).asExprOf[To]
    case _           =>
      report.errorAndAbort(
        s"got ${t.show}:\n${t}"
      )
  ref
}
