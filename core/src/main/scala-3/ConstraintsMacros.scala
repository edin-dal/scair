package scair.core.constraints

import scala.quoted._
import scair.ir._

def eqAttrImpl[To <: Attribute: Type](using Quotes): Expr[ConstraintImpl[EqAttr[To]]] = {
  import quotes.reflect._
  val ref = TypeRepr.of[To] match
    case tr : TermRef => Ref(tr.termSymbol).asExprOf[To]
    case _ => report.errorAndAbort(
        s"TermRef only"
    )
  '{
    new ConstraintImpl {
        def verify(attr: Attribute): Either[String, Attribute] =
            attr match {
                case a: To if a == ${ref} => Right(${ref})
                case _ => Left(s"Expected ${${ref}}, got ${attr}")
            }
    }
  }

}