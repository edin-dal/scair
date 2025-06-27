package scair.clair.constraints

import scair.ir.*
import scala.quoted.*

infix opaque type Eq[T <: Attribute, To <: T] = T

object Eq {
    def verifyMacro[T <: Attribute : Type, To <: T : Type](t: Expr[T])(using Quotes) = 
        import quotes.reflect.*
        val tree = TypeRepr.of[To]
        val term = tree match
            case r: TermRef => Ident(r)
            case _ => report.errorAndAbort(s"Expected a term designator, got ${tree}")
            
        // term.asExpr

        val expr = term.asExprOf[Attribute]
        
        '{
            $t match
            case _ if $t == $expr => $t.asInstanceOf[Eq[T, To]]
            case _ => throw new Exception(
                s"Type mismatch: Expected ${$expr.custom_print}, got ${$t.custom_print}"
            )
        }
    inline def verify[T <: Attribute, To <: T](t: T) = ${ verifyMacro[T, To]('t) }
        

}