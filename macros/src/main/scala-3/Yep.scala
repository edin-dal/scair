package scair.macros

import scala.quoted.*

def stuffImpl(using Quotes): Expr[Unit] =


  // val bExpansion = varOrImpl["X"]('{"X"}, '{2}, '{()})
  // varOrImpl["X"]('{"X"}, '{1}, bExpansion)

  val b = 
    (q: Quotes) => (next: Expr[Unit]) => varOrImpl["X"]('{"X"}, '{2}, next)(using Type.of["X"], q)
  val a =
    (q: Quotes) => (next: Expr[Unit]) => varOrImpl["X"]('{"X"}, '{1}, next)(using Type.of["X"], q)
  
  val chain = varOfChain(a, b, '{()})
  val expr = chain(quotes)
  println(expr.show)
  expr

def varOrImpl[Name <: String : Type](name: Expr[String], value: Expr[ValueT], next: Expr[Unit])(using Quotes) =
  import quotes.reflect.*
  println(s"varOrImpl for ${name.valueOrAbort} / ${value.show}")
  // Expr.summon[Value[Name]] match
  //   case Some(v) =>
  //     '{
        
  //       println(s"found ${$name} = ${$v}, have ${$value}")
  //     }
  //   case None => 
  //     '{
  //       println(s"${$name} not found, have ${$value}")
  //       given Value[Name] = $value
  //       $next
  //     }
      
  // '{
  //   println(s"${$name} not found, have ${$value}")
  //   given Value[Name] = $value
  //   $next
  // }
  '{
    summonOption[Value[Name]] match
      case Some(v) =>
        println(s"found ${$name} = $v, have ${$value}")
      case None    => 
        println(s"${$name} not found, have ${$value}")
        given Value[Name] = $value.asInstanceOf[Value[Name]]
        $next
  }
