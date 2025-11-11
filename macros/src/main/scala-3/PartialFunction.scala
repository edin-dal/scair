package scair.macros

import scala.quoted.*

def popPFImpl(using Quotes) =
  import quotes.reflect.*

  val methodTpe = MethodType(MethodTypeKind.Plain)(List("whatever"))(
    _ => List(TypeRepr.of[Any]),
    _ => TypeRepr.of[String]
  )

  val sym = Symbol.newMethod(
    Symbol.spliceOwner,
    "wow",
    methodTpe
  )

  val xSym = Symbol.newVal(
    Symbol.spliceOwner,
    "x",
    TypeRepr.of[Any],
    Flags.EmptyFlags,
    Symbol.noSymbol
  )

  val defdef = DefDef(
    sym,
    p =>
      Some(
        Match(
          Ref(p.head.head.symbol),
          List(
            CaseDef(Wildcard(), None, Literal(StringConstant("Hello world!")))
          )
        )
      )
  )

  Block(List(defdef), Ref(defdef.symbol).etaExpand(Symbol.spliceOwner))
    .asExprOf[Any => String]

inline def popPF = ${ popPFImpl }

def mapInlineApply[A: Type, B: Type](using
    Quotes
)(a: quotes.reflect.Term, b: Expr[A => B])(using Quotes) =
  import quotes.reflect.*
  '{ $b(${ a.asExprOf[A] }) }.asTerm

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

    // Was unsure abut going inside Closure?
    case _ =>
      mapInlineApply(a, b)

def mapInlineImpl[A: Type, B: Type](a: Expr[A], b: Expr[A => B])(using
    Quotes
): Expr[B] =
  import quotes.reflect.*

  mapInlineTerm(a.asTerm, b).asExprOf[B]

inline def mapInline[A, B](inline a: A, inline b: A => B): B =
  ${ mapInlineImpl('a, 'b) }

inline def code(inline expr: Any): String =
  ${ codeImpl('expr) }

def codeImpl(expr: Expr[Any])(using Quotes): Expr[String] =
  Expr(expr.show)

type ValueT = Int

opaque type Value[Name <: String] <: ValueT = ValueT

import scala.compiletime._

final inline def summonOption[T] =
  summonFrom {
    case t: T => Some(t)
    case _    => None
  }


// inline def varOrMacro[Name <: String](inline name: String, inline value: ValueT, inline next: Unit) =
//   ${ varOrImpl[Name]('name, 'value, 'next) }

// inline def varOr[Name <: String](inline value: ValueT, inline next: => Unit) =
//   val name = constValue[Name]
//   varOrMacro(name, value, next)

inline def varOfChain(a : (Quotes) => Expr[Unit] => Expr[Unit], b : (Quotes) => Expr[Unit] => Expr[Unit], s: Expr[Unit]) : (Quotes) => Expr[Unit] =
  ${ varOfChainImpl('a, 'b, 's) }

inline def stuff: Unit = ${ stuffImpl }

object Stuf:
  def stuf = stuff


// def stuffImpl(using Quotes): Expr[Unit] =
//   import quotes.reflect.*

//   val aGiven = '{ "hello from macroA" }

//   // manually call macroBImpl inside this macro
//   val bExpansion = '{macroB}
//   '{
//     given String = $aGiven
//     $bExpansion
//   }

inline def macroB = ${ macroBImpl}

inline def summonString =
  summonFrom[String] {
      case s: String => s
      case _         => "no string found"
    }

def macroBImpl(using Quotes): Expr[Unit] =
  '{
    println("macroB got: " + summonString)
  }
  

// inline def combinePF[A, B](inline pfs: PartialFunction[A, B]*) =
//   ${ combinePFImpl('pfs) }

// private def combinePFImpl[A: Type, B: Type](pfs: Expr[Seq[PartialFunction[A, B]]])(using Quotes) =
//   import quotes.reflect.*

//   val head = pfs match
//     case Varargs(Seq(head, _*)) => head

//   head.asTerm match
//     case Block(List(d@DefDef(name, paramss, tpt, Some(Match(selector, cases)))), e) =>
//       report.info(s"head: ${head.show}\nparamss: ${paramss}\ntpt: ${tpt.show}\nselector: ${selector.show}\ncases: ${cases.map(_.show).mkString}\nexpr: ${e.show}\nd: ${d.show}")

//   '{()}

// Build: new PartialFunction[A, B] { def apply(x: A) = x match { ... }; def isDefinedAt(x: A) = ... }
// val aSym = Symbol.newVal(Symbol.spliceOwner, "x", TypeRepr.of[A], Flags.EmptyFlags, Symbol.noSymbol)
// val aRef = Ref(aSym)

// val matchExpr = Match(aRef, allCases)

// // Build an efficient isDefinedAt by reusing pattern structure
// val isDefinedAtCases = allCases.map {
//   case CaseDef(pat, guard, _) => CaseDef(pat, guard, Literal(BooleanConstant(true)))
// }
// val isDefinedAtMatch = Match(aRef, isDefinedAtCases :+ CaseDef(Wildcard(), None, Literal(BooleanConstant(false))))

// val pfExpr =
//   '{
//     new PartialFunction[A, B] {
//       def apply(x: A): B = ${ matchExpr.asExprOf[B] }
//       def isDefinedAt(x: A): Boolean = ${ isDefinedAtMatch.asExprOf[Boolean] }
//     }
//   }

// pfExpr

// reduced

// private def reducePFs[A: Type, B: Type](pfs: Seq[Expr[PartialFunction[A, B]]])(using Quotes): Expr[PartialFunction[A, B]] =
//   import quotes.reflect.*
//   pfs match
//     case Seq(pf) => pf
//     case head1 :: head2 :: tail =>
//       val head = (head1.asTerm) match
//         case Block(List(DefDef(name, paramss, tpt, Some(Match(selector, cases)))), e) =>
//           head2.asTerm match
//             case Block(List(DefDef(_, _, _, Some(Match(_, cases2)))), _) =>
//               Block(List(DefDef(Symbol.newMethod(Symbol.spliceOwner.asInstanceOf[Symbol], name+"$"), tpt.tpe), _ => Some(Match(selector, cases ++ cases2))), e).asExprOf[PartialFunction[A, B]]

//       reducePFs(head +: tail)

// private def extractCases[A: Type, B: Type](pf: Expr[PartialFunction[A, B]])(using Quotes): Seq[quotes.reflect.CaseDef] =
//   import quotes.reflect.*
//   pf.asTerm.underlyingArgument match
//     // PartialFunction literals compile to an anonymous class with an apply + isDefinedAt derived from a Match
//     case b @ Block(List(DefDef(_, _, _, Some(Match(_, cases)))), _) =>
//       println(b.show)
//       cases
//     case other =>
//       report.error(s"Unsupported partial function form: ${other}")
//       Nil
