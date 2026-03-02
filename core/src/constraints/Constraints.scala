package scair.constraints

import scair.ir.*
import scair.utils.*

import scala.quoted.*

// ░█████╗░ ░█████╗░ ███╗░░██╗ ░██████╗ ████████╗ ██████╗░ ░█████╗░ ██╗ ███╗░░██╗ ████████╗ ░██████╗
// ██╔══██╗ ██╔══██╗ ████╗░██║ ██╔════╝ ╚══██╔══╝ ██╔══██╗ ██╔══██╗ ██║ ████╗░██║ ╚══██╔══╝ ██╔════╝
// ██║░░╚═╝ ██║░░██║ ██╔██╗██║ ╚█████╗░ ░░░██║░░░ ██████╔╝ ███████║ ██║ ██╔██╗██║ ░░░██║░░░ ╚█████╗░
// ██║░░██╗ ██║░░██║ ██║╚████║ ░╚═══██╗ ░░░██║░░░ ██╔══██╗ ██╔══██║ ██║ ██║╚████║ ░░░██║░░░ ░╚═══██╗
// ╚█████╔╝ ╚█████╔╝ ██║░╚███║ ██████╔╝ ░░░██║░░░ ██║░░██║ ██║░░██║ ██║ ██║░╚███║ ░░░██║░░░ ██████╔╝
// ░╚════╝░ ░╚════╝░ ╚═╝░░╚══╝ ╚═════╝░ ░░░╚═╝░░░ ╚═╝░░╚═╝ ╚═╝░░╚═╝ ╚═╝ ╚═╝░░╚══╝ ░░░╚═╝░░░ ╚═════╝░

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||     TOP LEVEL TRAITS     ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡==---==≡*/

trait Constraint

infix type !>[A <: Attribute, C <: Constraint] = A

/*≡==--==≡≡≡≡≡==--=≡≡*\
||    CONSTRAINTS    ||
\*≡==---==≡≡≡==---==≡*/

trait EqAttr[To <: Attribute] extends Constraint

object EqAttr extends ConstraintCompanion:

  def macroVerify(using
      Quotes
  )(
      constraintType: Type[?],
      attr: Expr[Attribute],
      ctx: MacroConstraintContext,
  ): (Expr[OK[Unit]], MacroConstraintContext) =
    import quotes.reflect.*
    val refType = TypeRepr.of(using constraintType) match
      case AppliedType(_, List(r)) => r
      case other                   =>
        report
          .errorAndAbort(
            s"EqAttr requires a type parameter, got: ${other.show}"
          )
    val refExpr: Expr[Attribute] = refType.simplified match
      case tr: TermRef =>
        Ref(tr.termSymbol).asExprOf[Attribute]
      case other =>
        report
          .errorAndAbort(
            s"EqAttr reference must be a singleton value, got: ${other.show}"
          )
    val check = '{
      val _r: OK[Unit] =
        if $attr != $refExpr then Err(s"Expected ${$refExpr}, got ${$attr}")
        else OK(())
      _r
    }
    (check, ctx)

trait Var[To <: String] extends Constraint

object Var extends ConstraintCompanion:

  def macroVerify(using
      Quotes
  )(
      constraintType: Type[?],
      attr: Expr[Attribute],
      ctx: MacroConstraintContext,
  ): (Expr[OK[Unit]], MacroConstraintContext) =
    import quotes.reflect.*
    val nameType = TypeRepr.of(using constraintType) match
      case AppliedType(_, List(n)) => n
      case other                   =>
        report
          .errorAndAbort(
            s"Var requires a type parameter, got: ${other.show}"
          )
    val name = nameType.asType match
      case '[type n <: String; `n`] =>
        Type.valueOfConstant[n]
          .getOrElse(
            report
              .errorAndAbort(
                "Var requires a literal string type parameter"
              )
          ).asInstanceOf[String]
    val key = s"var:$name"
    ctx.bindings.get(key) match
      case Some(boundAny) =>
        val bound = boundAny.asInstanceOf[Expr[Attribute]]
        val check = '{
          val _r: OK[Unit] =
            if $attr != $bound then Err(s"Expected ${$bound}, got ${$attr}")
            else OK(())
          _r
        }
        (check, ctx)
      case None =>
        ('{ OK(()) }, ctx.updated(key, attr))
