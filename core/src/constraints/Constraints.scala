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

object EqAttr extends ConstraintCompanion[EqAttr[Attribute]]:

  def macroVerify(using
      Quotes
  )(
      constraintType: Type[EqAttr[Attribute]],
      attr: Expr[Attribute],
      ctx: MacroConstraintContext,
  ): (Expr[OK[Unit]], MacroConstraintContext) =
    import quotes.reflect.*
    val refType = constraintType match
      case '[EqAttr[ref]] => TypeRepr.of[ref]

    val refExpr: Expr[Attribute] = refType.simplified match
      case tr: TermRef =>
        Ref(tr.termSymbol).asExprOf[Attribute]
      case other =>
        report
          .errorAndAbort(
            s"EqAttr reference must be a singleton value, got: ${other.show}"
          )
    (
      '{
        val _r: OK[Unit] =
          if $attr != $refExpr then Err(s"Expected ${$refExpr}, got ${$attr}")
          else OK(())
        _r
      },
      ctx,
    )

trait Var[To <: String] extends Constraint

object Var extends ConstraintCompanion[Var[String]]:

  def macroVerify(using
      Quotes
  )(
      constraintType: Type[Var[String]],
      attr: Expr[Attribute],
      ctx: MacroConstraintContext,
  ): (Expr[OK[Unit]], MacroConstraintContext) =
    import quotes.reflect.*
    val name = constraintType match
      case '[Var[name]] =>
        Type.valueOfConstant[name].getOrElse(
          report
            .errorAndAbort("Expected a constant string type parameter for Var")
        )

    ctx.bindings.get(name) match
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
        ('{ OK(()) }, ctx.updated(name, attr))
