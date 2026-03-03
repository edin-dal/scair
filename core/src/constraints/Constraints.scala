package scair.constraints

import scair.ir.*
import scair.utils.*

import scala.quoted.*

def loadConstraintCompanion[C <: Constraint: Type](using
    Quotes
): Option[scair.constraints.ConstraintCompanion[C]] =
  import quotes.reflect.*
  val companionSymbol = TypeRepr.of[C].typeSymbol.companionModule
  if !(companionSymbol.termRef <:<
      TypeRepr
        .of[
          scair.constraints.ConstraintCompanion[?]
        ])
  then
    report.errorAndAbort(
      s"Constraint ${Type.show[C]} does not have a valid companion object extending ConstraintCompanion."
    )
  val fullName = companionSymbol.fullName
  val cls = Thread.currentThread().getContextClassLoader
    .loadClass(fullName + "$")
  val instance = cls.getField("MODULE$").get(null)
  Some(instance.asInstanceOf[scair.constraints.ConstraintCompanion[C]])

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
  ): Expr[OK[Unit]] =
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
    '{
      val _r: OK[Unit] =
        if $attr != $refExpr then Err(s"Expected ${$refExpr}, got ${$attr}")
        else OK(())
      _r
    }

trait Var[To <: String] extends Constraint

object Var extends ConstraintCompanion[Var[String]]:

  def macroVerify(using
      Quotes
  )(
      constraintType: Type[Var[String]],
      attr: Expr[Attribute],
      ctx: MacroConstraintContext,
  ): Expr[OK[Unit]] =
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
        '{
          val _r: OK[Unit] =
            if $attr != $bound then Err(s"Expected ${$bound}, got ${$attr}")
            else OK(())
          _r
        }
      case None =>
        ctx.bindings(name) = attr
        '{ OK(()) }

trait AllOf[A <: Constraint, B <: Constraint] extends Constraint

object AllOf extends ConstraintCompanion[AllOf[Constraint, Constraint]]:

  def macroVerify(using
      Quotes
  )(
      constraintType: Type[AllOf[Constraint, Constraint]],
      attr: Expr[Attribute],
      ctx: MacroConstraintContext,
  ): Expr[OK[Unit]] =
    import quotes.reflect.*
    constraintType match
      case '[AllOf[a, b]] =>
        val cA = loadConstraintCompanion[a].getOrElse(
          report
            .errorAndAbort(s"No companion found for constraint ${Type.show[a]}")
        )
        val cB = loadConstraintCompanion[b].getOrElse(
          report
            .errorAndAbort(s"No companion found for constraint ${Type.show[b]}")
        )
        val rA = cA.macroVerify(constraintType = Type.of[a], attr, ctx)
        val rB = cB.macroVerify(constraintType = Type.of[b], attr, ctx)
        '{
          $rA match
            case OK(())   => $rB
            case Err(msg) => Err(msg)
        }
