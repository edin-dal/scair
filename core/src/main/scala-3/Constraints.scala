package scair.core.constraints

import scair.ir.*
import scair.utils.R

import scala.compiletime.constValue
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

class ConstraintContext():

  val var_constraints: DictType[String, Attribute] =
    DictType.empty[String, Attribute]

trait ConstraintImpl[c <: Constraint]:

  def verify(attr: Attribute)(using
      ctx: ConstraintContext
  ): R[Unit]

infix type !>[A <: Attribute, C <: Constraint] = A

/*≡==--==≡≡≡≡≡==--=≡≡*\
||    CONSTRAINTS    ||
\*≡==---==≡≡≡==---==≡*/

trait EqAttr[To <: Attribute] extends Constraint

trait Var[To <: String] extends Constraint

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||    CONSTRAINTIMPL GIVENS    ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

inline def eqAttr[To <: Attribute]: To =
  ${ eqAttrImpl[To] }

class ConstraintImplEqAttr[To <: Attribute](ref: To)
    extends ConstraintImpl[EqAttr[To]]:

  override def verify(attr: Attribute)(using
      ctx: ConstraintContext
  ): R[Unit] =
    if attr == ref then Right(())
    else Left(s"Expected ${ref}, got ${attr}")

inline given [To <: Attribute] => ConstraintImpl[EqAttr[To]] =
  val ref = eqAttr[To]
  new ConstraintImplEqAttr(ref)

class ConstraintImplVar[To <: String](name: To) extends ConstraintImpl[Var[To]]:

  override def verify(attr: Attribute)(using
      ctx: ConstraintContext
  ): R[Unit] =
    if ctx.var_constraints.contains(name) then
      if ctx.var_constraints.apply(name) != attr then
        Left(s"Expected ${ctx.var_constraints.apply(name)}, got ${attr}")
      else Right(())
    else
      ctx.var_constraints += ((name, attr))
      Right(())

inline given [To <: String] => ConstraintImpl[Var[To]] =
  val name = constValue[To]
  new ConstraintImplVar(name)
