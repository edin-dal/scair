package scair.core.constraints

import scair.ir.*
import scair.utils.OK

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

  val varConstraints: DictType[String, Attribute] =
    DictType.empty[String, Attribute]

trait ConstraintImpl[c <: Constraint]:

  def verify(attr: Attribute)(using
      ctx: ConstraintContext
  ): OK[Unit]

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
  ): OK[Unit] =
    if attr == ref then Right(())
    else Left(s"Expected $ref, got $attr")

inline given [To <: Attribute] => ConstraintImpl[EqAttr[To]] =
  val ref = eqAttr[To]
  new ConstraintImplEqAttr(ref)

class ConstraintImplVar[To <: String](name: To) extends ConstraintImpl[Var[To]]:

  override def verify(attr: Attribute)(using
      ctx: ConstraintContext
  ): OK[Unit] =
    if ctx.varConstraints.contains(name) then
      if ctx.varConstraints.apply(name) != attr then
        Left(s"Expected ${ctx.varConstraints.apply(name)}, got $attr")
      else Right(())
    else
      ctx.varConstraints += ((name, attr))
      Right(())

inline given [To <: String] => ConstraintImpl[Var[To]] =
  val name = constValue[To]
  new ConstraintImplVar(name)
