package scair.core.constraints

import scair.ir.*

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

trait Constraint[+A <: Attribute]

type BaseOf[C <: Constraint[?]] = C match 
  case Constraint[of] => of

class ConstraintContext() {

  val var_constraints: DictType[String, Attribute] =
    DictType.empty[String, Attribute]

}

trait ConstraintImpl[c <: Constraint[?]] {

  def verify(attr: Attribute)(using
      ctx: ConstraintContext
  ): Either[String, Unit]

}

infix type !>[A <: Attribute, C <: Constraint[A]] = A

// type @>[C <: Constraint[?]] = C match 
//   case Constraint[of] => of !> C

/*≡==--==≡≡≡≡≡==--=≡≡*\
||    CONSTRAINTS    ||
\*≡==---==≡≡≡==---==≡*/

trait EqAttr[To <: Attribute] extends Constraint[To]

trait Var[To <: String] extends Constraint[Attribute]

trait BaseAttr[Base <: Attribute] extends  Constraint[Base]
trait And[C1 <: Constraint[?], C2 <: Constraint[?]]
    extends Constraint[BaseOf[C1] & BaseOf[C2]]

type wrapped[A] = A match
  case Constraint[of] => A & Constraint[of]
  case Attribute => BaseAttr[A & Attribute]

infix type &&[C1, C2] = And[wrapped[C1], wrapped[C2]]
/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||    CONSTRAINTIMPL GIVENS    ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

inline def eqAttr[To <: Attribute]: To =
  ${ eqAttrImpl[To] }

inline given [To <: Attribute] => ConstraintImpl[EqAttr[To]] =
  val ref = eqAttr[To]
  new ConstraintImpl {
    override def verify(attr: Attribute)(using
        ctx: ConstraintContext
    ): Either[String, Unit] =
      if (attr == ref) Right(())
      else Left(s"Expected ${ref}, got ${attr}")
  }

inline given [To <: String] => ConstraintImpl[Var[To]] =
  val name: String = constValue[To]
  new ConstraintImpl {
    override def verify(attr: Attribute)(using
        ctx: ConstraintContext
    ): Either[String, Unit] = {
      if (ctx.var_constraints.contains(name)) {
        if (ctx.var_constraints.apply(name) != attr) then
          Left(s"Expected ${ctx.var_constraints.apply(name)}, got ${attr}")
        else Right(())
      } else {
        ctx.var_constraints += ((name, attr))
        Right(())
      }
    }

  }

inline given [C1 <: Constraint[?], C2 <: Constraint[?]] => (impl1 : ConstraintImpl[C1], impl2 : ConstraintImpl[C2]) => ConstraintImpl[And[C1, C2]] =
  new ConstraintImpl {
    override def verify(attr: Attribute)(using
        ctx: ConstraintContext
    ): Either[String, Unit] = {
      impl1.verify(attr).flatMap(_ => impl2.verify(attr))
    }
  }


inline given [Base <: Attribute] => ConstraintImpl[BaseAttr[Base]] =
  new ConstraintImpl {
    override def verify(attr: Attribute)(using
        ctx: ConstraintContext
    ): Either[String, Unit] = {
      attr match
        case _: Base => Right(())
        case _ => Left(s"Expected attribute of type BLABLABLA, got ${attr}")

      
    }
  }