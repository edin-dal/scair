package scair.core.constraints

import scair.ir.*

import scala.quoted.*

trait Constraint

infix type !>[A <: Attribute, C <: Constraint] = A

trait EqAttr[To <: Attribute] extends Constraint

trait ConstraintImpl[c <: Constraint] {

  def verify(attr: Attribute): Either[String, Unit]

}

inline def eqAttr[To <: Attribute]: To =
  ${ eqAttrImpl[To] }

inline given [To <: Attribute] => ConstraintImpl[EqAttr[To]] =
  val ref = eqAttr[To]
  new ConstraintImpl {
    override def verify(attr: Attribute): Either[String, Unit] =
      if (attr == ref) Right(())
      else Left(s"Expected ${ref}, got ${attr}")
  }
