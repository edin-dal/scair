package scair.core.constraints

import scair.ir._
import scala.quoted._

trait Constraint

infix type !>[A <: Attribute, C <: Constraint] = A

trait EqAttr[To <: Attribute] extends Constraint

trait ConstraintImpl[c <: Constraint]

inline def eqAttr[To <: Attribute]: To =
  ${ eqAttrImpl[To] }

inline given [To <: Attribute] => ConstraintImpl[EqAttr[To]] = 
  val ref = eqAttr[To]
  new ConstraintImpl {
    def verify(attr: Attribute): Either[String, Attribute] =
        attr match {
            case a: To if a == ref => Right(ref)
            case _ => Left(s"Expected ${ref}, got ${attr}")
        }
  }

