package scair.core.constraints

import scair.ir._

trait Constraint

infix type !>[A <: Attribute, C <: Constraint] = A

trait EqAttr[To <: Attribute] extends Constraint

trait ConstraintImpl[c <: Constraint]

inline def eqAttr[To <: Attribute]: ConstraintImpl[EqAttr[To]] =
  ${ eqAttrImpl[To] }

inline given [To <: Attribute] => ConstraintImpl[EqAttr[To]] = eqAttr[To]

