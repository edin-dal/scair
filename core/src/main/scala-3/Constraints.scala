package scair.core.constraints

import scair.ir._

trait Constraint

infix type !>[A <: Attribute, C <: Constraint] = A

