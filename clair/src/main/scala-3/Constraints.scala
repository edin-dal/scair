package scair.clair.constraint

import scair.ir.*
import scala.quoted._
import scala.annotation.Annotation
import scala.annotation.StaticAnnotation
import scair.dialects.builtin.*

trait Constraint {}

infix opaque type Constrained[A <: Attribute, C <: Constraint] = A
trait ConstrainedCompanion[A <: Attribute, C <: Constraint] {
    def constrain(a: A) : Either[String, A Constrained C] 
}

trait EqualAttr[To <: Attribute] extends Constraint

given [A <: Attribute, To <: Attribute] => (vto : ValueOf[To]) => ConstrainedCompanion[A,EqualAttr[To]] {
    def constrain(a: A) = {
        if a == vto.value then Right(Constrained[A, EqualAttr[To]](a)) else Left(s"Expected ${a} to be equal to ${vto.value}")
    }
}

val i16 = IntegerType(IntData(16), Signless)

object DaMain {
    def main(args: Array[String]): Unit = {
        val thingy = summon[ConstrainedCompanion[Attribute, EqualAttr[i16.type]]]
        val ai = IntegerType(IntData(16), Signless)
        val a = thingy.constrain(ai)
        println(a)
        val bi = IntegerType(IntData(32), Signless)
        val b = thingy.constrain(bi)
        println(b)
    }
}