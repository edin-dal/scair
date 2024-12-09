package scair.scairdl.constraints

import scair.exceptions.VerifyException
import scair.ir.*

import scala.reflect.ClassTag
import scala.util.Failure
import scala.util.Success
import scala.util.Try

// USEFUL COMMANDS

def check_same_class[T <: Attribute: ClassTag](that_attr: Attribute): Boolean =
  that_attr match {
    case _: T => true
    case _    => false
  }

def check_equal[T <: Attribute: ClassTag](that_attr: Attribute): Boolean =
  that_attr match {
    case _: T => true
    case _    => false
  }

// CONSTRAINTS

class ConstraintContext() {
  val var_constraints: DictType[String, Attribute] =
    DictType.empty[String, Attribute]
}

abstract class IRDLConstraint {

  def verify(that_attr: Attribute, constraint_ctx: ConstraintContext): Unit

  def verify(
      those_attrs: Seq[Attribute],
      constraint_ctx: ConstraintContext
  ): Unit =
    for (attr <- those_attrs) verify(attr, constraint_ctx)

  def &&(that: IRDLConstraint): IRDLConstraint = AllOf(Seq(this, that))

  def ||(that: IRDLConstraint): IRDLConstraint = AnyOf(Seq(this, that))
}

object AnyAttr extends IRDLConstraint {

  override def verify(
      that_attr: Attribute,
      constraint_ctx: ConstraintContext
  ): Unit = {}

  override def toString(): String = "AnyAttr"
}

case class EqualAttr(val this_attr: Attribute) extends IRDLConstraint {

  override def verify(
      that_attr: Attribute,
      constraint_ctx: ConstraintContext
  ): Unit = {

    if (this_attr != that_attr) {
      val errstr =
        s"${that_attr.name} does not equal ${this_attr.name}:\n" +
          s"Got ${that_attr.custom_print}, expected ${this_attr.custom_print}"
      throw new VerifyException(errstr)
    }
  }

  override def toString = this_attr.toString
}

given attr2constraint: Conversion[Attribute, IRDLConstraint] with {
  def apply(attr: Attribute): IRDLConstraint = EqualAttr(attr)
}

case class BaseAttr[T <: Attribute: ClassTag]() extends IRDLConstraint {

  override def verify(
      that_attr: Attribute,
      constraint_ctx: ConstraintContext
  ): Unit = {

    that_attr match {
      case _: T =>
      case _ =>
        val className = implicitly[ClassTag[T]].runtimeClass.getName
        val errstr =
          s"${that_attr.name}'s class does not equal ${className}\n"
        throw new VerifyException(errstr)
    }
  }
  override def toString =
    s"BaseAttr[${implicitly[ClassTag[T]].runtimeClass.getName}]()"
}

case class AnyOf(constraints_in: Seq[IRDLConstraint]) extends IRDLConstraint {

  val constraints: Seq[IRDLConstraint] = constraints_in.flatMap(_ match {
    case x: AnyOf => x.constraints
    case y        => Seq(y)
  })

  override def verify(
      that_attr: Attribute,
      constraint_ctx: ConstraintContext
  ): Unit = {

    val that_attr_class = that_attr.getClass
    if (
      !constraints.exists(entry => {
        Try {
          entry.verify(that_attr, constraint_ctx)
          true
        } match {
          case Success(i) => true
          case Failure(s) => false
        }
      })
    ) {
      val errstr =
        s"${that_attr.name} does not match $toString\n"
      throw new VerifyException(errstr)
    }
  }

  override def toString = constraints.mkString(" || ")
}

case class AllOf(constraints_in: Seq[IRDLConstraint]) extends IRDLConstraint {

  val constraints: Seq[IRDLConstraint] = constraints_in.flatMap(_ match {
    case x: AllOf => x.constraints
    case y        => Seq(y)
  })

  override def verify(
      that_attr: Attribute,
      constraint_ctx: ConstraintContext
  ): Unit =
    for (c <- constraints) c.verify(that_attr, constraint_ctx)

  override def toString = constraints.mkString(" && ")
}

case class ParametrizedAttrConstraint[T <: Attribute: ClassTag](
    val constraints: Seq[IRDLConstraint]
) extends IRDLConstraint {

  override def verify(
      that_attr: Attribute,
      constraint_ctx: ConstraintContext
  ): Unit =
    check_same_class(that_attr) match {
      case true =>
        that_attr match {
          case x: ParametrizedAttribute =>
            if (!(x.parameters.length == constraints.length)) {
              throw new VerifyException(
                s"Expected ${constraints.length} parameters, got ${x.parameters.length}\n"
              )
            }
            for ((p, c) <- x.parameters zip constraints)
              p match {
                case p: Attribute      => c.verify(p, constraint_ctx)
                case p: Seq[Attribute] => c.verify(p, constraint_ctx)
              }

          case _ =>
            throw new VerifyException(
              "Attribute being verified must be of type ParametrizedAttribute.\n"
            )
        }
      case false =>
        val className = implicitly[ClassTag[T]].runtimeClass.getName
        val errstr =
          s"${that_attr.name}'s class does not equal ${className}.\n"
        throw new VerifyException(errstr)
    }

  override def toString =
    s"ParametrizedAttrConstraint[${implicitly[ClassTag[T]].runtimeClass.getName}](${constraints})"
}

case class VarConstraint(val name: String, val constraint: IRDLConstraint)
    extends IRDLConstraint {

  override def verify(
      that_attr: Attribute,
      constraint_ctx: ConstraintContext
  ): Unit = {

    val var_consts = constraint_ctx.var_constraints

    var_consts.contains(name) match {
      case true =>
        if (var_consts.apply(name) != that_attr) {
          throw new VerifyException("oh mah gawd")
        }
      case false =>
        constraint.verify(that_attr, constraint_ctx)
        var_consts += ((name, that_attr))
    }
  }
}
