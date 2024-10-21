package scair.dialects.irdl

import scala.reflect.ClassTag

import scair.{Attribute, Operation, Value, ParametrizedAttribute, DictType}

type Operand[T <: Attribute] = Value[T]
type OpResult[T <: Attribute] = Value[T]

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
}

object AnyAttr extends IRDLConstraint {

  override def verify(
      that_attr: Attribute,
      constraint_ctx: ConstraintContext
  ): Unit = {}

  override def toString = s"AnyAttr"
}

class EqualAttr(val this_attr: Attribute) extends IRDLConstraint {

  override def verify(
      that_attr: Attribute,
      constraint_ctx: ConstraintContext
  ): Unit = {

    if (!(this_attr same_as that_attr)) {
      val errstr =
        s"${that_attr.name} does not equal ${this_attr.name}:\n" +
          this_attr.toString + " and " + that_attr.toString
      throw new Exception(errstr)
    }
  }

  override def toString = s"EqualAttr(${this_attr})"
}

class BaseAttr[T <: Attribute: ClassTag]() extends IRDLConstraint {

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
        throw new Exception(errstr)
    }
  }

  override def toString =
    s"BaseAttr[${implicitly[ClassTag[T]].runtimeClass.getName}]"
}

class AnyOf(val these_attrs: Seq[Attribute | IRDLConstraint])
    extends IRDLConstraint {

  override def verify(
      that_attr: Attribute,
      constraint_ctx: ConstraintContext
  ): Unit = {

    val that_attr_class = that_attr.getClass
    if (
      !these_attrs.exists(entry =>
        entry match {
          case attr: Attribute => that_attr_class == attr.getClass
          case constr: IRDLConstraint =>
            try {
              constr.verify(that_attr, constraint_ctx)
              true
            } catch { _ => false }
        }
      )
    ) {
      val errstr =
        s"${that_attr.name} does not match any of ${these_attrs}\n"
      throw new Exception(errstr)
    }
  }

  override def toString = s"AnyOf(${these_attrs})"
}

class ParametricAttr[T <: Attribute: ClassTag](
    val params: Seq[Attribute]
) extends IRDLConstraint {

  override def verify(
      that_attr: Attribute,
      constraint_ctx: ConstraintContext
  ): Unit =
    check_same_class(that_attr) match {
      case true =>
        that_attr match {
          case x: ParametrizedAttribute =>
            if (
              !(x.parameters.length == params.length &&
                (for ((i, j) <- x.parameters zip params)
                  yield i same_as j).foldLeft(true)((i, j) => i && j))
            ) {
              throw new Exception(
                s"Parameters of ${that_attr.name} do not match the constrained" +
                  s" parameters ${params}.\n"
              )
            }
          case _ =>
            throw new Exception(
              "Attribute being verified must be of type ParametrizedAttribute.\n"
            )
        }
      case false =>
        val className = implicitly[ClassTag[T]].runtimeClass.getName
        val errstr =
          s"${that_attr.name}'s class does not equal ${className}.\n"
        throw new Exception(errstr)
    }

  override def toString =
    s"ParametricAttr[${implicitly[ClassTag[T]].runtimeClass.getName}](${params})"
}

class VarConstraint(val name: String, val constraint: IRDLConstraint)
    extends IRDLConstraint {

  override def verify(
      that_attr: Attribute,
      constraint_ctx: ConstraintContext
  ): Unit = {

    val var_consts = constraint_ctx.var_constraints

    var_consts.contains(name) match {
      case true =>
        if (!(var_consts.apply(name) same_as that_attr)) {
          throw new Exception("oh mah gawd")
        }
      case false =>
        constraint.verify(that_attr, constraint_ctx)
        var_consts += ((name, that_attr))
    }
  }
  override def toString = s"VarConstraint(${name}, ${constraint})"
}

// CONSTRAINTS

// class Operand(name)
// class Type(name, Attribute)
// class Result
// class DictAttr

// M("(" + Operand("hello") + ")" + "this is an custom syntax" + ":" + Type("hello", StringAttribute.parse))

// %5 = "custom.op"(%3)this is an custom syntax:"this"
