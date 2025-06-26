package scair.scairdl.constraints

import scair.exceptions.VerifyException
import scair.ir.*

import scala.reflect.ClassTag
import scala.util.Failure
import scala.util.Success
import scala.util.Try

// ░█████╗░ ░█████╗░ ███╗░░██╗ ░██████╗ ████████╗ ██████╗░ ░█████╗░ ██╗ ███╗░░██╗ ████████╗ ░██████╗
// ██╔══██╗ ██╔══██╗ ████╗░██║ ██╔════╝ ╚══██╔══╝ ██╔══██╗ ██╔══██╗ ██║ ████╗░██║ ╚══██╔══╝ ██╔════╝
// ██║░░╚═╝ ██║░░██║ ██╔██╗██║ ╚█████╗░ ░░░██║░░░ ██████╔╝ ███████║ ██║ ██╔██╗██║ ░░░██║░░░ ╚█████╗░
// ██║░░██╗ ██║░░██║ ██║╚████║ ░╚═══██╗ ░░░██║░░░ ██╔══██╗ ██╔══██║ ██║ ██║╚████║ ░░░██║░░░ ░╚═══██╗
// ╚█████╔╝ ╚█████╔╝ ██║░╚███║ ██████╔╝ ░░░██║░░░ ██║░░██║ ██║░░██║ ██║ ██║░╚███║ ░░░██║░░░ ██████╔╝
// ░╚════╝░ ░╚════╝░ ╚═╝░░╚══╝ ╚═════╝░ ░░░╚═╝░░░ ╚═╝░░╚═╝ ╚═╝░░╚═╝ ╚═╝ ╚═╝░░╚══╝ ░░░╚═╝░░░ ╚═════╝░

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||     USEFUL COMMANDS     ||
\*≡==---==≡≡≡≡≡≡≡≡≡==---==≡*/

/** Checks if the given attribute is of the same class as the specified type
  * parameter.
  *
  * @tparam T
  *   the type parameter representing the class to check against
  * @param that_attr
  *   the attribute to check
  * @return
  *   true if the attribute is of the same class as the specified type
  *   parameter, false otherwise
  */
def check_same_class[T <: Attribute: ClassTag](that_attr: Attribute): Boolean =
  that_attr match {
    case _: T => true
    case _    => false
  }

/** Checks if the given attribute is equal to the specified type parameter.
  *
  * @tparam T
  *   the type parameter representing the class to check against
  * @param that_attr
  *   the attribute to check
  * @return
  *   true if the attribute is equal to the specified type parameter, false
  *   otherwise
  */
def check_equal[T <: Attribute: ClassTag](that_attr: Attribute): Boolean =
  that_attr match {
    case _: T => true
    case _    => false
  }

/*≡==--==≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||   CONSTRAINT CONTEXT   ||
\*≡==---==≡≡≡≡≡≡≡≡==---==≡*/

/** Represents the context for constraints, holding variable constraints. */
class ConstraintContext() {

  val var_constraints: DictType[String, Attribute] = DictType
    .empty[String, Attribute]

}

/** Abstract class representing an IRDL constraint. */
abstract class IRDLConstraint {

  /** Verifies if the given attribute satisfies the constraint within the
    * provided context.
    *
    * @param that_attr
    *   the attribute to verify
    * @param constraint_ctx
    *   the context in which to verify the constraint
    */
  def verify(that_attr: Attribute, constraint_ctx: ConstraintContext): Unit

  /** Verifies if the given sequence of attributes satisfies the constraint
    * within the provided context.
    *
    * @param those_attrs
    *   the sequence of attributes to verify
    * @param constraint_ctx
    *   the context in which to verify the constraint
    */
  def verify(
      those_attrs: Seq[Attribute],
      constraint_ctx: ConstraintContext
  ): Unit = for (attr <- those_attrs) verify(attr, constraint_ctx)

  /** Combines this constraint with another constraint using logical AND.
    *
    * @param that
    *   the other constraint to combine with
    * @return
    *   a new AllOf constraint representing the logical AND of this and the
    *   other constraint
    */
  def &&(that: IRDLConstraint): IRDLConstraint = AllOf(Seq(this, that))

  /** Combines this constraint with another constraint using logical OR.
    *
    * @param that
    *   the other constraint to combine with
    * @return
    *   a new AnyOf constraint representing the logical OR of this and the other
    *   constraint
    */
  def ||(that: IRDLConstraint): IRDLConstraint = AnyOf(Seq(this, that))
}

/*≡==--==≡≡≡≡==--=≡≡*\
||     ANY ATTR     ||
\*≡==---==≡≡==---==≡*/

/** An IRDL constraint that matches any attribute. */
object AnyAttr extends IRDLConstraint {

  /** Verifies if the given attribute satisfies the constraint within the
    * provided context. This implementation always succeeds as it matches any
    * attribute.
    *
    * @param that_attr
    *   the attribute to verify
    * @param constraint_ctx
    *   the context in which to verify the constraint
    */
  override def verify(
      that_attr: Attribute,
      constraint_ctx: ConstraintContext
  ): Unit = {}

  /** Returns a string representation of the constraint. */
  override def toString(): String = "AnyAttr"
}

/*≡==--==≡≡≡≡==--=≡≡*\
||    EQUAL ATTR    ||
\*≡==---==≡≡==---==≡*/

/** An IRDL constraint that checks if an attribute is equal to a specified
  * attribute.
  *
  * @param this_attr
  *   the attribute to compare against
  */
case class EqualAttr(val this_attr: Attribute) extends IRDLConstraint {

  /** Verifies if the given attribute satisfies the constraint within the
    * provided context. This implementation checks if the given attribute is
    * equal to the specified attribute.
    *
    * @param that_attr
    *   the attribute to verify
    * @param constraint_ctx
    *   the context in which to verify the constraint
    * @throws VerifyException
    *   if the attributes are not equal
    */
  override def verify(
      that_attr: Attribute,
      constraint_ctx: ConstraintContext
  ): Unit = {

    if (this_attr != that_attr) {
      val errstr = s"${that_attr.name} does not equal ${this_attr.name}:\n" +
        s"Got ${that_attr.custom_print}, expected ${this_attr.custom_print}"
      throw new VerifyException(errstr)
    }
  }

  /** Returns a string representation of the constraint. */
  override def toString = this_attr.toString
}

/** A given conversion from Attribute to IRDLConstraint. */
given attr2constraint: Conversion[Attribute, IRDLConstraint] with {
  def apply(attr: Attribute): IRDLConstraint = EqualAttr(attr)
}

/*≡==--==≡≡≡==--=≡≡*\
||    BASE ATTR    ||
\*≡==---==≡==---==≡*/

/** A case class representing a base attribute constraint.
  *
  * @tparam T
  *   the type parameter representing the class to check against
  */
case class BaseAttr[T <: Attribute: ClassTag]() extends IRDLConstraint {

  /** Verifies if the given attribute satisfies the constraint within the
    * provided context. This implementation checks if the given attribute is of
    * the specified type.
    *
    * @param that_attr
    *   the attribute to verify
    * @param constraint_ctx
    *   the context in which to verify the constraint
    * @throws VerifyException
    *   if the attribute is not of the specified type
    */
  override def verify(
      that_attr: Attribute,
      constraint_ctx: ConstraintContext
  ): Unit = {

    that_attr match {
      case _: T =>
      case _    =>
        val className = implicitly[ClassTag[T]].runtimeClass.getName
        val errstr = s"${that_attr.name}'s class does not equal ${className}\n"
        throw new VerifyException(errstr)
    }
  }

  /** Returns a string representation of the constraint. */
  override def toString =
    s"BaseAttr[${implicitly[ClassTag[T]].runtimeClass.getName}]()"

}

/*≡==--==≡≡≡≡==--=≡≡*\
||      ANY OF      ||
\*≡==---==≡≡==---==≡*/

/** A case class representing a constraint that matches if any of the provided
  * constraints match.
  *
  * @param constraints_in
  *   the sequence of constraints to check against
  */
case class AnyOf(constraints_in: Seq[IRDLConstraint]) extends IRDLConstraint {

  val constraints: Seq[IRDLConstraint] = constraints_in.flatMap(_ match {
    case x: AnyOf => x.constraints
    case y        => Seq(y)
  })

  /** Verifies if the given attribute satisfies any of the provided constraints
    * within the given context.
    *
    * @param that_attr
    *   the attribute to verify
    * @param constraint_ctx
    *   the context in which to verify the constraint
    * @throws VerifyException
    *   if the attribute does not satisfy any of the provided constraints
    */
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
      val errstr = s"${that_attr.name} does not match $toString\n"
      throw new VerifyException(errstr)
    }
  }

  /** Returns a string representation of the constraint. */
  override def toString = constraints.mkString(" || ")
}

/*≡==--==≡≡≡≡==--=≡≡*\
||      ALL OF      ||
\*≡==---==≡≡==---==≡*/

/** A case class representing a constraint that matches if all of the provided
  * constraints match.
  *
  * @param constraints_in
  *   the sequence of constraints to check against
  */
case class AllOf(constraints_in: Seq[IRDLConstraint]) extends IRDLConstraint {

  val constraints: Seq[IRDLConstraint] = constraints_in.flatMap(_ match {
    case x: AllOf => x.constraints
    case y        => Seq(y)
  })

  /** Verifies if the given attribute satisfies all of the provided constraints
    * within the given context.
    *
    * @param that_attr
    *   the attribute to verify
    * @param constraint_ctx
    *   the context in which to verify the constraint
    * @throws VerifyException
    *   if the attribute does not satisfy all of the provided constraints
    */
  override def verify(
      that_attr: Attribute,
      constraint_ctx: ConstraintContext
  ): Unit = for (c <- constraints) c.verify(that_attr, constraint_ctx)

  /** Returns a string representation of the constraint. */
  override def toString = constraints.mkString(" && ")
}

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||    PARAMETRIZED ATTR    ||
\*≡==---==≡≡≡≡≡≡≡≡≡==---==≡*/

/** A case class representing a constraint for parametrized attributes.
  *
  * @tparam T
  *   the type parameter representing the class to check against
  * @param constraints
  *   the sequence of constraints to check against the parameters of the
  *   attribute
  */
case class ParametrizedAttrConstraint[T <: Attribute: ClassTag](
    val constraints: Seq[IRDLConstraint]
) extends IRDLConstraint {

  /** Verifies if the given attribute satisfies the constraint within the
    * provided context. This implementation checks if the given attribute is of
    * the specified type and if its parameters satisfy the provided constraints.
    *
    * @param that_attr
    *   the attribute to verify
    * @param constraint_ctx
    *   the context in which to verify the constraint
    * @throws VerifyException
    *   if the attribute does not satisfy the constraint
    */
  override def verify(
      that_attr: Attribute,
      constraint_ctx: ConstraintContext
  ): Unit = check_same_class(that_attr) match {
    case true => that_attr match {
        case x: ParametrizedAttribute =>
          if (!(x.parameters.length == constraints.length)) {
            throw new VerifyException(s"Expected ${constraints
                .length} parameters, got ${x.parameters.length}\n")
          }
          for ((p, c) <- x.parameters zip constraints) p match {
            case p: Attribute => c.verify(p, constraint_ctx)
            case p: Seq[_]    => c
                .verify(p.asInstanceOf[Seq[Attribute]], constraint_ctx)
          }

        case _ => throw new VerifyException(
            "Attribute being verified must be of type ParametrizedAttribute.\n"
          )
      }
    case false =>
      val className = implicitly[ClassTag[T]].runtimeClass.getName
      val errstr = s"${that_attr.name}'s class does not equal ${className}.\n"
      throw new VerifyException(errstr)
  }

  /** Returns a string representation of the constraint. */
  override def toString = s"ParametrizedAttrConstraint[${implicitly[ClassTag[T]]
      .runtimeClass.getName}](${constraints})"

}

/*≡==--==≡≡≡≡≡≡≡≡==--=≡≡*\
||    VAR CONSTRAINT    ||
\*≡==---==≡≡≡≡≡≡==---==≡*/

/** A case class representing a variable constraint.
  *
  * @param name
  *   the name of the variable
  * @param constraint
  *   the constraint to apply to the variable
  */
case class VarConstraint(val name: String, val constraint: IRDLConstraint)
    extends IRDLConstraint {

  /** Verifies if the given attribute satisfies the constraint within the
    * provided context. This implementation checks if the variable constraint is
    * already present in the context. If present, it verifies if the attribute
    * matches the existing constraint. If not present, it adds the attribute to
    * the context after verifying it against the constraint.
    *
    * @param that_attr
    *   the attribute to verify
    * @param constraint_ctx
    *   the context in which to verify the constraint
    * @throws VerifyException
    *   if the attribute does not satisfy the constraint
    */
  override def verify(
      that_attr: Attribute,
      constraint_ctx: ConstraintContext
  ): Unit = {

    val var_consts = constraint_ctx.var_constraints

    var_consts.contains(name) match {
      case true => if (var_consts.apply(name) != that_attr) {
          throw new VerifyException("oh mah gawd")
        }
      case false =>
        constraint.verify(that_attr, constraint_ctx)
        var_consts += ((name, that_attr))
    }
  }

}
