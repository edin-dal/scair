package scair.dialects.constraints

import scair.constraints.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.utils.*

import scala.quoted.*

// ███████╗ ██╗░░██╗ ░█████╗░ ███╗░░░███╗ ██████╗░ ██╗░░░░░ ███████╗
// ██╔════╝ ╚██╗██╔╝ ██╔══██╗ ████╗░████║ ██╔══██╗ ██║░░░░░ ██╔════╝
// █████╗░░ ░╚███╔╝░ ███████║ ██╔████╔██║ ██████╔╝ ██║░░░░░ █████╗░░
// ██╔══╝░░ ░██╔██╗░ ██╔══██║ ██║╚██╔╝██║ ██╔═══╝░ ██║░░░░░ ██╔══╝░░
// ███████╗ ██╔╝╚██╗ ██║░░██║ ██║░╚═╝░██║ ██║░░░░░ ███████╗ ███████╗
// ╚══════╝ ╚═╝░░╚═╝ ╚═╝░░╚═╝ ╚═╝░░░░░╚═╝ ╚═╝░░░░░ ╚══════╝ ╚══════╝

/** An out-of-tree constraint that cannot be expressed by composing the built-in
  * algebra: it computes over the attribute rather than matching on it.
  *
  * The equivalent of xDSL's `VectorRankConstraint`, and the model for any
  * constraint a downstream dialect needs to define for itself. The extension
  * point is the *companion object*: give it `ConstraintGen` and the interpreter
  * will call it while expanding any operation that uses `Width`.
  */
trait Width[N <: Int] extends Constraint

object Width extends ConstraintGen:

  def verify(c: Type[?], attr: Expr[Attribute], path: String)(using
      Quotes,
      GenCtx,
      ErrCtx,
  ) =
    c match
      case '[Width[n]] =>
        val expected = Expr(Type.valueOfConstant[n].get.asInstanceOf[Int])
        Some('{
          $attr match
            case IntegerType(IntData(bits), _) if bits == BigInt($expected) =>
              OK()
            case a =>
              ${
                fail(path, '{ "an integer of width " + $expected }, 'a)
              }
        })

  /** A width pins the type down completely only together with a signedness, so
    * this constraint alone infers a signless integer of that width.
    */
  override def infer(c: Type[?])(using Quotes, GenCtx) =
    c match
      case '[Width[n]] =>
        val expected = Expr(Type.valueOfConstant[n].get.asInstanceOf[Int])
        Some('{ IntegerType(IntData(BigInt($expected)), Signless) })
