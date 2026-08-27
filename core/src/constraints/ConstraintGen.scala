package scair.constraints

import scair.ir.*
import scair.utils.*

import scala.quoted.*

// ░█████╗░ ░█████╗░ ███╗░░██╗ ░██████╗ ████████╗ ██████╗░ ░█████╗░ ██╗ ███╗░░██╗ ████████╗
// ██╔══██╗ ██╔══██╗ ████╗░██║ ██╔════╝ ╚══██╔══╝ ██╔══██╗ ██╔══██╗ ██║ ████╗░██║ ╚══██╔══╝
// ██║░░╚═╝ ██║░░██║ ██╔██╗██║ ╚█████╗░ ░░░██║░░░ ██████╔╝ ███████║ ██║ ██╔██╗██║ ░░░██║░░░
// ██║░░██╗ ██║░░██║ ██║╚████║ ░╚═══██╗ ░░░██║░░░ ██╔══██╗ ██╔══██║ ██║ ██║╚████║ ░░░██║░░░
// ╚█████╔╝ ╚█████╔╝ ██║░╚███║ ██████╔╝ ░░░██║░░░ ██║░░██║ ██║░░██║ ██║ ██║░╚███║ ░░░██║░░░
// ░╚════╝░ ░╚════╝░ ╚═╝░░╚══╝ ╚═════╝░ ░░░╚═╝░░░ ╚═╝░░╚═╝ ╚═╝░░╚═╝ ╚═╝ ╚═╝░░╚══╝ ░░░╚═╝░░░
//
// ░██████╗░ ███████╗ ███╗░░██╗
// ██╔════╝░ ██╔════╝ ████╗░██║
// ██║░░██╗░ █████╗░░ ██╔██╗██║
// ██║░░╚██╗ ██╔══╝░░ ██║╚████║
// ╚██████╔╝ ███████╗ ██║░╚███║
// ░╚═════╝░ ╚══════╝ ╚═╝░░╚══╝

/** How a constraint is compiled.
  *
  * Every constraint works this way -- there is no privileged set the
  * interpreter knows about. `EqAttr`, `Var`, `AllOf` and the rest each carry
  * their own generator on their companion object, exactly as a constraint
  * defined in a downstream dialect does, and `verifyC` is nothing but dispatch
  * to whichever generator the constraint names. Whatever the built-ins can do,
  * an out-of-tree constraint can do too, because they use the same interface.
  *
  * To define a constraint, declare it and give its companion object this trait.
  * `scair.dialects.constraints.Width` is a worked example of a constraint that
  * has to compute over the attribute, and the built-ins in `Constraints.scala`
  * are worked examples of everything else.
  *
  * A generator runs *during* the macro expansion of any operation that uses the
  * constraint, and only its output reaches the class file; the generator itself
  * has no runtime presence at the use site. Because it is executed at compile
  * time, it must live in a module compiled before its use sites -- the same
  * separate-compilation rule that already governs `derives OpDefs`.
  */
trait ConstraintGen:

  /** Emit code checking that `attr` satisfies the constraint type `c`.
    *
    * `None` means the constraint needs no code at all, which is how `AnyAttr`
    * and a `Var`'s first, binding occurrence stay genuinely free.
    *
    * @param c
    *   the full, applied constraint type, e.g. `Width[32]`. Match on it to
    *   recover the constraint's type arguments.
    * @param path
    *   a compile-time description of what is being checked, e.g. `operand
    *   'lhs'`. Pass it to [[fail]] so failures say where they happened; it only
    *   ever reaches the generated code as a constant, on a failure branch.
    */
  def verify(c: Type[?], attr: Expr[Attribute], path: String)(using
      Quotes,
      GenCtx,
      ErrCtx,
  ): Option[Expr[OK[Unit]]]

  /** Recursing into a position whose value cannot be named further down the
    * generated method -- a pattern binding, or a branch that may not run --
    * must go through `GenCtx.unnameable`, so that a constraint variable bound
    * in there gets a runtime slot instead of a dangling expression. `Param` and
    * `AnyOf` are the two built-ins that do this.
    */

  /** Emit an expression producing the single attribute this constraint admits,
    * if it admits exactly one.
    *
    * This is xDSL's `can_infer` and `infer` fused: at compile time both halves
    * are available at once, so "can it be inferred" and "here is the inferred
    * value" are one question, and the contract that the first implies the
    * second holds by construction rather than by documentation.
    */
  def infer(c: Type[?])(using Quotes, GenCtx): Option[Expr[Attribute]] = None

  /** Bind this constraint's variables from a value already known to satisfy it,
    * without emitting any check.
    *
    * Used to seed inference: the assembly-format parser knows the operand types
    * it just parsed, so binding them makes result types determined by them
    * inferable. Only unconditional positions should bind.
    */
  def bind(c: Type[?], attr: Expr[Attribute])(using Quotes, GenCtx): Unit = ()
