package scair.constraints

import scair.ir.Attribute
import scair.utils.OK

import scala.quoted.*

/** Compile-time context threaded through the macro's field iteration.
  * Constraint companions use this to track state across fields (e.g., Var
  * bindings).
  */
class MacroConstraintContext(
    val bindings: Map[String, Any] = Map.empty
):

  def updated(key: String, value: Any): MacroConstraintContext =
    MacroConstraintContext(bindings + (key -> value))

/** Protocol for constraint-driven compile-time code generation. Each
  * constraint's companion object may extend this trait to provide its own
  * verification code generation logic. The macro discovers companions via
  * classloading at expansion time and calls `macroVerify`.
  *
  * Out-of-tree constraints: define a companion extending this trait in a module
  * compiled before the operations that use it.
  */
trait ConstraintCompanion:

  /** Generate verification code for this constraint at macro expansion time.
    *
    * @param constraintType
    *   The full constraint type (e.g., `Var["T"]`)
    * @param attr
    *   Expression for the attribute to verify
    * @param ctx
    *   Compile-time context for cross-field state
    * @return
    *   (verification code snippet, updated context)
    */
  def macroVerify(using
      Quotes
  )(
      constraintType: Type[?],
      attr: Expr[Attribute],
      ctx: MacroConstraintContext,
  ): (Expr[OK[Unit]], MacroConstraintContext)
