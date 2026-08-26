package scair.ir

// ░██████╗ ██╗ ██████╗░ ███████╗   ███████╗ ███████╗ ███████╗ ███████╗ ░█████╗░ ████████╗ ░██████╗
// ██╔════╝ ██║ ██╔══██╗ ██╔════╝   ██╔════╝ ██╔════╝ ██╔════╝ ██╔════╝ ██╔══██╗ ╚══██╔══╝ ██╔════╝
// ╚█████╗░ ██║ ██║░░██║ █████╗░░   █████╗░░ █████╗░░ █████╗░░ ██║░░╚═╝ ██║░░██║ ░░░██║░░░ ╚█████╗░
// ░╚═══██╗ ██║ ██║░░██║ ██╔══╝░░   ██╔══╝░░ ██╔══╝░░ ██╔══╝░░ ██║░░██╗ ██║░░██║ ░░░██║░░░ ░╚═══██╗
// ██████╔╝ ██║ ██████╔╝ ███████╗   ███████╗ ██║░░░░░ ██║░░░░░ ╚█████╔╝ ╚█████╔╝ ░░░██║░░░ ██████╔╝
// ╚═════╝░ ╚═╝ ╚═════╝░ ╚══════╝   ╚══════╝ ╚═╝░░░░░ ╚═╝░░░░░ ░╚════╝░ ░╚════╝░ ░░░╚═╝░░░ ╚═════╝░

/** Mirrors `mlir/lib/Interfaces/SideEffectInterfaces.{h,cpp}`.
  *
  * TODO: Upstream, `NoMemoryEffect` is not a marker but the ODS shorthand
  * `MemoryEffects<[]>` for an op implementing `MemoryEffectOpInterface` with an
  * empty effect list, over a model of `Allocate`/`Free`/`Read`/`Write` effects,
  * resources, and `EffectInstance`s. Code motion only ever asks the yes/no
  * `isMemoryEffectFree`, so we keep the marker for now; the effect-instance
  * model, and with it `hasSingleEffect`, `hasEffect` and
  * `getEffectsRecursively`, is future work.
  */

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||   MEMORY EFFECT TRAITS   ||
\*≡==---==≡≡≡≡≡≡≡≡≡==---==≡*/

/** An operation that has no memory effects of its own. */
trait NoMemoryEffect extends Operation

/** ODS `def RecursiveMemoryEffects : NativeOpTrait<"RecursiveMemoryEffects">`.
  *
  * The memory effects of the operation are those of the operations nested
  * within its regions. Having no effect interface of its own, the operation
  * itself is taken to have no memory effects.
  */
trait RecursiveMemoryEffects extends Operation

/*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
||   SIDE EFFECT UTILS   ||
\*≡==---==≡≡≡≡≡≡≡==---==≡*/

/** Whether the operation has no memory effects, recursing through operations
  * that derive their effects from their regions.
  */
def isMemoryEffectFree(op: Operation): Boolean =
  op match
    case _: NoMemoryEffect         => true
    case _: RecursiveMemoryEffects =>
      op.regions
        .forall(_.blocks.forall(_.operations.forall(isMemoryEffectFree)))
    // Neither: the effects of the operation are unknown, so it cannot be known
    // to be movable.
    case _ => false
