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

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||   SPECULATION INTERFACES   ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

/** Mirrors `Speculation::Speculatability`. */
enum Speculatability:

  /** The operation cannot be speculatively executed. This could be because it
    * may invoke undefined behavior or have other side effects.
    */
  case NotSpeculatable

  /** The operation can be speculatively executed. It does not have any side
    * effects or undefined behavior.
    */
  case Speculatable

  /** The operation can be speculatively executed if all the operations in all
    * attached regions can also be speculatively executed.
    */
  case RecursivelySpeculatable

/** MLIR's `ConditionallySpeculatable` op interface.
  *
  * Implement this directly to answer dynamically, as [[AlwaysSpeculatable]] and
  * [[RecursivelySpeculatable]] only ever give a fixed answer.
  */
trait ConditionallySpeculatable extends Operation:
  def getSpeculatability: Speculatability

/** ODS `AlwaysSpeculatableImplTrait`. */
trait AlwaysSpeculatable extends ConditionallySpeculatable:

  final override def getSpeculatability: Speculatability = Speculatability
    .Speculatable

/** ODS `RecursivelySpeculatableImplTrait`. */
trait RecursivelySpeculatable extends ConditionallySpeculatable:

  final override def getSpeculatability: Speculatability =
    Speculatability.RecursivelySpeculatable

/** ODS `def Pure : TraitList<[AlwaysSpeculatable, NoMemoryEffect]>`.
  *
  * Note that this is the conjunction of two independent axes, memory effects
  * and speculatability, rather than a member of either.
  */
trait Pure extends AlwaysSpeculatable with NoMemoryEffect

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

/** Whether the operation can be speculatively executed. */
def isSpeculatable(op: Operation): Boolean = op match
  case c: ConditionallySpeculatable =>
    c.getSpeculatability match
      case Speculatability.RecursivelySpeculatable =>
        op.regions.forall(_.blocks.forall(_.operations.forall(isSpeculatable)))
      case Speculatability.Speculatable    => true
      case Speculatability.NotSpeculatable => false
  // Not implementing the interface at all is a distinct, conservative state.
  case _ => false
