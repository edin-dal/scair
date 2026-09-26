package scair.passes.licm

import scair.MLContext
import scair.ir.*
import scair.transformations.*

import scala.collection.mutable.Queue

//
// ██╗░░░░░ ░█████╗░ ░█████╗░ ██████╗░
// ██║░░░░░ ██╔══██╗ ██╔══██╗ ██╔══██╗
// ██║░░░░░ ██║░░██║ ██║░░██║ ██████╔╝
// ██║░░░░░ ██║░░██║ ██║░░██║ ██╔═══╝░
// ███████╗ ╚█████╔╝ ╚█████╔╝ ██║░░░░░
// ╚══════╝ ░╚════╝░ ░╚════╝░ ╚═╝░░░░░
//
// ██╗ ███╗░░██╗ ██╗░░░██╗ ░█████╗░ ██████╗░ ██╗ ░█████╗░ ███╗░░██╗ ████████╗
// ██║ ████╗░██║ ██║░░░██║ ██╔══██╗ ██╔══██╗ ██║ ██╔══██╗ ████╗░██║ ╚══██╔══╝
// ██║ ██╔██╗██║ ╚██╗░██╔╝ ███████║ ██████╔╝ ██║ ███████║ ██╔██╗██║ ░░░██║░░░
// ██║ ██║╚████║ ░╚████╔╝░ ██╔══██║ ██╔══██╗ ██║ ██╔══██║ ██║╚████║ ░░░██║░░░
// ██║ ██║░╚███║ ░░╚██╔╝░░ ██║░░██║ ██║░░██║ ██║ ██║░░██║ ██║░╚███║ ░░░██║░░░
// ╚═╝ ╚═╝░░╚══╝ ░░░╚═╝░░░ ╚═╝░░╚═╝ ╚═╝░░╚═╝ ╚═╝ ╚═╝░░╚═╝ ╚═╝░░╚══╝ ░░░╚═╝░░░
//

extension (op: Operation)

  def nested: Seq[Operation] =
    op.regions.flatMap(_.blocks.flatMap(_.operations))

/** The core LICM algorithm, mirroring
  * `mlir/lib/Transforms/Utils/LoopInvariantCodeMotionUtils.cpp`.
  */

/** Whether `op` can be hoisted, by checking that neither it nor any operation
  * contained in it depends on a value defined inside the loop, and that it is
  * not a terminator.
  */
def canBeHoisted(
    op: Operation,
    definedOutside: Value[Attribute] => Boolean,
): Boolean =
  op match
    // Do not move terminators.
    case _: IsTerminator => false
    case _               =>
      // Walk the nested operations and check that all used values are either
      // defined outside of the loop or in a nested region, but not at the level
      // of the loop body.
      def walk(child: Operation): Boolean =
        child.operands.forall(operand =>
          operand.owner.exists(op.isAncestor) || definedOutside(operand)
        ) && child.nested.forall(walk) // nested operations must be hoistable
      walk(op)

/** Whether `node` sits directly in `region`. Compared by identity, as regions
  * and blocks are case classes and so structurally equal ones abound.
  */
private def isDirectlyIn(op: Operation, region: Region): Boolean =
  op.containerBlock.flatMap(_.containerRegion).exists(_ eq region)

/** The operations using any result of `op`. */
private def users(op: Operation): Seq[Operation] =
  op.results.flatMap(_.uses).map(_.operation).distinct

/** Move loop invariant code out of `regions`, and return how many operations
  * were moved.
  */
def moveLoopInvariantCode(
    regions: Seq[Region],
    isDefinedOutsideRegion: (Value[Attribute], Region) => Boolean,
    shouldMoveOutOfRegion: (Operation, Region) => Boolean,
    moveOutOfRegion: (Operation, Region) => Unit,
): Int =
  var numMoved = 0

  for region <- regions do
    // Add top-level operations in the loop body to the worklist. Nested regions
    // are deliberately left alone: their semantics are unknown to this
    // rewriting, and if they are loops they are processed in their own right.
    val worklist = Queue.from(region.blocks.flatMap(_.operations))

    def definedOutside(value: Value[Attribute]) =
      isDefinedOutsideRegion(value, region)

    while worklist.nonEmpty do
      val op = worklist.dequeue()
      // Skip operations that have already been moved.
      if isDirectlyIn(op, region) && shouldMoveOutOfRegion(op, region) &&
        canBeHoisted(op, definedOutside)
      then
        moveOutOfRegion(op, region)
        numMoved += 1
        // Since the operation has been moved, its users within the top level of
        // the loop body may have become invariant too.
        worklist.enqueueAll(users(op).filter(isDirectlyIn(_, region)))

  numMoved

/** Move loop invariant code out of `loop`, and return how many operations were
  * moved.
  */
def moveLoopInvariantCode(loop: LoopLike)(using Rewriter): Int =
  moveLoopInvariantCode(
    loop.loopRegions,
    (value, _) => loop.isDefinedOutsideOfLoop(value),
    (op, _) => isPure(op),
    (op, _) => loop.moveOutOfLoop(op),
  )

final class LoopInvariantCodeMotion(ctx: MLContext) extends ModulePass(ctx):
  override val name = "licm"

  override def transform(op: Operation): Operation =
    given Rewriter = RewriteMethods

    // Walk through all loops in innermost-loop-first order. This way, we first
    // LICM from the inner loop, and place the operations in the outer loop,
    // which in turn can be further LICM'ed.
    def walk(op: Operation): Unit =
      op.nested.foreach(walk)
      op match
        case loop: LoopLike => moveLoopInvariantCode(loop)
        case _              => ()

    walk(op)
    op
