package scair.dialects.test

import scair.clair.*
import scair.dialects.builtin.*
import scair.ir.*

object TestOp extends OperationCompanion[TestOp]:
  override def name: String = "test.op"

given OperationCompanion[TestOp] = TestOp

case class TestOp(
    override val operands: Seq[Operand[Attribute]] = Seq(),
    override val successors: Seq[Successor] = Seq(),
    override val results: Seq[Result[Attribute]] = Seq(),
    override val regions: Seq[Region] = Seq(),
    override val properties: Map[String, Attribute] = Map
      .empty[String, Attribute],
    override val attributes: DictType[String, Attribute] = DictType
      .empty[String, Attribute],
) extends Operation:
  override def name = "test.op"

  override def updated(
      operands: Seq[Value[Attribute]] = operands,
      successors: Seq[Block] = successors,
      results: Seq[Result[Attribute]] = results.map(_.typ).map(Result(_)),
      regions: Seq[Region] = detachedRegions,
      properties: Map[String, Attribute] = properties,
      attributes: DictType[String, Attribute] = attributes,
  ) =
    TestOp(
      operands,
      successors,
      results,
      regions,
      properties,
      attributes,
    )

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||   SPECULATION TEST OPS      ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

// Ported from mlir/test/lib/Dialect/Test/TestOps.td, where they exist to
// exercise every arm of the ConditionallySpeculatable interface.

/** Op used to test conditional speculation. This op can always be speculatively
  * executed.
  */
case class AlwaysSpeculatableOp(
    result: Result[IntegerType]
) extends DerivedOperation["test.always_speculatable_op"]
    with Pure derives OpDefs

/** Op used to test conditional speculation. This op can never be speculatively
  * executed.
  */
case class NeverSpeculatableOp(
    result: Result[IntegerType]
) extends DerivedOperation["test.never_speculatable_op"]
    with ConditionallySpeculatable derives OpDefs:

  override def getSpeculatability: Speculatability =
    Speculatability.NotSpeculatable

/** Op used to test conditional speculation. This op can be speculatively
  * executed if the input to it is a constant.
  *
  * Upstream this checks specifically for an `arith.constant`. The arith dialect
  * is downstream of core, so match the `ConstantLike` trait instead — which is
  * closer to the intent anyway.
  */
case class ConditionallySpeculatableOp(
    input: Operand[IntegerType],
    result: Result[IntegerType],
) extends DerivedOperation["test.conditionally_speculatable_op"]
    with ConditionallySpeculatable
    with NoMemoryEffect derives OpDefs:

  override def getSpeculatability: Speculatability =
    input.owner match
      case Some(_: ConstantLike) => Speculatability.Speculatable
      case _                     => Speculatability.NotSpeculatable

/** Op used to test conditional speculation. This op can be speculatively
  * executed only if all the ops in the attached region can be.
  */
case class RecursivelySpeculatableOp(
    body: Region,
    result: Result[IntegerType],
) extends DerivedOperation["test.recursively_speculatable_op"]
    with RecursivelySpeculatable
    with RecursiveMemoryEffects derives OpDefs

/** Used in a region, to yield the corresponding type for that operation. */
case class RegionYieldOp(
    result: Operand[Attribute]
) extends DerivedOperation["test.region_yield"]
    with AssemblyFormat["$result `:` type($result) attr-dict"]
    with IsTerminator
    with Pure derives OpDefs

val Test: Dialect = summonDialect[
  EmptyTuple,
  (
      TestOp,
      AlwaysSpeculatableOp,
      NeverSpeculatableOp,
      ConditionallySpeculatableOp,
      RecursivelySpeculatableOp,
      RegionYieldOp,
  ),
]
