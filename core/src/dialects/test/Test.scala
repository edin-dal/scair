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

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||       ENUM TEST OPS         ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

/** A standalone enum attribute, i.e., one that is an attribute in its own right
  * rather than being backed by an integer one. Mirrors upstream's
  * `Arith_RoundingModeAttr`, keyword for keyword, under the test namespace.
  */
enum RoundingMode(caseName: String)
    extends EnumAttr("test.rounding_mode", caseName):
  case ToNearestEven extends RoundingMode("to_nearest_even")
  case Downward extends RoundingMode("downward")
  case Upward extends RoundingMode("upward")
  case TowardZero extends RoundingMode("toward_zero")
  case ToNearestAway extends RoundingMode("to_nearest_away")

/** Op used to test standalone enum attributes, as a required and as an optional
  * property.
  */
case class RoundingModeOp(
    result: Result[IntegerType],
    mode: RoundingMode,
    fallbackMode: Option[RoundingMode] = None,
) extends DerivedOperation["test.rounding_mode_op"] derives OpDefs

val Test: Dialect = summonDialect[
  Tuple1[RoundingMode],
  (
      TestOp,
      AlwaysSpeculatableOp,
      NeverSpeculatableOp,
      ConditionallySpeculatableOp,
      RecursivelySpeculatableOp,
      RoundingModeOp,
  ),
]
