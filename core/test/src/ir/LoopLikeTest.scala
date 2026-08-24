package scair

import org.scalatest.*
import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*
import scair.clair.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.transformations.RewriteMethods

/** A single region loop, standing in for `scf.for` and friends. */
case class SingleRegionLoopOp(
    body: Region
) extends DerivedOperation["singleregionloop"]
    with LoopLike(body) derives OpDefs

/** A two region loop, standing in for `scf.while`. */
case class TwoRegionLoopOp(
    before: Region,
    after: Region,
) extends DerivedOperation["tworegionloop"]
    with LoopLike(before, after) derives OpDefs

/** Any old operation producing a value, to define values with. */
case class ProducerOp(
    result: Result[IntegerType]
) extends DerivedOperation["producer"] derives OpDefs

class LoopLikeTest extends AnyFlatSpec with BeforeAndAfter:

  val i32 = IntegerType(IntData(32), Signless)

  def producer = ProducerOp(Result(i32))

  "LoopLike" should "expose its loop regions" in {
    val body = Region(Block())
    val loop = SingleRegionLoopOp(body)

    loop.loopRegions shouldBe Seq(body)

    loop match
      case LoopLike(regions) => regions shouldBe Seq(body)
      case _                 => fail("should have matched LoopLike")
  }

  it should "expose every region of a multi region loop" in {
    val before = Region(Block())
    val after = Region(Block())
    val loop = TwoRegionLoopOp(before, after)

    loop.loopRegions shouldBe Seq(before, after)
  }

  "isDefinedOutsideOfLoop" should "hold for a value defined above the loop" in {
    val outer = producer
    val loop = SingleRegionLoopOp(Region(Block()))
    Block(operations = Seq(outer, loop))

    loop.isDefinedOutsideOfLoop(outer.result) shouldBe true
  }

  it should "not hold for a value defined in the loop body" in {
    val inner = producer
    val loop = SingleRegionLoopOp(Region(Block(operations = Seq(inner))))

    loop.isDefinedOutsideOfLoop(inner.result) shouldBe false
  }

  it should "not hold for a block argument of the loop body" in
    withClue("An induction variable is defined inside the loop: ") {
      val block = Block(argumentsTypes = Seq(i32), operations = Seq())
      val loop = SingleRegionLoopOp(Region(block))

      loop.isDefinedOutsideOfLoop(block.arguments.head) shouldBe false
    }

  it should "not hold for a value defined deep inside the loop body" in {
    val deep = producer
    val nested = SingleRegionLoopOp(Region(Block(operations = Seq(deep))))
    val loop = SingleRegionLoopOp(Region(Block(operations = Seq(nested))))

    loop.isDefinedOutsideOfLoop(deep.result) shouldBe false
  }

  it should
    "not hold for a value defined in either region of a two region loop" in {
      val inBefore = producer
      val inAfter = producer
      val loop = TwoRegionLoopOp(
        Region(Block(operations = Seq(inBefore))),
        Region(Block(operations = Seq(inAfter))),
      )

      loop.isDefinedOutsideOfLoop(inBefore.result) shouldBe false
      loop.isDefinedOutsideOfLoop(inAfter.result) shouldBe false
    }

  it should "not hold for a detached value" in
    withClue("A value with no owner is conservatively taken to be inside: ") {
      val loop = SingleRegionLoopOp(Region(Block()))
      loop.isDefinedOutsideOfLoop(Value(i32)) shouldBe false
    }

  "moveOutOfLoop" should "move an operation to just before the loop" in {
    given scair.transformations.Rewriter = RewriteMethods

    val hoisted = producer
    val kept = producer
    val body = Block(operations = Seq(hoisted, kept))
    val loop = SingleRegionLoopOp(Region(body))
    val outer = Block(operations = Seq(loop))

    loop.moveOutOfLoop(hoisted)

    outer.operations.toSeq shouldBe Seq(hoisted, loop)
    body.operations.toSeq shouldBe Seq(kept)
    loop.isDefinedOutsideOfLoop(hoisted.result) shouldBe true
  }
