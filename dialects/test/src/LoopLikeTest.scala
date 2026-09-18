package scair

import org.scalatest.*
import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*
import scair.MLContext
import scair.ir.*
import scair.parse.Parser

/** Checks that the operations that should be loop like actually are, and over
  * the right regions. The interface's own semantics are covered by core's
  * LoopLikeTest.
  */
class DialectLoopLikeTest extends AnyFlatSpec with BeforeAndAfter:

  val ctx = MLContext()
  scair.dialects.allDialects.foreach(ctx.registerDialect)

  /** Parses `text` and returns the operations of the top level block. */
  def parseOps(text: String): Seq[Operation] =
    Parser(ctx).parse(text).get.value.regions.head.blocks.head.operations.toSeq

  "scf.for" should "be loop like over its body" in {
    val op = parseOps("""
%i = "test.op"() : () -> index
"scf.for"(%i, %i, %i) ({
^bb0(%iv: index):
  "scf.yield"() : () -> ()
}) : (index, index, index) -> ()
""").last

    op shouldBe a[LoopLike]
    op.asInstanceOf[LoopLike].loopRegions shouldBe op.regions
  }

  "scf.while" should "be loop like over both its regions" in {
    val op = parseOps("""
%c = "test.op"() : () -> i1
"scf.while"(%c) ({
  "scf.condition"(%c) : (i1) -> ()
}, {
  "scf.yield"() : () -> ()
}) : (i1) -> ()
""").last

    val loop = op.asInstanceOf[LoopLike]
    loop.loopRegions.length shouldBe 2
    loop.loopRegions shouldBe op.regions
  }

  "scf.parallel" should "be loop like over its body" in {
    val op = parseOps("""
%i = "test.op"() : () -> index
"scf.parallel"(%i, %i, %i) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
^bb0(%iv: index):
  "scf.reduce"() : () -> ()
}) : (index, index, index) -> ()
""").last

    op.asInstanceOf[LoopLike].loopRegions shouldBe op.regions
  }

  "scf.forall" should "be loop like over its body" in {
    val op = parseOps("""
%i = "test.op"() : () -> index
"scf.forall"(%i, %i, %i) <{staticLowerBound = array<i64: 0>, staticUpperBound = array<i64: 1>, staticStep = array<i64: 1>, operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
^bb0(%iv: index):
  "scf.forall.in_parallel"() ({
  }) : () -> ()
}) : (index, index, index) -> ()
""").last

    op.asInstanceOf[LoopLike].loopRegions shouldBe op.regions
  }

  "affine.for" should "be loop like over its body" in {
    val op = parseOps("""
"affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, upperBoundMap = affine_map<() -> (256)>, step = 1 : index, operandSegmentSizes = array<i32: 0, 0, 0>}> ({
^bb0(%iv: index):
  "affine.yield"() : () -> ()
}) : () -> ()
""").last

    op.asInstanceOf[LoopLike].loopRegions shouldBe op.regions
  }

  "affine.parallel" should "be loop like over its body" in {
    val op = parseOps("""
%n = "test.op"() : () -> index
"affine.parallel"(%n) <{lowerBoundsMap = affine_map<() -> (0)>, lowerBoundsGroups = dense<1> : vector<1xi32>, upperBoundsMap = affine_map<()[s0] -> (s0)>, upperBoundsGroups = dense<1> : vector<1xi32>, steps = [1 : i64], reductions = []}> ({
^bb0(%iv: index):
  "affine.yield"() : () -> ()
}) : (index) -> ()
""").last

    op.asInstanceOf[LoopLike].loopRegions shouldBe op.regions
  }

  "operations that are not loops" should "not be loop like" in
    withClue("Only operations whose regions may repeat are loop like: ") {
      val ops = parseOps("""
%c = "test.op"() : () -> i1
"scf.if"(%c) ({
}, {
}) : (i1) -> ()
"scf.execute_region"() ({
}) : () -> ()
""")

      ops.tail.foreach(_ should not be a[LoopLike])
    }
