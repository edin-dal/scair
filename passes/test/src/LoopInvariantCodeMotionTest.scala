package scair

import org.scalatest.*
import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*
import scair.MLContext
import scair.ir.*
import scair.parse.Parser
import scair.passes.licm.*
import scair.transformations.RewriteMethods
import scair.transformations.Rewriter

/** Covers the number of operations moved, which the pass itself discards and so
  * no filecheck test can observe. Behaviour is covered by
  * tests/filecheck/transformations/licm.mlir.
  */
class LoopInvariantCodeMotionTest extends AnyFlatSpec with BeforeAndAfter:

  val ctx = MLContext()
  scair.dialects.allDialects.foreach(ctx.registerDialect)

  given Rewriter = RewriteMethods

  /** Parses `text` and returns its first loop. */
  def parseLoop(text: String): LoopLike =
    val module = Parser(ctx).parse(text).get.value
    module.nested.collectFirst { case l: LoopLike => l }.get

  val invariantLoop = """
%lb = "arith.constant"() <{value = 0 : index}> : () -> index
%c = "arith.constant"() <{value = 7 : i32}> : () -> i32
"scf.for"(%lb, %lb, %lb) ({
^bb0(%i: index):
  %a = "arith.addi"(%c, %c) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  %b = "arith.addi"(%a, %c) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  %v = "arith.addi"(%i, %lb) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
  "scf.yield"() : () -> ()
}) : (index, index, index) -> ()
"""

  "moveLoopInvariantCode" should "report how many operations it moved" in
    withClue("Both invariant adds move, the loop dependent one stays: ") {
      moveLoopInvariantCode(parseLoop(invariantLoop)) shouldBe 2
    }

  it should "report nothing moved when there is nothing invariant" in {
    val loop = parseLoop("""
%lb = "arith.constant"() <{value = 0 : index}> : () -> index
"scf.for"(%lb, %lb, %lb) ({
^bb0(%i: index):
  %v = "arith.addi"(%i, %i) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
  "scf.yield"() : () -> ()
}) : (index, index, index) -> ()
""")

    moveLoopInvariantCode(loop) shouldBe 0
  }

  it should "be idempotent" in {
    val loop = parseLoop(invariantLoop)

    moveLoopInvariantCode(loop) shouldBe 2
    moveLoopInvariantCode(loop) shouldBe 0
  }

  it should "hoist a chain in dependency order in a single run" in
    withClue("Users are re-checked once their operands have been hoisted: ") {
      val loop = parseLoop(invariantLoop)
      moveLoopInvariantCode(loop)

      val hoisted = loop.containerBlock.get.operations.toSeq
        .takeWhile(_ ne loop).filter(_.name == "arith.addi")

      // The chain must keep its order, or the result would not dominate its use.
      hoisted.length shouldBe 2
      hoisted(1).operands should contain(hoisted(0).results.head)
    }
