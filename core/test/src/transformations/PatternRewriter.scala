import org.scalatest.*
import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.transformations.*
import scair.parse.Parser
import scair.print.AssemblyPrinter
import scair.MLContext
import java.io.*
import java.io.StringWriter

class PatternRewriterTest extends AnyFlatSpec:

  "PatternRewriterWalker" should "not trigger on removed operations" in {
    // Create a simple module with an operation
    val ctx = MLContext()
    val parser = Parser(ctx, allowUnregisteredDialect = true)
    val input = parser.parse("""
%0 = "test.op1"() : () -> i32
%1 = "test.op2"(%0) : (i32) -> i32
"test.op3"(%1) : (i32) -> ()
""").get.value

    object TestPattern extends RewritePattern:
      override def matchAndRewrite(
          op: Operation,
          rewriter: PatternRewriter,
      ): Unit =
        val newRes = op.results.map(_ match
          case Result(I32) => Result(I64)
          case r           => r)
        if newRes != op.results then
          rewriter.replaceOp(op, op.updated(results = newRes))

    // Apply the pattern
    val walker = PatternRewriteWalker(TestPattern)

    walker.rewrite(input)
    val out = StringWriter()
    AssemblyPrinter(p = out).print(input)
    out.toString() shouldEqual """builtin.module {
  %0 = "test.op1"() : () -> i64
  %1 = "test.op2"(%0) : (i64) -> i64
  "test.op3"(%1) : (i64) -> ()
}
"""
  }

  /** Parses `text` as a module and returns it. */
  def parseModule(text: String): Operation =
    Parser(MLContext(), allowUnregisteredDialect = true).parse(text).get.value

  /** Renders `op` for comparison. */
  def render(op: Operation): String =
    val out = StringWriter()
    AssemblyPrinter(p = out).print(op)
    out.toString()

  "moveOpsBefore" should "move an operation up within its block" in {
    val module = parseModule("""
%0 = "test.op1"() : () -> i32
%1 = "test.op2"() : () -> i32
"test.op3"(%0, %1) : (i32, i32) -> ()
""")
    val block = module.regions.head.blocks.head
    val Seq(op1, op2, op3) = block.operations.toSeq: @unchecked

    RewriteMethods.moveOpsBefore(op1, op2)

    block.operations.toSeq shouldBe Seq(op2, op1, op3)
    op2.containerBlock shouldBe Some(block)
  }

  "moveOpsAfter" should "move an operation down within its block" in {
    val module = parseModule("""
%0 = "test.op1"() : () -> i32
"test.op2"() : () -> ()
"test.op3"() : () -> ()
""")
    val block = module.regions.head.blocks.head
    val Seq(op1, op2, op3) = block.operations.toSeq: @unchecked

    RewriteMethods.moveOpsAfter(op3, op1)

    block.operations.toSeq shouldBe Seq(op2, op3, op1)
  }

  it should "handle an operation already adjacent to the anchor" in
    withClue("The insertion point must name the slot the operation leaves: ") {
      val module = parseModule("""
"test.op1"() : () -> ()
"test.op2"() : () -> ()
""")
      val block = module.regions.head.blocks.head
      val Seq(op1, op2) = block.operations.toSeq: @unchecked

      // op2 is op1.next, so an eagerly computed InsertPoint.after(op1) would
      // point at op2 itself, which is detached by the time we insert.
      RewriteMethods.moveOpsAfter(op1, op2)

      block.operations.toSeq shouldBe Seq(op1, op2)
    }

  "moveOpsAt" should "move an operation out of a nested region" in {
    val module = parseModule("""
%0 = "test.outer"() : () -> i32
"test.region"() ({
  %1 = "test.inner"(%0) : (i32) -> i32
  "test.use"(%1) : (i32) -> ()
}) : () -> ()
""")
    val outerBlock = module.regions.head.blocks.head
    val Seq(outer, region) = outerBlock.operations.toSeq: @unchecked
    val innerBlock = region.regions.head.blocks.head
    val inner = innerBlock.operations.head

    RewriteMethods.moveOpsBefore(region, inner)

    outerBlock.operations.toSeq shouldBe Seq(outer, inner, region)
    inner.containerBlock shouldBe Some(outerBlock)
    innerBlock.operations.toSeq.length shouldBe 1
  }

  it should "keep use lists intact across a move" in
    withClue("A move detaches and re-inserts, and both maintain uses: ") {
      val module = parseModule("""
%0 = "test.def"() : () -> i32
%1 = "test.user"(%0) : (i32) -> i32
"test.anchor"() : () -> ()
""")
      val block = module.regions.head.blocks.head
      val Seq(defOp, user, anchor) = block.operations.toSeq: @unchecked
      val value = defOp.results.head

      value.uses.toSeq shouldBe Seq(Use(user, 0))

      RewriteMethods.moveOpsAfter(anchor, user)

      value.uses.toSeq shouldBe Seq(Use(user, 0))
      user.operands shouldBe Seq(value)
    }

  it should "reject moving a detached operation" in {
    val module = parseModule("""
"test.anchor"() : () -> ()
""")
    val anchor = module.regions.head.blocks.head.operations.head
    val detached = anchor.updated()

    intercept[Exception] {
      RewriteMethods.moveOpsBefore(anchor, detached)
    }.getMessage shouldBe "Cannot move an operation that has no parents."
  }

  "insertOpAfterMatchedOp" should "insert after, not before" in {
    val module = parseModule("""
"test.a"() : () -> ()
"test.b"() : () -> ()
""")

    object InsertAfterA extends RewritePattern:
      override def matchAndRewrite(
          op: Operation,
          rewriter: PatternRewriter,
      ): Unit =
        if op.name == "test.a" then
          rewriter
            .insertOpAfterMatchedOp(UnregisteredOperation("test.inserted")())

    PatternRewriteWalker(InsertAfterA).rewrite(module)

    render(module) shouldEqual """builtin.module {
  "test.a"() : () -> ()
  "test.inserted"() : () -> ()
  "test.b"() : () -> ()
}
"""
  }
