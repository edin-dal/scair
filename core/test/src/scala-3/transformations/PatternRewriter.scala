import org.scalatest.*
import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.transformations.*
import scair.parse.Parser
import scair.Printer
import scair.MLContext
import java.io.*

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
    val baos = ByteArrayOutputStream()
    Printer(p = new PrintWriter(baos)).print(input)
    baos.toString() shouldEqual """builtin.module {
  %0 = "test.op1"() : () -> i64
  %1 = "test.op2"(%0) : (i64) -> i64
  "test.op3"(%1) : (i64) -> ()
}
"""
  }
