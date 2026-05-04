package scair.ir

import org.scalatest.*
import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*
import org.scalatest.prop.TableDrivenPropertyChecks.forAll
import org.scalatest.prop.Tables.Table
import scair.print.AssemblyPrinter
import scair.dialects.builtin.*
import scair.ir.*
import scair.transformations.*
import scair.parse.Parser
import scair.MLContext
import java.io.*

import scair.dialects.test.TestOp

class BlockTest extends AnyFlatSpec with BeforeAndAfter:

  var printer = new AssemblyPrinter(true);

  forAll(
    Table(
      ("block", "ir"),
      (
        Block(),
        """^bb0:
""",
      ),
      (
        Block(Seq()),
        """^bb0:
""",
      ),
      (
        Block(Seq(), Seq()),
        """^bb0:
""",
      ),
      (
        Block(Seq(TestOp())),
        """^bb0:
  "test.op"() : () -> ()
""",
      ),
      (
        Block(TestOp()),
        """^bb0:
  "test.op"() : () -> ()
""",
      ),
      (
        Block(
          Seq(I32),
          (args: Iterable[Value[IntegerType]]) =>
            Seq(TestOp(operands = args.toSeq)),
        ),
        """^bb0(%0: i32):
  "test.op"(%0) : (i32) -> ()
""",
      ),
      (
        Block(
          I32,
          (arg: Value[IntegerType]) => Seq(TestOp(operands = Seq(arg))),
        ),
        """^bb0(%0: i32):
  "test.op"(%0) : (i32) -> ()
""",
      ),
      (
        Block.typed(
          (I32, IndexType()),
          (
              arg1: Value[IntegerType & Attribute],
              arg2: Value[IndexType & Attribute],
          ) => Seq(TestOp(operands = Seq(arg1, arg2))),
        ),
        """^bb0(%0: i32, %1: index):
  "test.op"(%0, %1) : (i32, index) -> ()
""",
      ),
    )
  ) { (block: Block, ir: String) =>
    val out = StringWriter()
    printer = new AssemblyPrinter(true, p = PrintWriter(out))
    // Run the pqrser on the input and check
    printer.print(block)
    out.toString() shouldEqual ir

  }

  "Block Creation" should "correctly index operations on creation" in {
    val ctx = MLContext()
    val parser = Parser(ctx, allowUnregisteredDialect = true)
    val input = parser.parse("""
%0 = "test.op1"() : () -> i32
%1 = "test.op2"(%0) : (i32) -> i32
"test.op3"(%1) : (i32) -> ()
""").get.value

    input.regions(0).blocks(0).operations.zipWithIndex.foreach {
      case (op, idx) =>
        op.blockIndex shouldEqual idx
    }
  }

  "Block Creation" should "correctly index nested operations on creation" in {
    val ctx = MLContext()
    val parser = Parser(ctx, allowUnregisteredDialect = true)
    val input = parser.parse("""
%0 = "test.op1"() ({
  %0 = "test.op1"() : () -> i32
  %1 = "test.op2"(%0) : (i32) -> i32
  "test.op3"(%1) ({
    %3 = "test.op1"() : () -> i32
    %4 = "test.op2"(%3) : (i32) -> i32
    "test.op3"(%1) : (i32) -> ()
  }) : (i32) -> ()
}) : () -> i32
%1 = "test.op2"(%0) : (i32) -> i32
"test.op3"(%1) : (i32) -> ()
""").get.value

    input.regions(0).blocks(0).operations(0).regions(0).blocks(0).operations.zipWithIndex.foreach {
      case (op, idx) =>
        op.blockIndex shouldEqual idx
    }

    input.regions(0).blocks(0).operations(0) // test.op1
      .regions(0).blocks(0).operations(2)    // nested test.op3
      .regions(0).blocks(0).operations.zipWithIndex.foreach {
      case (op, idx) =>
        op.blockIndex shouldEqual idx
    }
  }

  "Block Transformations via Passes" should
    "break the indexing of operations" in {
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

      final class TestPass(ctx: MLContext) extends WalkerPass(ctx):
        override val name = "test-test"
        override final val walker = PatternRewriteWalker(TestPattern)

      val pass = TestPass(ctx)

      val output = pass.transform(input)

      output.regions(0).blocks(0).isOpOrderValid shouldEqual false
    }
  
  "Block Transformations via Passes" should "preserve the indexing of untouched blocks" in {
    val ctx = MLContext()
    val parser = Parser(ctx, allowUnregisteredDialect = true)
    val input = parser.parse("""
%0 = "test.op1"() ({
  %0 = "test.op1"() : () -> i32
  %1 = "test.op2"(%0) : (i32) -> i32
  "test.op3"(%1) ({
    %0 = "test.op1"() : () -> i64
    %1 = "test.op2"(%0) : (i64) -> i64
    "test.op3"(%1) : (i64) -> ()
  }) : (i32) -> ()
}) : () -> i32
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

    final class TestPass(ctx: MLContext) extends WalkerPass(ctx):
      override val name = "test-test"
      override final val walker = PatternRewriteWalker(TestPattern)

    val pass = TestPass(ctx)

    val output = pass.transform(input)

    output.regions(0).blocks(0).isOpOrderValid shouldEqual false
    input.regions(0).blocks(0).operations(0) // test.op1
      .regions(0).blocks(0).operations(2)    // nested test.op3
      .regions(0).blocks(0).isOpOrderValid shouldEqual true
  }
