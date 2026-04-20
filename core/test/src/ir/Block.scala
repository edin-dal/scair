package scair.ir

import org.scalatest.*
import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*
import org.scalatest.prop.TableDrivenPropertyChecks.forAll
import org.scalatest.prop.Tables.Table
import scair.print.AssemblyPrinter
import scair.dialects.builtin.{I32, IntegerType, IndexType}
import java.io.StringWriter
import java.io.PrintWriter

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
