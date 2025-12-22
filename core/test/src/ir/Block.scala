package scair.ir

import org.scalatest.*
import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*
import org.scalatest.prop.TableDrivenPropertyChecks.forAll
import org.scalatest.prop.Tables.Table
import scair.Printer
import scair.dialects.builtin.I32
import java.io.StringWriter
import java.io.PrintWriter

import scair.dialects.test.TestOp

class BlockTest extends AnyFlatSpec with BeforeAndAfter:

  var printer = new Printer(true);

  forAll(
    Table(
      ("block", "ir"),
      (
        Block(),
        """^bb0():
""",
      ),
      (
        Block(Seq()),
        """^bb0():
""",
      ),
      (
        Block(Seq(), Seq()),
        """^bb0():
""",
      ),
      (
        Block(Seq(TestOp())),
        """^bb0():
  "test.op"() : () -> ()
""",
      ),
      (
        Block(TestOp()),
        """^bb0():
  "test.op"() : () -> ()
""",
      ),
      (
        Block(
          Seq(I32),
          (args: Iterable[Value[Attribute]]) =>
            Seq(TestOp(operands = args.toSeq)),
        ),
        """^bb0(%0: i32):
  "test.op"(%0) : (i32) -> ()
""",
      ),
      (
        Block(
          I32,
          (arg: Value[Attribute]) => Seq(TestOp(operands = Seq(arg))),
        ),
        """^bb0(%0: i32):
  "test.op"(%0) : (i32) -> ()
""",
      ),
    )
  ) { (block: Block, ir: String) =>
    val out = StringWriter()
    printer = new Printer(true, p = PrintWriter(out))
    // Run the pqrser on the input and check
    printer.print(block)(using 0)
    out.toString() shouldEqual ir

  }
