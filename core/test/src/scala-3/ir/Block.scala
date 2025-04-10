package scair.ir

import org.scalatest.*
import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*
import org.scalatest.prop.*
import org.scalatest.prop.TableDrivenPropertyChecks.forAll
import org.scalatest.prop.Tables.Table
import scair.Printer
import scair.dialects.builtin.I32
import scair.dialects.builtin.IntegerType

case class TestOp(
    override val operands: Seq[Value[Attribute]] = Seq(),
    override val successors: Seq[Block] = Seq(),
    override val results_types: Seq[Attribute] = Seq(),
    override val regions: Seq[Region] = Seq(),
    override val properties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val attributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends BaseOperation(
      name = "test.op",
      operands,
      successors,
      results_types,
      regions,
      properties,
      attributes
    )

class BlockTest extends AnyFlatSpec with BeforeAndAfter {

  var printer = new Printer(true);

  forAll(
    Table(
      ("block", "ir"),
      (
        Block(),
        """^bb0():
"""
      ),
      (
        Block(Seq()),
        """^bb0():
"""
      ),
      (
        Block(Seq(), Seq()),
        """^bb0():
"""
      ),
      (
        Block(Seq(TestOp())),
        """^bb0():
  "test.op"() : () -> ()"""
      ),
      (
        Block(TestOp()),
        """^bb0():
  "test.op"() : () -> ()"""
      ),
      (
        Block(
          Seq(I32),
          (args: Iterable[Value[Attribute]]) =>
            Seq(TestOp(operands = args.toSeq))
        ),
        """^bb0(%0: i32):
  "test.op"(%0) : (i32) -> ()"""
      ),
      (
        Block(
          I32,
          (arg: Value[Attribute]) => Seq(TestOp(operands = Seq(arg)))
        ),
        """^bb0(%0: i32):
  "test.op"(%0) : (i32) -> ()"""
      )
    )
  ) { (block: Block, ir: String) =>
    printer = new Printer(true)
    // Run the pqrser on the input and check
    printer.printBlock(block) shouldEqual ir

  }

}
