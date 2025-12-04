package scair

import org.scalatest.*
import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*
import scair.ir.*
import scair.clair.macros.*

case class FillerOp(
    override val operands: Seq[Operand[Attribute]] = Seq(),
    override val successors: Seq[Successor] = Seq(),
    override val results: Seq[Result[Attribute]] = Seq(),
    override val regions: Seq[Region] = Seq()
) extends DerivedOperation["filler", FillerOp] derives DerivedOperationCompanion

case class TerminatorOp(
    override val operands: Seq[Value[Attribute]] = Seq(),
    override val successors: Seq[Block] = Seq(),
    override val results: Seq[Result[Attribute]] = Seq(),
    override val regions: Seq[Region] = Seq()
) extends DerivedOperation["terminator", TerminatorOp]
    with IsTerminator derives DerivedOperationCompanion

case class NoTerminatorOp(
    override val operands: Seq[Value[Attribute]] = Seq(),
    override val successors: Seq[Block] = Seq(),
    override val results: Seq[Result[Attribute]] = Seq(),
    override val regions: Seq[Region] = Seq()
) extends DerivedOperation["noterminator", NoTerminatorOp]
    with NoTerminator derives DerivedOperationCompanion

class TraitTest extends AnyFlatSpec with BeforeAndAfter:

  "IsTerminator Test1" should "pass the test the IsTerminator trait" in {

    val filler1 = new FillerOp()
    val filler2 = new FillerOp()
    val terminator = new TerminatorOp()

    val block = new Block(operations = Seq(filler1, filler2, terminator))

    terminator.container_block = Some(block)
    block.verify()
  }

  "IsTerminator Test2" should "not pass the test the IsTerminator trait" in {
    withClue("Terminator not last in block: ") {
      val filler1 = new FillerOp()
      val filler2 = new FillerOp()
      val terminator = new TerminatorOp()

      val block = new Block(operations = Seq(filler1, terminator, filler2))

      terminator.container_block = Some(block)

      val exception = block.verify()

      exception shouldBe Left(
        "Operation 'terminator' marked as a terminator, but is not the last operation within its container block"
      )
    }
  }

  "IsTerminator Test3" should "not pass the test the IsTerminator trait" in {
    withClue("Terminator not contained in block: ") {
      val terminator = new TerminatorOp()

      val exception = terminator.verify()

      exception shouldBe Left(
        "Operation 'terminator' marked as a terminator, but is not contained in any block."
      )
    }
  }

  "NoTerminator Test1" should "pass the test the NoTerminator trait" in {

    val filler1 = new FillerOp()
    val filler2 = new FillerOp()
    val filler3 = new FillerOp()
    val filler4 = new FillerOp()
    val filler5 = new FillerOp()
    val filler6 = new FillerOp()

    val block1 = new Block(operations = Seq(filler1, filler2))
    val block2 = new Block(operations = Seq(filler3, filler4))
    val block3 = new Block(operations = Seq(filler5, filler6))

    val region1 = Region(block1)
    val region2 = Region(block2)
    val region3 = Region(block3)

    val noterminator =
      new NoTerminatorOp(regions = Seq(region1, region2, region3))

    noterminator.verify()
  }

  "NoTerminator Test2" should "not pass the test the NoTerminator trait" in {

    val filler1 = new FillerOp()
    val filler2 = new FillerOp()
    val filler3 = new FillerOp()
    val filler4 = new FillerOp()
    val filler5 = new FillerOp()
    val filler6 = new FillerOp()

    val block1 = new Block(operations = Seq(filler1, filler2))
    val block2 = new Block(operations = Seq(filler3, filler4))
    val block3 = new Block(operations = Seq(filler5, filler6))

    val region1 = Region(block1, block2)
    val region2 = Region(block3)

    val noterminator =
      new NoTerminatorOp(regions = Seq(region1, region2))

    val exception = noterminator.verify()

    exception shouldBe Left(
      "NoTerminator Operation 'noterminator' requires single-block regions"
    )
  }
