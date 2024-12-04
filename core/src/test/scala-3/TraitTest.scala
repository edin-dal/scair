package scair

import org.scalatest._
import Inspectors._
import flatspec._
import matchers.should.Matchers._
import prop._

import scala.collection.mutable
import Parser._

import scair.dialects.builtin._
import scair.ir._
import scair.traits._

object FillerOp extends OperationObject {
  override def name: String = "filler"
  override def factory: FactoryType = FillerOp.apply
}
case class FillerOp(
    override val operands: ListType[Value[Attribute]] = ListType(),
    override val successors: ListType[Block] = ListType(),
    override val results: ListType[Value[Attribute]] = ListType(),
    override val regions: ListType[Region] = ListType(),
    override val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends RegisteredOperation(name = "filler") {}

object TerminatorOp extends OperationObject {
  override def name: String = "terminator"
  override def factory: FactoryType = TerminatorOp.apply
}
case class TerminatorOp(
    override val operands: ListType[Value[Attribute]] = ListType(),
    override val successors: ListType[Block] = ListType(),
    override val results: ListType[Value[Attribute]] = ListType(),
    override val regions: ListType[Region] = ListType(),
    override val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends RegisteredOperation(name = "terminator")
    with IsTerminator {}

object NoTerminatorOp extends OperationObject {
  override def name: String = "noterminator"
  override def factory: FactoryType = NoTerminatorOp.apply
}
case class NoTerminatorOp(
    override val operands: ListType[Value[Attribute]] = ListType(),
    override val successors: ListType[Block] = ListType(),
    override val results: ListType[Value[Attribute]] = ListType(),
    override val regions: ListType[Region] = ListType(),
    override val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends RegisteredOperation(name = "noterminator")
    with NoTerminator {}

class TraitTest extends AnyFlatSpec with BeforeAndAfter {

  "IsTerminator Test1" should "pass the test the IsTerminator trait" in {

    val filler1 = new FillerOp()
    val filler2 = new FillerOp()
    val terminator = new TerminatorOp()

    val block = new Block(operations = ListType(filler1, filler2, terminator))

    terminator.container_block = Some(block)
    block.verify()
  }

  "IsTerminator Test2" should "not pass the test the IsTerminator trait" in {
    withClue("Terminator not last in block: ") {
      val filler1 = new FillerOp()
      val filler2 = new FillerOp()
      val terminator = new TerminatorOp()

      val block = new Block(operations = ListType(filler1, terminator, filler2))

      terminator.container_block = Some(block)

      val exception = intercept[Exception](
        block.verify()
      ).getMessage

      exception shouldBe "Operation 'terminator' marked as a terminator, but is not the last operation within its container block"
    }
  }

  "IsTerminator Test3" should "not pass the test the IsTerminator trait" in {
    withClue("Terminator not contained in block: ") {
      val terminator = new TerminatorOp()

      val exception = intercept[Exception](
        terminator.verify()
      ).getMessage

      exception shouldBe "Operation 'terminator' marked as a terminator, but is not contained in any block."
    }
  }

  "NoTerminator Test1" should "pass the test the NoTerminator trait" in {

    val filler1 = new FillerOp()
    val filler2 = new FillerOp()
    val filler3 = new FillerOp()
    val filler4 = new FillerOp()
    val filler5 = new FillerOp()
    val filler6 = new FillerOp()

    val block1 = new Block(operations = ListType(filler1, filler2))
    val block2 = new Block(operations = ListType(filler3, filler4))
    val block3 = new Block(operations = ListType(filler5, filler6))

    val region1 = new Region(Seq(block1))
    val region2 = new Region(Seq(block2))
    val region3 = new Region(Seq(block3))

    val noterminator =
      new NoTerminatorOp(regions = ListType(region1, region2, region3))

    noterminator.verify()
  }

  "NoTerminator Test2" should "not pass the test the NoTerminator trait" in {

    val filler1 = new FillerOp()
    val filler2 = new FillerOp()
    val filler3 = new FillerOp()
    val filler4 = new FillerOp()
    val filler5 = new FillerOp()
    val filler6 = new FillerOp()

    val block1 = new Block(operations = ListType(filler1, filler2))
    val block2 = new Block(operations = ListType(filler3, filler4))
    val block3 = new Block(operations = ListType(filler5, filler6))

    val region1 = new Region(Seq(block1, block2))
    val region2 = new Region(Seq(block2))

    val noterminator =
      new NoTerminatorOp(regions = ListType(region1, region2))

    val exception = intercept[Exception](
      noterminator.verify()
    ).getMessage

    exception shouldBe "NoTerminator Operation 'noterminator' requires single-block regions"
  }

}
