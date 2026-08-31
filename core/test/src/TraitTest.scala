package scair

import org.scalatest.*
import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*
import scair.ir.*
import scair.clair.*
import scair.dialects.builtin.*
import scair.dialects.test.*
import scair.utils.*

case class FillerOp(
    override val operands: Seq[Operand[Attribute]] = Seq(),
    override val successors: Seq[Successor] = Seq(),
    override val results: Seq[Result[Attribute]] = Seq(),
    override val regions: Seq[Region] = Seq(),
) extends DerivedOperation["filler"] derives OpDefs

case class TerminatorOp(
    override val operands: Seq[Value[Attribute]] = Seq(),
    override val successors: Seq[Block] = Seq(),
    override val results: Seq[Result[Attribute]] = Seq(),
    override val regions: Seq[Region] = Seq(),
) extends DerivedOperation["terminator"]
    with IsTerminator derives OpDefs

case class NoTerminatorOp(
    override val operands: Seq[Value[Attribute]] = Seq(),
    override val successors: Seq[Block] = Seq(),
    override val results: Seq[Result[Attribute]] = Seq(),
    override val regions: Seq[Region] = Seq(),
) extends DerivedOperation["noterminator"]
    with NoTerminator derives OpDefs

/** Declares no memory effects of its own. */
case class EffectFreeOp(
    override val regions: Seq[Region] = Seq()
) extends DerivedOperation["effectfree"]
    with NoMemoryEffect derives OpDefs

/** Declares nothing, so its effects are unknown. */
case class UnknownEffectsOp(
    override val regions: Seq[Region] = Seq()
) extends DerivedOperation["unknowneffects"] derives OpDefs

/** Derives its effects from the operations nested in its regions. */
case class RecursiveEffectsOp(
    override val regions: Seq[Region] = Seq()
) extends DerivedOperation["recursiveeffects"]
    with RecursiveMemoryEffects derives OpDefs

/** A constant, to drive `test.conditionally_speculatable_op`. */
case class ConstantOp(
    result: Result[IntegerType]
) extends DerivedOperation["constant"]
    with ConstantLike(IntData(0))
    with Pure derives OpDefs

class TraitTest extends AnyFlatSpec with BeforeAndAfter:

  "IsTerminator Test1" should "pass the test the IsTerminator trait" in {

    val filler1 = new FillerOp()
    val filler2 = new FillerOp()
    val terminator = new TerminatorOp()

    val block = Block(operations = Seq(filler1, filler2, terminator))

    terminator.containerBlock = Some(block)
    block.verify()
  }

  "IsTerminator Test2" should "not pass the test the IsTerminator trait" in
    withClue("Terminator not last in block: ") {
      val filler1 = new FillerOp()
      val filler2 = new FillerOp()
      val terminator = new TerminatorOp()

      val block = Block(operations = Seq(filler1, terminator, filler2))

      terminator.containerBlock = Some(block)

      val exception = block.verify()

      exception shouldBe Err(
        "Operation 'terminator' marked as a terminator, but is not the last operation within its container block",
        Some(terminator),
      )
    }

  "IsTerminator Test3" should "not pass the test the IsTerminator trait" in
    withClue("Terminator not contained in block: ") {
      val terminator = new TerminatorOp()

      val exception = terminator.verify()

      exception shouldBe Err(
        "Operation 'terminator' marked as a terminator, but is not contained in any block.",
        Some(terminator),
      )
    }

  "NoTerminator Test1" should "pass the test the NoTerminator trait" in {

    val filler1 = new FillerOp()
    val filler2 = new FillerOp()
    val filler3 = new FillerOp()
    val filler4 = new FillerOp()
    val filler5 = new FillerOp()
    val filler6 = new FillerOp()

    val block1 = Block(operations = Seq(filler1, filler2))
    val block2 = Block(operations = Seq(filler3, filler4))
    val block3 = Block(operations = Seq(filler5, filler6))

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

    val block1 = Block(operations = Seq(filler1, filler2))
    val block2 = Block(operations = Seq(filler3, filler4))
    val block3 = Block(operations = Seq(filler5, filler6))

    val region1 = Region(block1, block2)
    val region2 = Region(block3)

    val noterminator =
      new NoTerminatorOp(regions = Seq(region1, region2))

    val exception = noterminator.verify()

    exception shouldBe Err(
      "NoTerminator Operation 'noterminator' requires single-block regions",
      Some(noterminator),
    )
  }

  /*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
  ||   MEMORY EFFECTS      ||
  \*≡==---==≡≡≡≡≡≡≡==---==≡*/

  /** Wraps `ops` in a single region, single block operation of type `T`. */
  def containing(make: Seq[Region] => Operation)(ops: Operation*): Operation =
    make(Seq(Region(Block(operations = ops))))

  "isMemoryEffectFree" should "hold for an operation declaring no effects" in {
    isMemoryEffectFree(EffectFreeOp()) shouldBe true
  }

  it should "not hold for an operation whose effects are unknown" in {
    isMemoryEffectFree(UnknownEffectsOp()) shouldBe false
  }

  it should "not recurse into a region of an operation declaring no effects" in
    withClue("A declared absence of effects covers the whole operation: ") {
      val op = containing(EffectFreeOp(_))(UnknownEffectsOp())
      isMemoryEffectFree(op) shouldBe true
    }

  it should "hold for a recursive operation over effect free operations" in {
    val op = containing(RecursiveEffectsOp(_))(EffectFreeOp(), EffectFreeOp())
    isMemoryEffectFree(op) shouldBe true
  }

  it should "not hold for a recursive operation over an unknown operation" in {
    val op =
      containing(RecursiveEffectsOp(_))(EffectFreeOp(), UnknownEffectsOp())
    isMemoryEffectFree(op) shouldBe false
  }

  it should "hold for an empty recursive operation" in {
    isMemoryEffectFree(RecursiveEffectsOp()) shouldBe true
  }

  it should "recurse through nested recursive operations" in {
    val deep = containing(RecursiveEffectsOp(_))(UnknownEffectsOp())
    val op = containing(RecursiveEffectsOp(_))(EffectFreeOp(), deep)
    isMemoryEffectFree(op) shouldBe false

    val pure = containing(RecursiveEffectsOp(_))(EffectFreeOp())
    isMemoryEffectFree(containing(RecursiveEffectsOp(_))(pure)) shouldBe true
  }

  /*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
  ||     SPECULATION       ||
  \*≡==---==≡≡≡≡≡≡≡==---==≡*/

  val i32 = IntegerType(IntData(32), Signless)

  def alwaysSpeculatable = AlwaysSpeculatableOp(Result(i32))
  def neverSpeculatable = NeverSpeculatableOp(Result(i32))

  def recursivelySpeculatable(ops: Operation*) =
    RecursivelySpeculatableOp(Region(Block(operations = ops)), Result(i32))

  "isSpeculatable" should "hold for an always speculatable operation" in {
    isSpeculatable(alwaysSpeculatable) shouldBe true
  }

  it should "not hold for a never speculatable operation" in {
    isSpeculatable(neverSpeculatable) shouldBe false
  }

  it should "not hold for an operation implementing no interface" in
    withClue("Not implementing the interface is its own conservative state: ") {
      isSpeculatable(UnknownEffectsOp()) shouldBe false
      isSpeculatable(EffectFreeOp()) shouldBe false
    }

  it should "hold for a recursive operation over speculatable operations" in {
    val op = recursivelySpeculatable(alwaysSpeculatable, alwaysSpeculatable)
    isSpeculatable(op) shouldBe true
  }

  it should
    "not hold for a recursive operation over a never speculatable one" in {
      val op = recursivelySpeculatable(alwaysSpeculatable, neverSpeculatable)
      isSpeculatable(op) shouldBe false
    }

  it should "not hold for a recursive operation over an uninterfaced one" in {
    isSpeculatable(recursivelySpeculatable(UnknownEffectsOp())) shouldBe false
  }

  it should "hold for an empty recursive operation" in {
    isSpeculatable(recursivelySpeculatable()) shouldBe true
  }

  it should "recurse through nested recursive operations" in {
    val deep = recursivelySpeculatable(neverSpeculatable)
    isSpeculatable(recursivelySpeculatable(deep)) shouldBe false

    val fine = recursivelySpeculatable(alwaysSpeculatable)
    isSpeculatable(recursivelySpeculatable(fine)) shouldBe true
  }

  it should
    "be decided dynamically by a conditionally speculatable operation" in
    withClue("The interface answers per operation, not per trait: ") {
      val constant = ConstantOp(Result(i32))
      val onConstant = ConditionallySpeculatableOp(constant.result, Result(i32))
      isSpeculatable(onConstant) shouldBe true

      val nonConstant = alwaysSpeculatable
      val onNonConstant =
        ConditionallySpeculatableOp(nonConstant.result, Result(i32))
      isSpeculatable(onNonConstant) shouldBe false
    }

  "Pure" should "satisfy both axes, and still match NoMemoryEffect" in
    withClue("Canonicalize and CSE match on NoMemoryEffect: ") {
      val op = alwaysSpeculatable
      isMemoryEffectFree(op) shouldBe true
      isSpeculatable(op) shouldBe true
      op shouldBe a[NoMemoryEffect]
      op shouldBe a[ConditionallySpeculatable]
    }
