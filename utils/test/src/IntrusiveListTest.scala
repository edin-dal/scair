package scair.utils.test

import scair.utils.{IntrusiveNode, IntrusiveList}

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class IntrusiveListTest extends AnyFlatSpec with Matchers:

  case class I(val i: Int) extends IntrusiveNode[I]

  behavior of "IntrusiveList"

  "IntrusiveList.apply" should "instantiate empty list" in:
    IntrusiveList[I]() shouldEqual Seq()

  it should "instantiate list with elements" in:
    IntrusiveList[I](I(1), I(2), I(3)) shouldEqual Seq(I(1), I(2), I(3))

  "prepend" should "add elements and maintain correct order" in:
    val list = IntrusiveList[I]().prepend(I(3)).prepend(I(2)).prepend(I(1))
    list shouldEqual IntrusiveList[I](I(1), I(2), I(3))

  it should "allow prepending a removed middle element" in:
    val e1 = I(1); val e2 = I(2); val e3 = I(3)
    val list = IntrusiveList[I](e1, e2, e3)
    list.remove(1) // remove e2
    list shouldEqual Seq(I(1), I(3))
    list.prepend(e2) // re-add e2 at beginning
    list shouldEqual Seq(I(2), I(1), I(3))
    list.size shouldEqual 3

  it should "allow prepending a removed last element" in:
    val e1 = I(1); val e2 = I(2); val e3 = I(3)
    val list = IntrusiveList[I](e1, e2, e3)
    list.remove(2)
    list.prepend(e3)
    list shouldEqual Seq(I(3), I(1), I(2))
    list.size shouldEqual 3

  "addOne" should "add elements and maintain correct order" in:
    val list = IntrusiveList[I]().addOne(I(1)).addOne(I(2)).addOne(I(3))
    list shouldEqual IntrusiveList[I](I(1), I(2), I(3))

  it should "allow appending a removed middle element" in:
    val e1 = I(1); val e2 = I(2); val e3 = I(3)
    val list = IntrusiveList[I](e1, e2, e3)
    list.remove(1) // remove e2
    list shouldEqual Seq(I(1), I(3))
    list.addOne(e2) // re-add e2 at end
    list shouldEqual Seq(I(1), I(3), I(2))
    list.size shouldEqual 3

  it should "allow appending a removed first element" in:
    val e1 = I(1); val e2 = I(2); val e3 = I(3)
    val list = IntrusiveList[I](e1, e2, e3)
    list.remove(0)
    list.addOne(e1)
    list shouldEqual Seq(I(2), I(3), I(1))
    list.size shouldEqual 3

  "remove" should "remove first element correctly" in:
    val list = IntrusiveList[I](I(1), I(2), I(3))
    list.remove(0)
    list shouldEqual Seq(I(2), I(3))
    list.remove(0)
    list shouldEqual Seq(I(3))
    list.remove(0)
    list shouldEqual Seq()
    list.isEmpty shouldBe true

  it should "remove middle element correctly" in:
    val list = IntrusiveList[I](I(1), I(2), I(3))
    list.remove(1)
    list shouldEqual Seq(I(1), I(3))

  it should "remove last element correctly" in:
    val list = IntrusiveList[I](I(1), I(2), I(3))
    list.remove(2)
    list shouldEqual Seq(I(1), I(2))
    list.remove(1)
    list shouldEqual Seq(I(1))
    list.remove(0)
    list shouldEqual Seq()
    list.isEmpty shouldBe true

  it should "remove first elements correctly" in:
    val list = IntrusiveList[I](I(1), I(2), I(3), I(4), I(5))
    list.remove(0, 2)
    list shouldEqual Seq(I(3), I(4), I(5))
    list.remove(0, 3)
    list shouldEqual Seq()

  it should "remove middle elements correctly" in:
    val list = IntrusiveList[I](I(1), I(2), I(3), I(4), I(5))
    list.remove(1, 2)
    list shouldEqual Seq(I(1), I(4), I(5))
    list.remove(0, 3)
    list shouldEqual Seq()

  it should "remove last elements correctly" in:
    val list = IntrusiveList[I](I(1), I(2), I(3), I(4), I(5))
    list.remove(3, 2)
    list shouldEqual Seq(I(1), I(2), I(3))
    list.remove(0, 3)
    list shouldEqual Seq()

  "size" should "return correct sizes" in:
    IntrusiveList[I]().size shouldEqual 0
    IntrusiveList[I](I(1)).size shouldEqual 1
    IntrusiveList[I](I(1), I(2)).size shouldEqual 2
    IntrusiveList[I](I(1), I(2), I(3)).size shouldEqual 3

  "isEmpty" should "be consistent" in:
    IntrusiveList[I]().isEmpty shouldBe true
    IntrusiveList[I](I(1)).isEmpty shouldBe false
    IntrusiveList[I](I(1), I(2)).isEmpty shouldBe false
    IntrusiveList[I](I(1), I(2), I(3)).isEmpty shouldBe false

  "contains" should "contain added elements" in:
    val list = IntrusiveList[I](I(1), I(3), I(5))
    list.contains(I(1)) shouldBe true
    list.contains(I(2)) shouldBe false
    list.contains(I(3)) shouldBe true
    list.contains(I(4)) shouldBe false
    list.contains(I(5)) shouldBe true

  "clear" should "clear all elements" in:
    val list = IntrusiveList[I](I(1), I(2), I(3), I(4), I(5))
    list.clear()
    list shouldEqual Seq()
    list.isEmpty shouldBe true
    list.length shouldEqual 0

  "apply" should "access first element" in:
    val list = IntrusiveList[I](I(1), I(2), I(3))
    list(0) shouldEqual I(1)

  it should "access middle element" in:
    val list = IntrusiveList[I](I(1), I(2), I(3))
    list(1) shouldEqual I(2)

  it should "access last element" in:
    val list = IntrusiveList[I](I(1), I(2), I(3))
    list(2) shouldEqual I(3)

  it should "throw on out of bounds on negative index" in:
    val list = IntrusiveList[I](I(1), I(2))
    intercept[IndexOutOfBoundsException](list(-1))

  it should "throw on out of bounds empty access" in:
    val list = IntrusiveList[I]()
    intercept[IndexOutOfBoundsException](list(0))

  it should "throw on out of bounds non-empty access" in:
    val list = IntrusiveList[I](I(1), I(2))
    intercept[IndexOutOfBoundsException](list(3))

  "update" should "update first element" in:
    val list = IntrusiveList[I](I(0), I(0), I(0))
    list.update(0, I(1))
    list shouldEqual Seq(I(1), I(0), I(0))

  it should "update middle element" in:
    val list = IntrusiveList[I](I(0), I(0), I(0))
    list.update(1, I(1))
    list shouldEqual Seq(I(0), I(1), I(0))

  it should "update last element" in:
    val list = IntrusiveList[I](I(0), I(0), I(0))
    list.update(2, I(1))
    list shouldEqual Seq(I(0), I(0), I(1))

  it should "update only element" in:
    val list = IntrusiveList[I](I(0))
    list.update(0, I(1))
    list shouldEqual Seq(I(1))

  "insert" should "insert at the beginning" in:
    val list = IntrusiveList[I](I(2), I(3))
    list.insert(0, I(1))
    list shouldEqual Seq(I(1), I(2), I(3))

  it should "insert in the middle" in:
    val list = IntrusiveList[I](I(1), I(3))
    list.insert(1, I(2))
    list shouldEqual Seq(I(1), I(2), I(3))

  it should "insert at the end" in:
    val list = IntrusiveList[I](I(1), I(2))
    list.insert(2, I(3))
    list shouldEqual Seq(I(1), I(2), I(3))

  "insertAll" should "insert at the beginning" in:
    val list = IntrusiveList[I](I(4), I(5))
    list.insertAll(0, Seq(I(1), I(2), I(3)))
    list shouldEqual Seq(I(1), I(2), I(3), I(4), I(5))

  it should "insert in the middle" in:
    val list = IntrusiveList[I](I(1), I(5))
    list.insertAll(1, Seq(I(2), I(3), I(4)))
    list shouldEqual Seq(I(1), I(2), I(3), I(4), I(5))

  it should "insert at the end" in:
    val list = IntrusiveList[I](I(1), I(2))
    list.insertAll(2, Seq(I(3), I(4), I(5)))
    list shouldEqual Seq(I(1), I(2), I(3), I(4), I(5))

  "Multiple operations" should "allow removing and re-adding only element" in:
    val (e1, e2, e3) = (I(1), I(2), I(3))
    val list = IntrusiveList[I](e1, e2, e3)
    list.clear()
    list.addOne(e2)
    list shouldEqual Seq(I(2))
    list.size shouldEqual 1
    e2.prev shouldEqual None
    e2.next shouldEqual None

  "patchInPlace" should "patch at the beginning" in:
    val list = IntrusiveList[I](I(0), I(0), I(0), I(4), I(5))
    list.patchInPlace(0, Seq(I(1), I(2), I(3)), 3)
    list shouldEqual Seq(I(1), I(2), I(3), I(4), I(5))

  it should "patch in the middle" in:
    val list = IntrusiveList[I](I(1), I(0), I(0), I(0), I(5))
    list.patchInPlace(1, Seq(I(2), I(3), I(4)), 3)
    list shouldEqual Seq(I(1), I(2), I(3), I(4), I(5))

  it should "patch at the end" in:
    val list = IntrusiveList[I](I(1), I(2), I(0), I(0), I(0))
    list.patchInPlace(2, Seq(I(3), I(4), I(5)), 3)
    list shouldEqual Seq(I(1), I(2), I(3), I(4), I(5))
