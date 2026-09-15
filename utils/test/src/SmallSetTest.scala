package scair.utils.test

import scair.collection.SmallSet

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SmallSetTest extends AnyFlatSpec with Matchers:

  behavior of "SmallSet"

  it should "start empty" in:
    val set = SmallSet[Int]()
    set shouldBe empty
    set.size shouldEqual 0
    set.contains(1) shouldBe false

  it should "add elements without duplicates" in:
    val set = SmallSet[Int]()
    set += 1 += 2 += 1 += 3
    set.size shouldEqual 3
    set.toSet shouldEqual Set(1, 2, 3)
    set.contains(2) shouldBe true
    set.contains(4) shouldBe false

  it should "grow past its initial capacity" in:
    val set = SmallSet[Int](initialCapacity = 1)
    set ++= (1 to 5)
    set.toSet shouldEqual Set(1, 2, 3, 4, 5)

  it should "remove elements" in:
    val set = SmallSet[Int]()
    set ++= Seq(1, 2, 3)
    set -= 2
    set.toSet shouldEqual Set(1, 3)
    set -= 42
    set.toSet shouldEqual Set(1, 3)
    set -= 1
    set -= 3
    set shouldBe empty

  it should "filter in place" in:
    val set = SmallSet[Int]()
    set ++= (1 to 6)
    set.filterInPlace(_ % 2 == 0)
    set.toSet shouldEqual Set(2, 4, 6)
    set.filterInPlace(_ => false)
    set shouldBe empty

  it should "iterate and foreach over all elements" in:
    val set = SmallSet[Int]()
    set ++= Seq(3, 1, 2)
    set.iterator.toSet shouldEqual Set(1, 2, 3)
    var sum = 0
    set.foreach(sum += _)
    sum shouldEqual 6

  it should "clear" in:
    val set = SmallSet[Int]()
    set ++= Seq(1, 2)
    set.clear()
    set shouldBe empty
    set += 3
    set.toSet shouldEqual Set(3)

  it should "keep behaving as a set past the threshold" in:
    val set = SmallSet[Int](threshold = 4)
    set ++= (1 to 10)
    set.size shouldEqual 10
    set.toSet shouldEqual (1 to 10).toSet
    set += 5
    set.size shouldEqual 10
    set -= 5
    set.contains(5) shouldBe false
    set.filterInPlace(_ > 8)
    set.toSet shouldEqual Set(9, 10)
    set.clear()
    set shouldBe empty
