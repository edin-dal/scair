package scair.ir

import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*
import scair.dialects.builtin.*
import scair.dialects.test.TestOp

import scala.collection.mutable.ListBuffer

class WalkTest extends AnyFlatSpec:

  //   root
  //     inner1
  //       leaf1
  //       leaf2
  //     inner2
  //       leaf3
  //   sibling
  def tree() =
    val leaf1 = TestOp(attributes = Map("name" -> StringData("leaf1")))
    val leaf2 = TestOp(attributes = Map("name" -> StringData("leaf2")))
    val leaf3 = TestOp(attributes = Map("name" -> StringData("leaf3")))
    val inner1 = TestOp(
      attributes = Map("name" -> StringData("inner1")),
      regions = Seq(Region(Seq(Block(Seq(leaf1, leaf2))))),
    )
    val inner2 = TestOp(
      attributes = Map("name" -> StringData("inner2")),
      regions = Seq(Region(Seq(Block(Seq(leaf3))))),
    )
    val root = TestOp(
      attributes = Map("name" -> StringData("root")),
      regions = Seq(Region(Seq(Block(Seq(inner1, inner2))))),
    )
    val sibling = TestOp(attributes = Map("name" -> StringData("sibling")))
    val block = Block(Seq(root, sibling))
    (block, root, inner1, inner2, leaf1, leaf2, leaf3, sibling)

  def name(op: Operation) =
    op.attributes("name").asInstanceOf[StringData].stringLiteral

  "walk" should "visit all operations in pre-order" in:
    val (block, _, _, _, _, _, _, _) = tree()
    val visited = ListBuffer.empty[String]
    block.walk(op =>
      visited += name(op)
      WalkResult.Advance
    ) shouldBe WalkResult.Advance
    visited.toSeq shouldEqual Seq(
      "root",
      "inner1",
      "leaf1",
      "leaf2",
      "inner2",
      "leaf3",
      "sibling",
    )

  it should "not descend into skipped operations" in:
    val (block, _, _, _, _, _, _, _) = tree()
    val visited = ListBuffer.empty[String]
    block.walk(op =>
      visited += name(op)
      if name(op) == "inner1" then WalkResult.Skip else WalkResult.Advance
    ) shouldBe WalkResult.Advance
    visited.toSeq shouldEqual
      Seq("root", "inner1", "inner2", "leaf3", "sibling")

  it should "stop on interruption" in:
    val (block, _, _, _, _, _, _, _) = tree()
    val visited = ListBuffer.empty[String]
    block.walk(op =>
      visited += name(op)
      if name(op) == "leaf2" then WalkResult.Interrupt else WalkResult.Advance
    ) shouldBe WalkResult.Interrupt
    visited.toSeq shouldEqual Seq("root", "inner1", "leaf1", "leaf2")

  it should "allow erasing the visited operation" in:
    val (block, root, _, _, _, _, _, sibling) = tree()
    block.walkAll(op =>
      if name(op) == "root" then
        block.detachOp(op)
        op.erase()
    )
    block.operations.toSeq shouldEqual Seq(sibling)

  "walkPostOrder" should "visit all operations in post-order" in:
    val (block, _, _, _, _, _, _, _) = tree()
    val visited = ListBuffer.empty[String]
    block.walkPostOrder(op =>
      visited += name(op)
      WalkResult.Advance
    ) shouldBe WalkResult.Advance
    visited.toSeq shouldEqual Seq(
      "leaf1",
      "leaf2",
      "inner1",
      "leaf3",
      "inner2",
      "root",
      "sibling",
    )

  it should "stop on interruption" in:
    val (block, _, _, _, _, _, _, _) = tree()
    val visited = ListBuffer.empty[String]
    block.walkPostOrder(op =>
      visited += name(op)
      if name(op) == "inner1" then WalkResult.Interrupt else WalkResult.Advance
    ) shouldBe WalkResult.Interrupt
    visited.toSeq shouldEqual Seq("leaf1", "leaf2", "inner1")

  "walkAll" should "visit all operations" in:
    val (_, root, _, _, _, _, _, _) = tree()
    var count = 0
    root.walkAll(_ => count += 1)
    count shouldEqual 6

  "parentOp" should "return the directly containing operation" in:
    val (_, root, inner1, _, leaf1, _, _, sibling) = tree()
    leaf1.parentOp shouldBe Some(inner1)
    inner1.parentOp shouldBe Some(root)
    root.parentOp shouldBe None
    sibling.parentOp shouldBe None

  "findAncestorOp" should "lift operations to a region" in:
    val (_, root, inner1, inner2, leaf1, _, leaf3, sibling) = tree()
    val region = root.regions.head
    region.findAncestorOp(leaf1) shouldBe Some(inner1)
    region.findAncestorOp(leaf3) shouldBe Some(inner2)
    region.findAncestorOp(inner1) shouldBe Some(inner1)
    region.findAncestorOp(root) shouldBe None
    region.findAncestorOp(sibling) shouldBe None
    inner1.regions.head.findAncestorOp(leaf3) shouldBe None

  it should "lift operations to a block" in:
    val (block, root, inner1, _, leaf1, _, _, sibling) = tree()
    block.findAncestorOp(leaf1) shouldBe Some(root)
    block.findAncestorOp(root) shouldBe Some(root)
    block.findAncestorOp(sibling) shouldBe Some(sibling)
    inner1.regions.head.blocks.head.findAncestorOp(root) shouldBe None
