package scair.ir

import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*

import scair.dialects.builtin.*
import scair.clair.{DerivedOperation, OpDefs}

import scair.dialects.test.TestOp

case class DerivedCopyOp(
    inputs: Seq[Operand[Attribute]],
    outputs: Seq[Result[Attribute]],
    bodies: Seq[Region],
    label: StringData,
) extends DerivedOperation["test.derived_copy"] derives OpDefs

class DeepCopyTest extends AnyFlatSpec:

  "DerivedOperation.updated" should
    "use supplied regions without detaching the originals" in:
      val originalRegion = Region(Block())
      val replacementRegion = Region(Block())
      val original = DerivedCopyOp(
        Seq(),
        Seq(),
        Seq(originalRegion),
        StringData("original"),
      )

      val updated = original.updated(regions = Seq(replacementRegion))

      (updated.regions.head eq replacementRegion) shouldBe true
      originalRegion.containerOperation shouldBe Some(original)
      replacementRegion.containerOperation shouldBe Some(updated)
      (original.regions.head eq originalRegion) shouldBe true

  "DerivedOperation.deepCopy" should
    "copy nested trees and remap internal SSA values while retaining external captures" in:
      val external = Result(I32)
      val externalProducer = TestOp(results = Seq(external))
      val location = FileLineColLoc("copy.mlir", 4, 2)
      def op(
          inputs: Seq[Value[Attribute]] = Seq(),
          outputs: Seq[Result[Attribute]] = Seq(),
          bodies: Seq[Region] = Seq(),
      ): DerivedCopyOp =
        val result =
          DerivedCopyOp(inputs, outputs, bodies, StringData("preserved"))
        result.attributes = Map("attr" -> I64)
        result.at(location)

      val body = Block(
        I32,
        argument =>
          val producer = op(outputs = Seq(Result(I32)))
          val nestedBody = Block(
            I32,
            nestedArgument =>
              Seq(
                op(inputs =
                  Seq(
                    producer.results.head,
                    argument,
                    nestedArgument,
                    external,
                  )
                )
              ),
          )
          Seq(
            producer,
            op(inputs = Seq(producer.results.head, argument, external)),
            op(inputs = Seq(argument), bodies = Seq(Region(nestedBody))),
          ),
      )
      val original = op(outputs = Seq(Result(I32)), bodies = Seq(Region(body)))
      val copied = original.deepCopy

      def checkTree(source: Operation, copy: Operation): Unit =
        (copy eq source) shouldBe false
        copy shouldBe a[DerivedCopyOp]
        copy.properties shouldBe source.properties
        copy.attributes shouldBe source.attributes
        copy.location shouldBe source.location
        copy.results.size shouldBe source.results.size
        source.results.zip(copy.results).foreach { (oldResult, newResult) =>
          (oldResult eq newResult) shouldBe false
          newResult.typ shouldBe oldResult.typ
          oldResult.owner shouldBe Some(source)
          newResult.owner shouldBe Some(copy)
        }
        copy.regions.size shouldBe source.regions.size
        source.regions.zip(copy.regions).foreach { (oldRegion, newRegion) =>
          (oldRegion eq newRegion) shouldBe false
          oldRegion.containerOperation shouldBe Some(source)
          newRegion.containerOperation shouldBe Some(copy)
          newRegion.blocks.size shouldBe oldRegion.blocks.size
          oldRegion.blocks.zip(newRegion.blocks).foreach {
            (oldBlock, newBlock) =>
              (oldBlock eq newBlock) shouldBe false
              oldBlock.containerRegion shouldBe Some(oldRegion)
              newBlock.containerRegion shouldBe Some(newRegion)
              newBlock.arguments.size shouldBe oldBlock.arguments.size
              oldBlock.arguments.zip(newBlock.arguments).foreach {
                (oldArg, newArg) =>
                  (oldArg eq newArg) shouldBe false
                  newArg.typ shouldBe oldArg.typ
                  oldArg.owner shouldBe Some(oldBlock)
                  newArg.owner shouldBe Some(newBlock)
              }
              newBlock.operations.size shouldBe oldBlock.operations.size
              oldBlock.operations.zip(newBlock.operations).foreach {
                (oldOp, newOp) =>
                  oldOp.containerBlock shouldBe Some(oldBlock)
                  newOp.containerBlock shouldBe Some(newBlock)
                  checkTree(oldOp, newOp)
              }
          }
        }
      checkTree(original, copied)

      val copiedBody = copied.regions.head.blocks.head
      val copiedOps = copiedBody.operations.toSeq
      val copiedDefinition = copiedOps.head.results.head
      val copiedArgument = copiedBody.arguments.head
      copiedOps(1).operands shouldBe
        Seq(copiedDefinition, copiedArgument, external)
      copiedOps(2).operands shouldBe Seq(copiedArgument)
      val copiedNestedBody = copiedOps(2).regions.head.blocks.head
      copiedNestedBody.operations.head.operands shouldBe Seq(
        copiedDefinition,
        copiedArgument,
        copiedNestedBody.arguments.head,
        external,
      )
      body.operations.toSeq(1).operands shouldBe Seq(
        body.operations.head.results.head,
        body.arguments.head,
        external,
      )
      body.operations.toSeq(2).regions.head.blocks.head.operations.head
        .operands shouldBe Seq(
        body.operations.head.results.head,
        body.arguments.head,
        body.operations.toSeq(2).regions.head.blocks.head.arguments.head,
        external,
      )
      external.owner shouldBe Some(externalProducer)

  "Operation.deepCopy" should "deep copy a simple operation" in:
    val a = TestOp(
      properties = Map("prop" -> I32),
      attributes = Map("attr" -> I64),
      results = Seq(Result(I32)),
    )
    val b = a.deepCopy
    a should not be b
    a.attributes.equals(b.attributes) shouldBe true
    a.properties.equals(b.properties) shouldBe true
    a should matchPattern {
      case TestOp(
            operands = Seq(),
            successors = Seq(),
            results = Seq(Result(I32)),
            regions = Seq(),
          ) =>
        ()
    }

  it should "deep copy use-def operations" in:
    val value = Result(I32)
    val prod = TestOp(results = Seq(value))
    val user = TestOp(operands = Seq(value))

    val a = Block(operations = Seq(prod, user))
    val b = a.deepCopy

    a should not be b
    b should matchPattern {
      case Block(
            operations = BlockOperations(
              TestOp(results = Seq(u)),
              TestOp(operands = Seq(v)),
            )
          ) if (u eq v) && !(v eq value) =>
        ()
    }

  it should "deep copy nested operations" in:
    // Children
    val ca0 = TestOp(results = Seq(Result(I32)))
    val ca1 = TestOp(results = Seq(Result(I32)))
    // Parent
    val pa = TestOp(
      regions = Seq(
        Region(Block(argumentsTypes = Seq(I32), operations = Seq(ca0))),
        Region(Block(argumentsTypes = Seq(I32), operations = Seq(ca1))),
      )
    )

    val pb = pa.deepCopy

    pa should not be pb
    pb should matchPattern {
      case TestOp(
            regions = Seq(
              Region(
                Block(
                  operations = BlockOperations(
                    a @ TestOp(results = Seq(Result(I32)))
                  )
                )
              ),
              Region(
                Block(
                  operations = BlockOperations(
                    b @ TestOp(results = Seq(Result(I32)))
                  )
                )
              ),
            )
          )
          if !(a eq ca0) && !(b eq ca1) &&
            !(a.results.head eq ca0.results.head) &&
            !(b.results.head eq ca1.results.head) =>
        ()
    }
