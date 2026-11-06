package scair.passes.mlir_opt

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers.*
import scair.MLContext
import scair.dialects.builtin.*
import scair.ir.*
import scair.transformations.PassFailure

class MLIROptTest extends AnyFlatSpec:

  private def input(): ModuleOp =
    val module = ModuleOp(Region(Seq(Block())))
    module.attributes += "payload" -> StringData("é" * 100000)
    module

  "MLIROpt" should "handle large UTF-8 input and output" in {
    val ctx = MLContext()
    ctx.registerDialect(BuiltinDialect)
    val module = input()
    val output = MLIROpt(executable = "cat").transform(module)(using ctx)
    output.attributes("payload") shouldEqual module.attributes("payload")
  }

  private def unregistered(): ModuleOp =
    ModuleOp(Region(Seq(Block(Seq(UnregisteredOperation("foo.bar")())))))

  it should "accept unregistered operations if the context allows them" in {
    val ctx = MLContext(allowUnregisteredDialect = true)
    ctx.registerDialect(BuiltinDialect)
    val output =
      MLIROpt(executable = "cat").transform(unregistered())(using ctx)
    output.regions.head.blocks.head.operations.head.name shouldEqual "foo.bar"
  }

  it should "reject unregistered operations otherwise" in {
    val ctx = MLContext()
    ctx.registerDialect(BuiltinDialect)
    intercept[PassFailure] {
      MLIROpt(executable = "cat").transform(unregistered())(using ctx)
    }
  }

  it should "report a process failure with large input" in {
    val failure = intercept[PassFailure] {
      MLIROpt(executable = "false").transform(input())(using MLContext())
    }
    failure.getMessage shouldEqual "'false' failed"
  }
