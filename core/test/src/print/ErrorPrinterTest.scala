package scair.print

import org.scalatest.*
import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.utils.Err

import java.io.*

class ErrorPrinterTest extends AnyFlatSpec:

  def printError(error: Err, op: Operation): String =
    val out = StringWriter()
    ErrorPrinter(error, out).printTopLevel(op)
    out.toString

  "ErrorPrinter" should "underline the operation the error is attached to" in {
    val op = UnregisteredOperation("test.op")()
    printError(Err("wrong", Some(op)), op) shouldEqual """"test.op"() : () -> ()
        |^^^^^^^^^^^^^^^^^^^^^^
        |> wrong
        |""".stripMargin
  }

  it should "underline a nested operation once" in {
    val op = UnregisteredOperation("test.op")()
    val module = ModuleOp(Region(Block(Seq(op))))
    printError(Err("wrong", Some(op)), module) shouldEqual
      """"builtin.module"() ({
        |  "test.op"() : () -> ()
        |  ^^^^^^^^^^^^^^^^^^^^^^
        |  > wrong
        |}) : () -> ()
        |""".stripMargin
  }

  it should "print the message after the IR if the error is not attached" in {
    val op = UnregisteredOperation("test.op")()
    printError(Err("wrong"), op) shouldEqual """"test.op"() : () -> ()
        |> wrong
        |""".stripMargin
  }

  it should "print the message after the IR if the object is not printed" in {
    val block = Block()
    val op = ModuleOp(Region(block))
    printError(Err("wrong", Some(block)), op) shouldEqual
      """"builtin.module"() ({
        |^bb0:
        |}) : () -> ()
        |> wrong
        |""".stripMargin
  }
