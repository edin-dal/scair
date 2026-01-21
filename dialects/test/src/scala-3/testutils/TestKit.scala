package scair.dialects.testutils

import scair.ir.*
import scair.Printer
import scair.utils.{Err, OK}
import scair.dialects.builtin.*

import org.scalatest.Assertion
import org.scalatest.Assertions.{fail, succeed}
import org.scalatest.matchers.should.Matchers.*

import java.io.{PrintWriter, StringWriter}
import scala.reflect.ClassTag

object IRTestKit:

  extension [T](ok: OK[T])

    /** Assert OK and show the error message when it isn't. */
    def shouldBeOK(clue: String = ""): Assertion =
      ok match
        case e: Err =>
          fail(
            if clue.isEmpty then s"Expected OK, got Err(${e.msg})"
            else s"$clue: Err(${e.msg})"
          )
        case _ =>
          succeed

  extension (op: Operation)

    /** Verify an op and fail with a useful clue. */
    def shouldVerify(): Assertion =
      val res = op.verify()
      res.shouldBeOK(s"verify failed for op `${op.name}`")

  extension (m: ModuleOp)

    /** Verify a module and fail with a useful clue. */
    def shouldVerify(): Assertion =
      val res = m.verify()
      res.shouldBeOK("verify failed for module")

  // --- printing ---
  def printIR(op: Operation): String =
    val sw = new StringWriter()
    Printer(p = new PrintWriter(sw)).print(op)
    sw.toString.trim

  def assertPrinted(
      s: String,
      includes: Seq[String] = Nil,
      excludes: Seq[String] = Nil,
  ): Assertion =
    includes.foreach(inc => s should include(inc))
    excludes.foreach(exc => s should not include exc)
    succeed

  // --- IR traversal utilities ---
  def countOps[T: ClassTag](root: Operation): Int =
    val cls = summon[ClassTag[T]].runtimeClass
    var n = 0

    def walkOp(o: Operation): Unit =
      if cls.isInstance(o) then n += 1
      o.regions.foreach(walkRegion)

    def walkRegion(r: Region): Unit =
      r.blocks.foreach(b => b.operations.foreach(walkOp))

    walkOp(root)
    n
