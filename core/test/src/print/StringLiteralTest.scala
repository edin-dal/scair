package scair

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers.*
import scair.dialects.builtin.StringData
import scair.print.AssemblyPrinter

import java.io.StringWriter

class StringLiteralTest extends AnyFlatSpec:

  private def printed(value: String): String =
    val out = StringWriter()
    AssemblyPrinter(p = out).print(StringData(value))
    out.toString

  "String literals" should "print one char per byte when all chars fit" in {
    printed("\u00ff\u0000a") shouldBe "\"\\FF\\00a\""
  }

  it should "encode strings with wider chars as UTF-8" in {
    printed("\u20ac") shouldBe "\"\\E2\\82\\AC\""
    printed("\ud83d\ude00") shouldBe "\"\\F0\\9F\\98\\80\""
  }
