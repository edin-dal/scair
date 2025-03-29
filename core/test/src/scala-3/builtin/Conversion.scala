package scair

import scair.core.macros.TransparentData

import scair.dialects.builtin._
import org.scalatest.*
import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*

class ConversionTest extends AnyFlatSpec {
  val str: String = StringData("AsAString")
  str shouldEqual "AsAString"

  val strAttr: StringData = "AsAnAttr"
  strAttr shouldEqual StringData("AsAnAttr")

  val int: Long = IntData(42)
  int shouldEqual 42

  val intAttr: IntData = 24.toLong
  intAttr shouldEqual IntData(24)

  val float: Double = FloatData(42.0)
  float shouldEqual 42.0

  val floatAttr: FloatData = 24.0
  floatAttr shouldEqual FloatData(24.0)
}
