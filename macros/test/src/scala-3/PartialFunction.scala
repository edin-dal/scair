import org.scalatest.*
import verbs.*
import matchers.*
import flatspec.*

import scair.macros.{popPF, mapInline, code}
import org.scalactic.exceptions.NullArgumentException
import scair.macros.stuff

inline def p1(x: Int) = x + 1

class PartialFunctionTest extends AnyFlatSpec with should.Matchers:

  "popPF" should "pop a PF" in:
    popPF(0) should be("Hello world!")
    popPF("anything") should be("Hello world!")

  "mapInline" should "do its thing" in:
    code(mapInline(1, _ + 1)) should be("(2: scala.Int)")
    code(mapInline(2, p1)) should be("((3: scala.Int): scala.Int)")

    val b: Boolean = true
    val i: Int = 42
    code(mapInline(if b then 1 else 0, _ + 10)) should be(
      "(if (b) 11 else 10: scala.Int)"
    )
    code(
      mapInline(
        b match
          case true  => 1
          case false => 0,
        _ + 100
      )
    ) should be("""
(b match {
  case true =>
    101
  case false =>
    100
}: scala.Int)
""".strip())
    code(
      mapInline(
        b match
          case true  => 1
          case false => i,
        _ + 100
      )
    ) should be("""
(b match {
  case true =>
    101
  case false =>
    i.+(100)
}: scala.Int)
""".strip())
    code(
      mapInline(
        try
          throw new Exception("fail")
          3
        catch
          case _: NullArgumentException => 0
          case _: Exception             => scala.Range(42, 84).sum
        ,
        _ + 100
      )
    ) should be("""
(try {
  throw new scala.Exception("fail")
  103
} catch {
  case _: org.scalactic.exceptions.NullArgumentException =>
    100
  case _: scala.Exception =>
    {
      val _$5: scala.Int = scala.Range.apply(42, 84).sum[scala.Int](scala.math.Numeric.IntIsIntegral)
      _$5.+(100)
    }
}: scala.Int)
""".strip())

    stuff
