import org.scalatest._
import verbs._
import matchers._
import flatspec._

import scair.macros.{popPF, andThen, andThenImpl, code}

inline def p1(x:Int) = x+1
class PartialFunctionTest extends AnyFlatSpec with should.Matchers:

    "popPF" should "pop a PF" in:
        popPF(0) should be("Hello world!")
        popPF("anything") should be("Hello world!")

    "andThen" should "do its thing" in:
        code(andThen(1, _+1)) should be("(2: scala.Int)")
        code(andThen(2, p1)) should be("((3: scala.Int): scala.Int)")
        
        val b : Boolean = true
        code(andThen(if b then 1 else 0, _ + 10)) should be("(if (b) 11 else 10: scala.Int)")