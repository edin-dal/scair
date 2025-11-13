package scair.eenum.enumattr

import fastparse.*
import scair.dialects.builtin.*
import scair.AttrParser
import scair.Printer
import scair.ir.AttributeCompanion
import scair.ir.*
import java.io.PrintWriter
// import scair.eenum.mirror.*

abstract class I32Enum(override val name: String) extends IntegerEnumAttr with scala.reflect.Enum {
  def ordinalIntAttr: IntegerAttr = IntegerAttr(IntData(this.ordinal), I32) 
}

enum Color(name: String) extends I32Enum(name):
  case Red extends Color("red")
  case Green extends Color("green")
  case Blue extends Color("blue")

object ColorTest:

  def what(yo: scala.reflect.Enum): Int = yo.ordinal

  def main(args: Array[String]): Unit =
    val redAttr: Color = Color.Red
    val greenAttr: Color = Color.Green

    // println(what(redAttr))
    // println(Color)

    val p: PrintWriter = new PrintWriter(System.out)
    val printer = new Printer(strictly_generic = true, p = p)
    printer.print(redAttr)

    p.flush()
    
    // println(s"Red attribute: $redAttr")