package scair

import scair.ir.*

import java.io.*
import scala.collection.mutable

// ██████╗░ ██████╗░ ██╗ ███╗░░██╗ ████████╗ ███████╗ ██████╗░
// ██╔══██╗ ██╔══██╗ ██║ ████╗░██║ ╚══██╔══╝ ██╔════╝ ██╔══██╗
// ██████╔╝ ██████╔╝ ██║ ██╔██╗██║ ░░░██║░░░ █████╗░░ ██████╔╝
// ██╔═══╝░ ██╔══██╗ ██║ ██║╚████║ ░░░██║░░░ ██╔══╝░░ ██╔══██╗
// ██║░░░░░ ██║░░██║ ██║ ██║░╚███║ ░░░██║░░░ ███████╗ ██║░░██║
// ╚═╝░░░░░ ╚═╝░░╚═╝ ╚═╝ ╚═╝░░╚══╝ ░░░╚═╝░░░ ╚══════╝ ╚═╝░░╚═╝

case class Printer(
    val strictly_generic: Boolean = false,
    val indent: String = "  ",
    var valueNextID: Int = 0,
    var blockNextID: Int = 0,
    val valueNameMap: mutable.Map[Value[? <: Attribute], String] =
      mutable.Map.empty[Value[? <: Attribute], String],
    val blockNameMap: mutable.Map[Block, String] =
      mutable.Map.empty[Block, String],
    private val p: PrintWriter = new PrintWriter(System.out)
) {

  /*≡==--==≡≡≡==--=≡≡*\
  ||      TOOLS      ||
  \*≡==---==≡==---==≡*/

  def assignValueName(value: Value[? <: Attribute]): String =
    val name = valueNameMap.contains(value) match {
      case true => valueNameMap(value)
      case false =>
        val name = valueNextID.toString
        valueNextID = valueNextID + 1
        valueNameMap(value) = name
        name
    }
    return s"%$name"

  def assignBlockName(block: Block): String =
    val name = blockNameMap.contains(block) match {
      case true =>
        blockNameMap(block)
      case false =>
        val name = blockNextID.toString
        blockNextID = blockNextID + 1
        blockNameMap(block) = name
        name
    }
    return s"^bb$name"

  def print(str: String): Unit = p.print(str)

  @deprecated(
    "Just a first way to work with Java's PrintWriter from Scala. Find better!"
  )
  def flush() = p.flush()

  // def print(any : Any) = printOperation()

  /*≡==--==≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
  ||    ATTRIBUTE PRINTER    ||
  \*≡==---==≡≡≡≡≡≡≡≡≡==---==≡*/

  def print(attribute: Attribute): Unit = print(attribute.custom_print)

  /*≡==--==≡≡≡≡≡≡≡==--=≡≡*\
  ||    VALUE PRINTER    ||
  \*≡==---==≡≡≡≡≡==---==≡*/

  def print(value: Value[? <: Attribute]): Unit = print(assignValueName(value))

  /*≡==--==≡≡≡≡≡≡≡==--=≡≡*\
  ||    BLOCK PRINTER    ||
  \*≡==---==≡≡≡≡≡==---==≡*/

  def printArgument(value: Value[? <: Attribute]) =
    print(value)
    print(": ")
    print(value.typ)

  def print(block: Block)(using indentLevel: Int = 0): Unit =
    print(indent * indentLevel)
    print(s"${assignBlockName(block)}(")

    if block.arguments.nonEmpty then
      printArgument(block.arguments.head)
      block.arguments.tail.foreach(a => {
        print(", ")
        printArgument(a)
      })

    print("):\n")
    print(block.operations)(using indentLevel + 1)

  /*≡==--==≡≡≡≡≡≡≡≡==--=≡≡*\
  ||    REGION PRINTER    ||
  \*≡==---==≡≡≡≡≡≡==---==≡*/

  def print(region: Region)(using indentLevel: Int): Unit =
    this.copy()._printRegion(region)

  private def _printRegion(region: Region)(using indentLevel: Int) = {

    print("{\n")
    region.blocks match {
      case Nil             => ()
      case entry +: blocks =>
        // If the entry block has no arguments, we can avoid printing the header
        // Unless it is empty, which would make the next block read as the entry!
        if (entry.arguments.nonEmpty || entry.operations.isEmpty) then
          print(entry)
        else print(entry.operations)(using indentLevel + 1)
        blocks.foreach(block => print(block))
    }
    print(indent * indentLevel + "}")
  }

  /*≡==--==≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
  ||    OPERATION PRINTER    ||
  \*≡==---==≡≡≡≡≡≡≡≡≡==---==≡*/

  def printGenericMLIROperation(op: Operation)(using
      indentLevel: Int
  ) = {
    print(s"\"${op.name}\"(")
    if op.operands.nonEmpty then
      print(op.operands.head)
      for (o <- op.operands.tail)
        print(", ")
        print(o)
    print(")")
    if op.successors.nonEmpty then
      print("[")
      print(assignBlockName(op.successors.head))
      for (s <- op.successors.tail)
        print(", ")
        print(assignBlockName(s))
      print("]")
    if op.properties.nonEmpty then
      print(" <{")
      print(op.properties.head._1)
      print(" = ")
      print(op.properties.head._2.custom_print)
      for ((k, v) <- op.properties.tail)
        print(s", ")
        print(k)
        print(" = ")
        print(v.custom_print)
      print("}>")
    if op.regions.nonEmpty then
      print(" (")
      print(op.regions.head)
      for (r <- op.regions.tail)
        print(", ")
        print(r)
      print(")")
    if op.attributes.nonEmpty then
      print(" {")
      print(op.attributes.head._1)
      print(" = ")
      print(op.attributes.head._2.custom_print)
      for ((k, v) <- op.attributes.tail)
        print(s", ")
        print(k)
        print(" = ")
        print(v.custom_print)
      print("}")
    print(" : (")
    if op.operands.nonEmpty then
      print(op.operands.head.typ.custom_print)
      op.operands.tail.foreach(o => {
        print(", ")
        print(o.typ.custom_print)
      })
    print(") -> (")
    if op.results.nonEmpty then
      print(op.results.head.typ.custom_print)
      op.results.tail.foreach(r => {
        print(", ")
        print(r.typ.custom_print)
      })
    print(")")
  }

  def print(op: Operation)(using indentLevel: Int): Unit = {
    print(indent * indentLevel)
    if op.results.nonEmpty then
      print(op.results.head)
      op.results.tail.foreach(r => {
        print(", ")
        print(r)
      })
      print(" = ")
    if strictly_generic then
      printGenericMLIROperation(
        op
      )
    else op.custom_print(this)

    print("\n")
    p.flush()
  }

  def print(ops: IterableOnce[Operation])(using
      indentLevel: Int
  ): Unit = ops.foreach(print)

}
