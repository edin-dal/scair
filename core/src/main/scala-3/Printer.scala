package scair

import scair.ir.*

import scala.collection.mutable

import java.io.*
import scair.dialects.builtin.ModuleOp.parse

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
    val p : PrintWriter = new PrintWriter(System.out)
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

  /*≡==--==≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
  ||    ATTRIBUTE PRINTER    ||
  \*≡==---==≡≡≡≡≡≡≡≡≡==---==≡*/

  def printAttribute(attribute: Attribute) = {
    p.print(attribute.custom_print)
  }

  /*≡==--==≡≡≡≡≡≡≡==--=≡≡*\
  ||    VALUE PRINTER    ||
  \*≡==---==≡≡≡≡≡==---==≡*/

  def printValue(value: Value[? <: Attribute]) = {
    p.print(s"${assignValueName(value)}")
  }
  /*≡==--==≡≡≡≡≡≡≡==--=≡≡*\
  ||    BLOCK PRINTER    ||
  \*≡==---==≡≡≡≡≡==---==≡*/

  def printBlockArgument(value: Value[? <: Attribute]) = {
    printValue(value)
    p.print(": ")
    printAttribute(value.typ)
  }

  def printBlock(block: Block, indentLevel: Int = 0) : Unit = {
    p.print(indent * indentLevel)
    p.print(s"${assignBlockName(block)}(")

    if block.arguments.nonEmpty then
      printBlockArgument(block.arguments.head)
      block.arguments.tail.foreach(a => {
        p.print(", ")
        printBlockArgument(a)
      })

    p.print("):\n")
    printOperations(block.operations.toSeq, indentLevel + 1)
  }

  /*≡==--==≡≡≡≡≡≡≡≡==--=≡≡*\
  ||    REGION PRINTER    ||
  \*≡==---==≡≡≡≡≡≡==---==≡*/

  def printRegion(region: Region, indentLevel: Int = 0) =
    this.copy()._printRegion(region, indentLevel)

  private def _printRegion(region: Region, indentLevel: Int = 0) = {

    p.print("{\n")
    region.blocks match {
      case Nil => ()
      case entry +: blocks =>
          // If the entry block has no arguments, we can avoid printing the header
          // Unless it is empty, which would make the next block read as the entry!
          if (entry.arguments.nonEmpty || entry.operations.isEmpty) then
            printBlock(entry, indentLevel)
          else
            printOperations(entry.operations.toSeq, indentLevel + 1)
          blocks.foreach(block => printBlock(block, indentLevel))
    }    
    p.print(indent * indentLevel + "}")
  }

  /*≡==--==≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
  ||    OPERATION PRINTER    ||
  \*≡==---==≡≡≡≡≡≡≡≡≡==---==≡*/

  def printGenericMLIROperation(
      op: Operation,
      indentLevel: Int = 0
  ) = {
    p.print(s"\"${op.name}\"(")
    if op.operands.nonEmpty then
      printValue(op.operands.head)
      for (o <- op.operands.tail)
        p.print(", ")
        printValue(o)
    p.print(")")
    if op.successors.nonEmpty then
      p.print("[")
      p.print(assignBlockName(op.successors.head))
      for (s <- op.successors.tail)
        p.print(", ")
        p.print(assignBlockName(s))
      p.print("]")
    if op.properties.nonEmpty then
      p.print(" <{")
      p.print(op.properties.head._1)
      p.print(" = ")
      p.print(op.properties.head._2.custom_print)
      for ((k, v) <- op.properties.tail)
        p.print(s", ")
        p.print(k)
        p.print(" = ")
        p.print(v.custom_print)
      p.print("}>")
    if op.regions.nonEmpty then
      p.print(" (")
      printRegion(op.regions.head, indentLevel)
      for (r <- op.regions.tail)
        p.print(", ")
        printRegion(r, indentLevel)
      p.print(")")
    if op.attributes.nonEmpty then
      p.print(" {")
      p.print(op.attributes.head._1)
      p.print(" = ")
      p.print(op.attributes.head._2.custom_print)
      for ((k, v) <- op.attributes.tail)
        p.print(s", ")
        p.print(k)
        p.print(" = ")
        p.print(v.custom_print)
      p.print("}")
    p.print(" : (")
    if op.operands.nonEmpty then
      p.print(op.operands.head.typ.custom_print)
      op.operands.tail.foreach(o => {
        p.print(", ")
        p.print(o.typ.custom_print)
      })
    p.print(") -> (")
    if op.results.nonEmpty then
      p.print(op.results.head.typ.custom_print)
      op.results.tail.foreach(r => {
        p.print(", ")
        p.print(r.typ.custom_print)
      })
    p.print(")")
  }

  def printOperation(op: Operation, indentLevel: Int = 0) : Unit = {
    p.print(indent * indentLevel)
    if op.results.nonEmpty then
      printValue(op.results.head)
      op.results.tail.foreach(r => {
        p.print(", ")
        printValue(r)
      })
      p.print(" = ")
    if strictly_generic then
      printGenericMLIROperation(
        op,
        indentLevel
      )
    else
      op.custom_print(this)
    
    p.print("\n")
    p.flush()
  }

  def printOperations(
      ops: Seq[Operation],
      indentLevel: Int = 0
  ) = {
    for { op <- ops }
      printOperation(op, indentLevel)
  }

}
