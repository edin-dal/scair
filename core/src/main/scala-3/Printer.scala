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
    val valueNameMap: mutable.Map[Value[? <: Attribute], String] = mutable.Map
      .empty[Value[? <: Attribute], String],
    val blockNameMap: mutable.Map[Block, String] = mutable.Map
      .empty[Block, String],
    private val p: PrintWriter = new PrintWriter(System.out)
) {

  /*≡==--==≡≡≡==--=≡≡*\
  ||      TOOLS      ||
  \*≡==---==≡==---==≡*/

  def assignValueName(value: Value[? <: Attribute]): String =
    val name = valueNameMap.contains(value) match {
      case true  => valueNameMap(value)
      case false =>
        val name = valueNextID.toString
        valueNextID = valueNextID + 1
        valueNameMap(value) = name
        name
    }
    return s"%$name"

  def assignBlockName(block: Block): String =
    val name = blockNameMap.contains(block) match {
      case true  => blockNameMap(block)
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

  type Printable = Value[?] | Block | Region | Operation | Attribute | String

  inline def print(inline thing: Printable)(using indentLevel: Int): Unit =
    thing match {
      case s: String    => print(s)
      case v: Value[?]  => print(v)
      case b: Block     => print(b)
      case r: Region    => print(r)
      case o: Operation => print(o)
      case a: Attribute => print(a)
    }

  inline def print(
      inline things: (Printable | Iterable[Printable])*
  )(using indentLevel: Int): Unit = {
    things.foreach(
      _ match
        case p: Printable           => print(p)
        case i: Iterable[Printable] => printList(i)
    )
  }

  inline def printList[T <: Printable](
      inline iterable: Iterable[T],
      inline start: String = "",
      inline sep: String = ", ",
      inline end: String = ""
  )(using indentLevel: Int): Unit = {
    printListF(iterable, (x: Printable) => print(x), start, sep, end)
  }

  inline def printListF[T](
      inline iterable: Iterable[T],
      f: T => Unit,
      inline start: String = "",
      inline sep: String = ", ",
      inline end: String = ""
  )(using indentLevel: Int): Unit = {
    inline if start != "" then print(start)
    inline if sep == "" then iterable.foreach(f)
    else if iterable.nonEmpty then
      f(iterable.head)
      iterable.tail.foreach(e =>
        print(sep)
        f(e)
      )
    inline if end != "" then print(end)
  }

  def printArgument(value: Value[? <: Attribute])(using indentLevel: Int) =
    print(value, ": ", value.typ)

  def print(block: Block)(using indentLevel: Int): Unit =
    print(indent * indentLevel, assignBlockName(block))
    printListF(block.arguments, a => printArgument(a), "(", ", ", ")")
    print(":\n")
    printList(block.operations, sep = "")(using indentLevel + 1)

  /*≡==--==≡≡≡≡≡≡≡≡==--=≡≡*\
  ||    REGION PRINTER    ||
  \*≡==---==≡≡≡≡≡≡==---==≡*/

  def print(region: Region)(using indentLevel: Int): Unit = this.copy()
    ._printRegion(region)

  private def _printRegion(region: Region)(using indentLevel: Int) = {

    print("{\n")
    region.blocks match {
      case Nil             => ()
      case entry +: blocks =>
        // If the entry block has no arguments, we can avoid printing the header
        // Unless it is empty, which would make the next block read as the entry!
        if (entry.arguments.nonEmpty || entry.operations.isEmpty) then
          print(entry)
        else printList(entry.operations, sep = "")(using indentLevel + 1)
        blocks.foreach(block => print(block))
    }
    print(indent * indentLevel + "}")
  }

  /*≡==--==≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
  ||    OPERATION PRINTER    ||
  \*≡==---==≡≡≡≡≡≡≡≡≡==---==≡*/

  def printAttrDict(
      attrs: Map[String, Attribute]
  )(using indentLevel: Int): Unit = {
    printListF(
      attrs,
      (k, v) => { print(k, " = ", v.custom_print) },
      " {",
      ", ",
      "}"
    )
  }

  def printOptionalAttrDict(attrs: Map[String, Attribute])(using
      indentLevel: Int
  ): Unit = { if attrs.nonEmpty then printAttrDict(attrs) }

  def printGenericMLIROperation(op: Operation)(using indentLevel: Int) = {
    print("\"", op.name, "\"(", op.operands, ")")
    if op.successors.nonEmpty then
      printListF(op.successors, b => print(assignBlockName(b)), "[", ", ", "]")
    if op.properties.nonEmpty then
      printListF(
        op.properties,
        (k, v) => { print(k, " = ", v.custom_print) },
        " <{",
        ", ",
        "}>"
      )
    if op.regions.nonEmpty then printList(op.regions, " (", ", ", ")")
    printOptionalAttrDict(op.attributes.toMap)
    print(" : ")
    printListF(op.operands, o => { print(o.typ.custom_print) }, "(", ", ", ")")
    print(" -> ")
    printListF(op.results, r => { print(r.typ.custom_print) }, "(", ", ", ")")
  }

  def print(op: Operation)(using indentLevel: Int = 0): Unit = {
    print(indent * indentLevel)
    if op.results.nonEmpty then
      printList(op.results)
      print(" = ")
    if strictly_generic then printGenericMLIROperation(op)
    else op.custom_print(this)

    print("\n")
    flush()
  }

}
