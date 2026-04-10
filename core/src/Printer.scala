package scair

import scair.ir.*

import java.io.*
import scala.annotation.targetName
import scala.collection.mutable

// ██████╗░ ██████╗░ ██╗ ███╗░░██╗ ████████╗ ███████╗ ██████╗░
// ██╔══██╗ ██╔══██╗ ██║ ████╗░██║ ╚══██╔══╝ ██╔════╝ ██╔══██╗
// ██████╔╝ ██████╔╝ ██║ ██╔██╗██║ ░░░██║░░░ █████╗░░ ██████╔╝
// ██╔═══╝░ ██╔══██╗ ██║ ██║╚████║ ░░░██║░░░ ██╔══╝░░ ██╔══██╗
// ██║░░░░░ ██║░░██║ ██║ ██║░╚███║ ░░░██║░░░ ███████╗ ██║░░██║
// ╚═╝░░░░░ ╚═╝░░╚═╝ ╚═╝ ╚═╝░░╚══╝ ░░░╚═╝░░░ ╚══════╝ ╚═╝░░╚═╝

abstract class Printer(strictlyGeneric: Boolean, p: Writer):

  type Printable = Value[?] | Block | Region | Operation | Attribute | String

  def copy: Printer
  def print(str: String): Unit = p.write(str)

  def print(op: Operation): Unit =
    withIndent(())
    if op.results.nonEmpty then
      printList(op.results)
      print(" = ")
    if strictlyGeneric then
      printGenericMLIROperation(
        op
      )
    else op.customPrint(this)

    print("\n")
    flush()

  @deprecated(
    "Just a first way to work with Java's Writer from Scala. Find better!"
  )
  final def flush() = p.flush()

  def print(attribute: Attribute): Unit
  def print(region: Region): Unit
  def print(block: Block): Unit
  def print(value: Value[? <: Attribute]): Unit

  def printGenericMLIROperation(op: Operation): Unit

  def indented(toPrint: => Unit): Unit
  def withIndent(toPrint: => Unit): Unit

  def print(parameter: Attribute | Seq[Attribute]): Unit =
    parameter match
      case seq: Seq[?]     => printList(seq.asInstanceOf[Seq[Attribute]])
      case attr: Attribute => print(attr)

  def printArgument(value: Value[? <: Attribute]) =
    print(value, ": ", value.typ)

  def printAttrDict(
      attrs: Map[String, Attribute]
  ): Unit =
    printListF(
      attrs,
      (k, v) => print(k, " = ", v),
      " {",
      ", ",
      "}",
    )

  def printOptionalAttrDict(
      attrs: Map[String, Attribute]
  ): Unit =
    if attrs.nonEmpty then printAttrDict(attrs)

  @targetName("printVariadicHelper")
  inline def print(
      inline things: (Printable | IterableOnce[Printable])*
  ): Unit =
    things
      .foreach(_ match
        case p: Printable               => print(p)
        case i: IterableOnce[Printable] =>
          printList(i))

  inline def printList[T <: Printable](
      inline iterable: IterableOnce[T],
      inline start: String = "",
      inline sep: String = ", ",
      inline end: String = "",
  ): Unit =
    printListF(iterable, (x: Printable) => print(x), start, sep, end)

  inline def printListF[T](
      inline iterable: IterableOnce[T],
      f: T => Unit,
      inline start: String = "",
      inline sep: String = ", ",
      inline end: String = "",
  ): Unit =
    inline if start != "" then print(start)
    inline if sep == "" then iterable.foreach(f)
    else if iterable.nonEmpty then
      val it = iterable.iterator
      f(it.next())
      it.foreach(e =>
        print(sep)
        f(e)
      )
    inline if end != "" then print(end)

  @targetName("printDispatch")
  def print(thing: Printable): Unit = thing match
    case s: String    => print(s)
    case v: Value[?]  => print(v)
    case b: Block     => print(b)
    case r: Region    => print(r)
    case o: Operation => print(o)
    case a: Attribute => print(a)

case class IRPrinter(
    val strictlyGeneric: Boolean = false,
    val indent: String = "  ",
    var valueNextID: Int = 0,
    var blockNextID: Int = 0,
    val valueNameMap: mutable.Map[Value[? <: Attribute], String] = mutable.Map
      .empty,
    val blockNameMap: mutable.Map[Block, String] = mutable.Map.empty,
    private val p: Writer = new PrintWriter(System.out),
    private var aliasesMap: Map[Attribute, String] = Map.empty,
    private var indentLevel: Int = 0,
) extends Printer(strictlyGeneric, p):

  override def copy: IRPrinter = copy()

  /*≡==--==≡≡≡==--=≡≡*\
  ||      TOOLS      ||
  \*≡==---==≡==---==≡*/

  def assignValueName(value: Value[? <: Attribute]): String =
    val name = valueNameMap.contains(value) match
      case true  => valueNameMap(value)
      case false =>
        val name = valueNextID.toString
        valueNextID = valueNextID + 1
        valueNameMap(value) = name
        name
    return s"%$name"

  def assignBlockName(block: Block): String =
    val name = blockNameMap.contains(block) match
      case true =>
        blockNameMap(block)
      case false =>
        val name = blockNextID.toString
        blockNextID = blockNextID + 1
        blockNameMap(block) = name
        name
    return s"^bb$name"

  /*≡==--==≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
  ||    ATTRIBUTE PRINTER    ||
  \*≡==---==≡≡≡≡≡≡≡≡≡==---==≡*/

  def indented(toPrint: => Unit): Unit =
    indentLevel = indentLevel + 1
    toPrint
    indentLevel = indentLevel - 1

  def withIndent(toPrint: => Unit): Unit =
    for _ <- 0 until indentLevel do print(indent)
    toPrint

  def print(attribute: Attribute): Unit =
    aliasesMap.get(attribute) match
      case Some(alias) => print(alias)
      case None        => attribute.customPrint(this)

  def print(attributes: Seq[Attribute]): Unit =
    printList(attributes, "[", ", ", "]")

  /*≡==--==≡≡≡≡≡≡≡==--=≡≡*\
  ||    VALUE PRINTER    ||
  \*≡==---==≡≡≡≡≡==---==≡*/

  def print(value: Value[? <: Attribute]): Unit = print(assignValueName(value))

  /*≡==--==≡≡≡≡≡≡≡==--=≡≡*\
  ||    BLOCK PRINTER    ||
  \*≡==---==≡≡≡≡≡==---==≡*/

  def print(block: Block): Unit =
    withIndent(print(assignBlockName(block)))
    if block.arguments.nonEmpty then
      printListF(block.arguments, printArgument, "(", ", ", ")")
    print(":\n")
    indented(printList(block.operations, sep = ""))

  /*≡==--==≡≡≡≡≡≡≡≡==--=≡≡*\
  ||    REGION PRINTER    ||
  \*≡==---==≡≡≡≡≡≡==---==≡*/

  def print(region: Region): Unit =
    this.copy()._printRegion(region)

  private def _printRegion(region: Region) =

    print("{\n")
    region.blocks match
      case Nil             => ()
      case entry +: blocks =>
        // If the entry block has no arguments, we can avoid printing the header
        // Unless it is empty, which would make the next block read as the entry!
        if entry.arguments.nonEmpty || entry.operations.isEmpty then
          print(entry)
        else indented(printList(entry.operations, sep = ""))
        blocks.foreach(block => print(block))
    withIndent(print("}"))

  def printAliases(ops: Seq[Operation]) =
    val printer = AliasPrinter(strictlyGeneric = strictlyGeneric, p = p)
    printer.print(ops)
    this.aliasesMap = printer.getAliases

  /*≡==--==≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
  ||    OPERATION PRINTER    ||
  \*≡==---==≡≡≡≡≡≡≡≡≡==---==≡*/

  def printGenericMLIROperation(op: Operation) =
    print("\"", op.name, "\"(", op.operands, ")")
    if op.successors.nonEmpty then
      printListF(op.successors, b => print(assignBlockName(b)), "[", ", ", "]")
    if op.properties.nonEmpty then
      printListF(
        op.properties,
        (k, v) => print(k, " = ", v),
        " <{",
        ", ",
        "}>",
      )
    if op.regions.nonEmpty then printList(op.regions, " (", ", ", ")")
    printOptionalAttrDict(op.attributes.toMap)
    print(" : ")
    printListF(
      op.operands,
      o => print(o.typ),
      "(",
      ", ",
      ")",
    )
    print(" -> ")
    if op.results.length == 1 then print(op.results.head.typ)
    else
      printListF(
        op.results,
        r => print(r.typ),
        "(",
        ", ",
        ")",
      )

  def printTopLevel(op: Operation): Unit =
    printTopLevel(Seq(op)): Unit

  def printTopLevel(ops: Seq[Operation]): Unit =
    printAliases(ops)
    print(ops)

case class AliasPrinter(
    strictlyGeneric: Boolean = false,
    private val p: Writer = new PrintWriter(System.out),
    private val aliasesCounters: mutable.Map[String, Int] = mutable.Map.empty,
    private val aliases: mutable.Map[Attribute, String] = mutable.Map.empty,
) extends Printer(strictlyGeneric, p):

  private val irPrinter = IRPrinter(strictlyGeneric = strictlyGeneric, p = p)

  def getAliases: Map[Attribute, String] =
    aliases.toMap

  override def indented(toPrint: => Unit): Unit = toPrint
  override def withIndent(toPrint: => Unit): Unit = toPrint

  override def printGenericMLIROperation(op: Operation): Unit =
    op.properties.values.foreach(print)
    op.regions.foreach(print)
    op.attributes.values.foreach(print)
    op.operands.typ.foreach(print)
    op.results.typ.foreach(print)

  override def print(region: Region): Unit =
    region.blocks.foreach(print)

  override def print(block: Block): Unit =
    block.arguments.foreach(arg => print(arg.typ))
    block.operations.foreach(print)

  override def print(value: Value[?]): Unit = ()

  override def copy: AliasPrinter = copy()

  override def print(string: String) = ()

  override def print(attribute: Attribute): Unit =
    attribute.customPrint(this)
    attribute match
      case aliased: AliasedAttribute =>
        aliases.getOrElseUpdate(
          attribute, {
            val alias = aliased.alias
            val counter = aliasesCounters.getOrElseUpdate(alias, 0)
            aliasesCounters(alias) = counter + 1
            val aliasName = aliased.prefix + alias +
              (counter match
                case 0 => ""
                case c => c.toString)
            irPrinter.print(aliasName)
            irPrinter.print(" = ")
            aliased.customPrint(irPrinter)
            irPrinter.print("\n")
            aliasName
          },
        )
      case _ => ()
