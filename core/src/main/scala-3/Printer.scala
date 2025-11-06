package scair

import scair.ir.*

import java.io.*
import scala.collection.mutable
import scala.collection.mutable.LinkedHashSet

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
      mutable.Map.empty,
    val blockNameMap: mutable.Map[Block, String] = mutable.Map.empty,
    private val p: PrintWriter = new PrintWriter(System.out),
    private var aliasesMap: Map[Attribute, String] = Map.empty
):

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

  def print(str: String): Unit = p.print(str)

  @deprecated(
    "Just a first way to work with Java's PrintWriter from Scala. Find better!"
  )
  def flush() = p.flush()

  /*≡==--==≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
  ||    ATTRIBUTE PRINTER    ||
  \*≡==---==≡≡≡≡≡≡≡≡≡==---==≡*/

  def print(attribute: Attribute): Unit =
    aliasesMap.get(attribute) match
      case Some(alias) => print(alias)
      case None        => attribute.custom_print(this)

  def print(attributes: Seq[Attribute]): Unit =
    printList(attributes, "[", ", ", "]")

  def print(parameter: Attribute | Seq[Attribute]): Unit =
    parameter match
      case seq: Seq[?]     => printList(seq.asInstanceOf[Seq[Attribute]])
      case attr: Attribute => print(attr)

  /*≡==--==≡≡≡≡≡≡≡==--=≡≡*\
  ||    VALUE PRINTER    ||
  \*≡==---==≡≡≡≡≡==---==≡*/

  def print(value: Value[? <: Attribute]): Unit = print(assignValueName(value))

  /*≡==--==≡≡≡≡≡≡≡==--=≡≡*\
  ||    BLOCK PRINTER    ||
  \*≡==---==≡≡≡≡≡==---==≡*/

  type Printable = Value[?] | Block | Region | Operation | Attribute | String

  inline def print(inline thing: Printable)(using
      indentLevel: Int
  ): Unit = thing match
    case s: String    => print(s)
    case v: Value[?]  => print(v)
    case b: Block     => print(b)
    case r: Region    => print(r)
    case o: Operation => print(o)
    case a: Attribute => print(a)

  inline def print(inline things: (Printable | IterableOnce[Printable])*)(using
      indentLevel: Int
  ): Unit =
    things.foreach(_ match
      case p: Printable               => print(p)
      case i: IterableOnce[Printable] =>
        printList(i))

  inline def printList[T <: Printable](
      inline iterable: IterableOnce[T],
      inline start: String = "",
      inline sep: String = ", ",
      inline end: String = ""
  )(using
      indentLevel: Int = 0
  ): Unit =
    printListF(iterable, (x: Printable) => print(x), start, sep, end)

  inline def printListF[T](
      inline iterable: IterableOnce[T],
      f: T => Unit,
      inline start: String = "",
      inline sep: String = ", ",
      inline end: String = ""
  )(using
      indentLevel: Int = 0
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

  def print(region: Region)(using indentLevel: Int): Unit =
    this.copy()._printRegion(region)

  private def _printRegion(region: Region)(using indentLevel: Int) =

    print("{\n")
    region.blocks match
      case Nil             => ()
      case entry +: blocks =>
        // If the entry block has no arguments, we can avoid printing the header
        // Unless it is empty, which would make the next block read as the entry!
        if entry.arguments.nonEmpty || entry.operations.isEmpty then
          print(entry)
        else printList(entry.operations, sep = "")(using indentLevel + 1)
        blocks.foreach(block => print(block))
    print(indent * indentLevel + "}")

  def printAliases(ops: Seq[Operation]) =
    val printer = AliasPrinter(strictly_generic = strictly_generic, p = p)
    printer.print(ops)(using indentLevel = 0)
    val aliasesMap = mutable.Map.empty[Attribute, String]
    val aliasesCounters = mutable.Map.empty[String, Int]
    printer.toAlias.foreach(attr =>
      aliasesMap.getOrElseUpdate(
        attr, {
          val alias = attr.alias
          val counter = aliasesCounters.getOrElseUpdate(alias, 0)
          aliasesCounters(alias) = counter + 1
          val aliasName = attr.prefix + alias + (counter match
            case 0 => ""
            case c => c.toString)
          print(aliasName)
          print(" = ")
          attr.custom_print(this)
          print("\n")
          aliasName
        }
      )
    )
    this.aliasesMap = aliasesMap.toMap

  /*≡==--==≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
  ||    OPERATION PRINTER    ||
  \*≡==---==≡≡≡≡≡≡≡≡≡==---==≡*/

  def printAttrDict(
      attrs: Map[String, Attribute]
  )(using indentLevel: Int = 0): Unit =
    printListF(
      attrs,
      (k, v) => print(k, " = ", v),
      " {",
      ", ",
      "}"
    )

  def printOptionalAttrDict(
      attrs: Map[String, Attribute]
  )(using indentLevel: Int): Unit =
    if attrs.nonEmpty then printAttrDict(attrs)

  def printGenericMLIROperation(op: Operation)(using
      indentLevel: Int
  ) =
    print("\"", op.name, "\"(", op.operands, ")")
    if op.successors.nonEmpty then
      printListF(op.successors, b => print(assignBlockName(b)), "[", ", ", "]")
    if op.properties.nonEmpty then
      printListF(
        op.properties,
        (k, v) => print(k, " = ", v),
        " <{",
        ", ",
        "}>"
      )
    if op.regions.nonEmpty then printList(op.regions, " (", ", ", ")")
    printOptionalAttrDict(op.attributes.toMap)
    print(" : ")
    printListF(
      op.operands,
      o => print(o.typ),
      "(",
      ", ",
      ")"
    )
    print(" -> ")
    if op.results.length == 1 then print(op.results.head.typ)
    else
      printListF(
        op.results,
        r => print(r.typ),
        "(",
        ", ",
        ")"
      )

  def print(op: Operation)(using indentLevel: Int = 0): Unit =
    print(indent * indentLevel)
    if op.results.nonEmpty then
      printList(op.results)
      print(" = ")
    if strictly_generic then
      printGenericMLIROperation(
        op
      )
    else op.custom_print(this)

    print("\n")
    flush()

  def printTopLevel(op: Operation): Unit =
    printTopLevel(Seq(op)): Unit

  def printTopLevel(ops: Seq[Operation]): Unit =
    printAliases(ops)
    print(ops)(using indentLevel = 0)

class AliasPrinter(
    strictly_generic: Boolean = false,
    private val p: PrintWriter = new PrintWriter(System.out),
    val toAlias: mutable.LinkedHashSet[AliasedAttribute] =
      mutable.LinkedHashSet.empty
) extends Printer(strictly_generic = strictly_generic, p = p):

  override def copy(
      strictly_generic: Boolean = false,
      indent: String = "  ",
      valueNextID: Int = 0,
      blockNextID: Int = 0,
      valueNameMap: mutable.Map[Value[? <: Attribute], String] =
        mutable.Map.empty,
      blockNameMap: mutable.Map[Block, String] = mutable.Map.empty,
      p: PrintWriter = new PrintWriter(System.out),
      aliasesMap: Map[Attribute, String] = Map.empty
  ): AliasPrinter =
    new AliasPrinter(
      strictly_generic = strictly_generic,
      p = p,
      toAlias = toAlias
    )

  override def print(string: String) = ()

  override def print(attribute: Attribute): Unit =
    attribute.custom_print(this)
    attribute match
      case aliased: AliasedAttribute => toAlias.add(aliased)
      case _                         => ()
