package scair.print

import scair.ir.*

import java.io.PrintWriter
import java.io.Writer
import scala.collection.mutable

case class AssemblyPrinter(
    val strictlyGeneric: Boolean = false,
    val indent: String = "  ",
    var valueNextID: Int = 0,
    var blockNextID: Int = 0,
    val valueNameMap: mutable.Map[Value[? <: Attribute], String] = mutable.Map
      .empty,
    val blockNameMap: mutable.Map[Block, String] = mutable.Map.empty,
    protected val p: Writer = new PrintWriter(System.out),
    private var aliasesMap: Map[Attribute, String] = Map.empty,
    private var indentLevel: Int = 0,
) extends Printer(strictlyGeneric, p):

  protected def writer: Writer = p

  override def scoped: AssemblyPrinter = copy()

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
    printBlockBody(block)

  /*≡==--==≡≡≡≡≡≡≡≡==--=≡≡*\
  ||    REGION PRINTER    ||
  \*≡==---==≡≡≡≡≡≡==---==≡*/

  def print(region: Region): Unit =
    this.scoped._printRegion(region)

  private def _printRegion(region: Region) =

    print("{\n")
    region.blocks match
      case Nil             => ()
      case entry +: blocks =>
        // If the entry block has no arguments, we can avoid printing the header
        // Unless it is empty, which would make the next block read as the entry!
        if entry.arguments.nonEmpty || entry.operations.isEmpty then
          print(entry)
        else printBlockBody(entry)
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
