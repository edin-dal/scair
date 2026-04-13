package scair.print

import scair.ir.*

import java.io.PrintWriter
import java.io.Writer
import scala.collection.mutable

case class AliasPrinter(
    strictlyGeneric: Boolean = false,
    private val p: Writer = PrintWriter(System.out),
    private val aliasesCounters: mutable.Map[String, Int] = mutable.Map.empty,
    private val aliases: mutable.Map[Attribute, String] = mutable.Map.empty,
) extends Printer(strictlyGeneric, p):

  private val irPrinter =
    AssemblyPrinter(strictlyGeneric = strictlyGeneric, p = p)

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
