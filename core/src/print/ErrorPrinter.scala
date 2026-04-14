package scair.print

import scair.ir.*
import scair.utils.Err

import java.io.FilterWriter
import java.io.PrintWriter
import java.io.Writer
import scala.annotation.tailrec
import scala.collection.mutable

private class ErrorPrinterFilter(writer: Writer) extends FilterWriter(writer):

  final val currentColumn = mutable.ListBuffer(0)

  @tailrec
  private final def accountLines(str: String, start: Int): Unit =
    str.indexOf('\n', start) match
      case -1 =>
        currentColumn(currentColumn.length - 1) += str.length - start
      case i =>
        currentColumn(currentColumn.length - 1) += i - start
        currentColumn += 0
        accountLines(str, i + 1)

  def lastLength: Int =
    currentColumn.last match
      case 0 =>
        if currentColumn.length == 1 then 0
        else currentColumn(currentColumn.length - 2)
      case l => l

  override def write(str: String): Unit =
    accountLines(str, 0)
    super.write(str)

final class ErrorPrinter(
    error: Err,
    w: ErrorPrinterFilter = ErrorPrinterFilter(new PrintWriter(System.out)),
    indent: String = "  ",
    valueNextID: Int = 0,
    blockNextID: Int = 0,
    valueNameMap: mutable.Map[Value[? <: Attribute], String] = mutable.Map
      .empty,
    blockNameMap: mutable.Map[Block, String] = mutable.Map.empty,
    aliasesMap: Map[Attribute, String] = Map.empty,
    indentLevel: Int = 0,
) extends AssemblyPrinter(
      true,
      indent,
      valueNextID,
      blockNextID,
      valueNameMap,
      blockNameMap,
      w,
      aliasesMap,
      indentLevel,
    ):

  val obj = error.obj.getOrElse(null)

  override def scoped =
    ErrorPrinter(
      error,
      w,
      indent,
      valueNextID,
      blockNextID,
      valueNameMap,
      blockNameMap,
      aliasesMap,
      indentLevel,
    )

  final private def withUnderlinedMessage(toPrint: => Unit): Unit =
    val startColumn = p.asInstanceOf[ErrorPrinterFilter].lastLength
    toPrint
    val endColumn = p.asInstanceOf[ErrorPrinterFilter].lastLength
    1 to startColumn foreach (_ => super.print(" "))
    startColumn + 1 to endColumn foreach (_ => super.print("^"))
    super.print("\n")
    error.msg.linesIterator.foreach(l =>
      1 to startColumn foreach (_ => super.print(" "))
      super.print("> ")
      super.print(l)
      super.print("\n")
    )
    flush()

  override def print(operation: Operation): Unit =
    if obj eq operation then withUnderlinedMessage(super.print(operation))
    else super.print(operation)

  override def print(attribute: Attribute): Unit =
    if obj eq attribute then withUnderlinedMessage(super.print(attribute))
    else super.print(attribute)

  override def print(value: Value[? <: Attribute]): Unit =
    if obj eq value then withUnderlinedMessage(super.print(value))
    else super.print(value)
