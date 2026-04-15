package scair.print

import scair.ir.*
import scair.utils.Err

import java.io.PrintWriter
import java.io.Writer
import scala.annotation.tailrec
import scala.collection.mutable

private class ErrorPrinterFilter(writer: Writer) extends PrintWriter(writer):

  final val currentColumn = mutable.ListBuffer(0)

  var msg: Option[(String, Int, Int)] = None

  @tailrec
  private final def writeRec(str: String, start: Int): Unit =
    str.indexOf('\n', start) match
      case -1 =>
        currentColumn(currentColumn.length - 1) += str.length - start
        super.write(str)
      case i =>
        currentColumn(currentColumn.length - 1) += i - start
        currentColumn += 0
        super.write(str, start, i - start)
        msg.map(printMessage)
        writeRec(str, i + 1)

  def nextMessage(content: String, start: Int, end: Int): Unit =
    // scala.Console.println(s"Received message, current: $lastLength")
    currentColumn.last match
      case 0 => // new line, just print the message
        printMessage(content, start, end)
      case _ =>
        msg = Some((content, start, end))

  final def printMessage(content: String, start: Int, end: Int): Unit =
    msg = None
    1 to start foreach (_ => super.write(" "))
    start + 1 to end foreach (_ => super.write("^"))
    super.write("\n")
    content.linesIterator.foreach(l =>
      1 to start foreach (_ => super.write(" "))
      super.write("> ")
      super.write(l)
      super.write("\n")
    )
    flush()

  def lastLength: Int =
    currentColumn.last match
      case 0 =>
        if currentColumn.length == 1 then 0
        else currentColumn(currentColumn.length - 2)
      case l => l

  override def write(str: String): Unit =
    // scala.Console.println(s"Printing: '$str'")
    writeRec(str, 0)
    // scala.Console.println(s"Finished printing, current: $lastLength")

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

  final private def printMessage(content: String, start: Int, end: Int): Unit =
    1 to start foreach (_ => super.print(" "))
    start + 1 to end foreach (_ => super.print("^"))
    super.print("\n")
    error.msg.linesIterator.foreach(l =>
      1 to start foreach (_ => super.print(" "))
      super.print("> ")
      super.print(l)
      super.print("\n")
    )
    flush()

  final private def withUnderlinedMessage(toPrint: => Unit): Unit =
    val startColumn = w.lastLength
    toPrint
    val endColumn = w.lastLength
    w.nextMessage(error.msg, startColumn, endColumn)

  override def print(operation: Operation): Unit =
    if obj eq operation then withUnderlinedMessage(super.print(operation))
    else super.print(operation)

  override def print(attribute: Attribute): Unit =
    if obj eq attribute then withUnderlinedMessage(super.print(attribute))
    else super.print(attribute)

  override def print(value: Value[? <: Attribute]): Unit =
    if obj eq value then withUnderlinedMessage(super.print(value))
    else super.print(value)
