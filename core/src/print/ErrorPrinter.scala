package scair.print

import scair.ir.*
import scair.utils.Err

import java.io.PrintWriter
import java.io.Writer
import scala.annotation.tailrec
import scala.collection.mutable

private final class ErrorPrinterFilter(writer: Writer)
    extends PrintWriter(writer):

  final val currentColumn = mutable.ListBuffer(0)

  var msg: Option[(String, Int, Int)] = None

  @tailrec
  def writeRec(str: String, start: Int): Unit =
    str.indexOf('\n', start) match
      case -1 =>
        currentColumn(currentColumn.length - 1) += str.length - start
        super.write(str, start, str.length - start)
      case i =>
        currentColumn(currentColumn.length - 1) += i - start
        currentColumn += 0
        super.write(str, start, i - start + 1)
        msg.map(printMessage)
        writeRec(str, i + 1)

  def withUnderlinedMessage(toPrint: => Unit, message: String): Unit =
    val startColumn = lastLength
    toPrint
    val end = lastLength
    currentColumn.last match
      case 0 => // new line, just print the message
        printMessage(message, startColumn, end)
      case _ => // Otherwise print it after the current line
        msg = Some((message, startColumn, end))

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
    writeRec(str, 0)

final class ErrorPrinter(
    error: Err,
    w: ErrorPrinterFilter = ErrorPrinterFilter(new PrintWriter(System.out)),
    _indent: String = "  ",
    _valueNextID: Int = 0,
    _blockNextID: Int = 0,
    _valueNameMap: mutable.Map[Value[? <: Attribute], String] = mutable.Map
      .empty,
    _blockNameMap: mutable.Map[Block, String] = mutable.Map.empty,
    _aliasesMap: Map[Attribute, String] = Map.empty,
    _indentLevel: Int = 0,
) extends AssemblyPrinter(
      true,
      _indent,
      _valueNextID,
      _blockNextID,
      _valueNameMap,
      _blockNameMap,
      w,
      _aliasesMap,
      _indentLevel,
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

  override def print(operation: Operation): Unit =
    if obj eq operation then
      w.withUnderlinedMessage(super.print(operation), error.msg)
    else super.print(operation)

  override def print(attribute: Attribute): Unit =
    if obj eq attribute then
      w.withUnderlinedMessage(super.print(attribute), error.msg)
    else super.print(attribute)

  override def print(value: Value[? <: Attribute]): Unit =
    if obj eq value then w.withUnderlinedMessage(super.print(value), error.msg)
    else super.print(value)
