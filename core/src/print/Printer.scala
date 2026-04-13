package scair.print

import scair.ir.*

import java.io.Writer
import scala.annotation.targetName

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
