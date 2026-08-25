package scair.print

import scair.dialects.builtin.UnitAttr
import scair.ir.*

import java.io.Writer
import scala.annotation.targetName
import scala.collection.immutable.ListMap

// ██████╗░ ██████╗░ ██╗ ███╗░░██╗ ████████╗ ███████╗ ██████╗░
// ██╔══██╗ ██╔══██╗ ██║ ████╗░██║ ╚══██╔══╝ ██╔════╝ ██╔══██╗
// ██████╔╝ ██████╔╝ ██║ ██╔██╗██║ ░░░██║░░░ █████╗░░ ██████╔╝
// ██╔═══╝░ ██╔══██╗ ██║ ██║╚████║ ░░░██║░░░ ██╔══╝░░ ██╔══██╗
// ██║░░░░░ ██║░░██║ ██║ ██║░╚███║ ░░░██║░░░ ███████╗ ██║░░██║
// ╚═╝░░░░░ ╚═╝░░╚═╝ ╚═╝ ╚═╝░░╚══╝ ░░░╚═╝░░░ ╚══════╝ ╚═╝░░╚═╝

abstract class Printer(strictlyGeneric: Boolean, p: Writer):

  type Printable = Value[?] | Block | Region | Operation | Attribute | String

  def scoped: Printer
  def print(str: String): Unit = p.write(str)

  def print(op: Operation): Unit =
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

  def printArgument(value: Value[? <: Attribute]) =
    print(value, ": ", value.typ)

  /** Prints a single `key = value` pair of an attribute dictionary. A
    * [[UnitAttr]] value is printed as the bare key, as MLIR does.
    */
  def printAttrEntry(k: String, v: Attribute): Unit =
    v match
      case _: UnitAttr => print(k)
      case _           => print(k, " = ", v)

  def printAttrDict(
      attrs: Map[String, Attribute]
  ): Unit =
    printListF(
      attrs,
      (k, v) => printAttrEntry(k, v),
      " {",
      ", ",
      "}",
    )

  def printBlockBody(block: Block): Unit =
    indented(
      printListF(block.operations, op => withIndent(print(op)), sep = "")
    )

  def printOptionalAttrDict(
      attrs: Map[String, Attribute]
  ): Unit =
    if attrs.nonEmpty then printAttrDict(attrs)

  /** Prints the attribute dictionary of a hand-written custom syntax: the
    * operation's attributes, plus the `names` properties its syntax does not
    * spell out itself, in the order named. This is the rule the `attr-dict`
    * directive follows for declarative formats, which hand-written printers
    * have to apply themselves.
    */
  def printOptionalAttrDict(
      attrs: Map[String, Attribute],
      properties: Map[String, Attribute],
      names: Seq[String],
  ): Unit =
    names.flatMap(name => properties.get(name).map(name -> _)) match
      case Seq()     => printOptionalAttrDict(attrs)
      case unspelled => printAttrDict(ListMap.from(attrs) ++ unspelled)

  @targetName("printVariadicHelper")
  inline def print(
      inline things: (Printable | IterableOnce[Printable])*
  ): Unit =
    things
      .foreach(_ match
        case p: Printable                          => print(p)
        case i: IterableOnce[Printable @unchecked] =>
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
