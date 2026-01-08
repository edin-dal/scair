package scair.ir

import fastparse.*
import scair.Printer
import scair.dialects.builtin.IntegerAttr
import scair.parse.Parser
import scair.utils.OK

import java.io.PrintWriter
import java.io.StringWriter

//
// ░█████╗░ ████████╗ ████████╗ ██████╗░ ██╗ ██████╗░ ██╗░░░██╗ ████████╗ ███████╗
// ██╔══██╗ ╚══██╔══╝ ╚══██╔══╝ ██╔══██╗ ██║ ██╔══██╗ ██║░░░██║ ╚══██╔══╝ ██╔════╝
// ███████║ ░░░██║░░░ ░░░██║░░░ ██████╔╝ ██║ ██████╦╝ ██║░░░██║ ░░░██║░░░ █████╗░░
// ██╔══██║ ░░░██║░░░ ░░░██║░░░ ██╔══██╗ ██║ ██╔══██╗ ██║░░░██║ ░░░██║░░░ ██╔══╝░░
// ██║░░██║ ░░░██║░░░ ░░░██║░░░ ██║░░██║ ██║ ██████╦╝ ╚██████╔╝ ░░░██║░░░ ███████╗
// ╚═╝░░╚═╝ ░░░╚═╝░░░ ░░░╚═╝░░░ ╚═╝░░╚═╝ ╚═╝ ╚═════╝░ ░╚═════╝░ ░░░╚═╝░░░ ╚══════╝
//

/*≡==--==≡≡≡≡==--=≡≡*\
||    ATTRIBUTES    ||
\*≡==---==≡≡==---==≡*/

sealed trait Attribute:
  def name: String
  def prefix: String = "#"
  def customVerify(): OK[Unit] = OK()
  def printParameters(p: Printer): Unit

  def customPrint(p: Printer): Unit =
    given indentLevel: Int = 0
    p.print(prefix, name)
    printParameters(p)

  override def toString(): String =
    val out = StringWriter()
    val p = Printer(p = PrintWriter(out))
    customPrint(p)
    p.flush()
    out.toString()

trait TypeAttribute extends Attribute:
  override def prefix: String = "!"

trait IntegerEnumAttr extends Attribute:
  def ordinalIntAttr: IntegerAttr

  override def printParameters(p: Printer): Unit = ()

  override def customPrint(p: Printer): Unit =
    p.print(ordinalIntAttr)

abstract trait ParametrizedAttribute() extends Attribute:

  def parameters: Seq[Attribute | Seq[Attribute]]

  override def printParameters(p: Printer): Unit =
    if parameters.size > 0 then
      p.printListF(
        parameters,
        p.print,
        "<",
        ", ",
        ">",
      )

  override def equals(attr: Any): Boolean =
    attr match
      case x: ParametrizedAttribute =>
        x.name == this.name && x.getClass == this.getClass &&
        x.parameters == this.parameters
      case _ => false

object DataAttribute:
  // Make all DataAttributes implicitely convertible to their held data.
  given [D]: Conversion[DataAttribute[D], D] = _.data

abstract class DataAttribute[D](
    override val name: String,
    val data: D,
) extends Attribute:

  override def printParameters(p: Printer) =
    p.print("<", data.toString, ">")(using indentLevel = 0)

  override def equals(attr: Any): Boolean =
    attr match
      case x: DataAttribute[?] =>
        x.name == this.name && x.getClass == this.getClass &&
        x.data == this.data
      case _ => false

trait AttributeCompanion[T <: Attribute]:
  def name: String
  def parse[$: P](using Parser): P[T]
  export scair.parse.whitespace

trait AliasedAttribute(val alias: String) extends Attribute
