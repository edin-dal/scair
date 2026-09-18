package scair.ir

import fastparse.*
import scair.dialects.builtin.IntegerAttr
import scair.parse.Parser
import scair.print.AssemblyPrinter
import scair.print.Printer
import scair.utils.OK

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
    val p = AssemblyPrinter(p = out)
    customPrint(p)
    p.flush()
    out.toString()

  /*
   * Return an error message wrapping this attribute. Purposefully shadowing the Err
   * constructor in an Operation's body, to just automatically wrap the error message
   * with the attribute that caused it, without having to explicitly pass 'this' every
   * time.
   */
  def Err(msg: String, obj: Option[AnyRef] = Some(this)) = scair.utils
    .Err(msg, obj)

trait TypeAttribute extends Attribute:
  override def prefix: String = "!"

trait IntegerEnumAttr extends Attribute:
  def ordinalIntAttr: IntegerAttr

  override def printParameters(p: Printer): Unit = ()

  override def customPrint(p: Printer): Unit =
    p.print(ordinalIntAttr)

object EnumAttr:

  inline given enumAttrCompanion[E <: EnumAttr]: AttributeCompanion[E] =
    scair.enums.EnumAttrCompanion[E](scair.enums.enumValues[E].toSeq)

abstract class EnumAttr(val enumName: String, val caseName: String)
    extends Attribute
    with scala.reflect.Enum:

  override def name: String = enumName

  override def printParameters(p: Printer): Unit =
    p.print("<", caseName, ">")

abstract trait ParametrizedAttribute() extends Attribute:

  def parameters: Seq[Attribute]

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
    p.print("<", data.toString, ">")

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
