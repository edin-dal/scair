package scair.ir

import fastparse.*
import scair.AttrParser
import scair.Printer

import java.io.PrintWriter
import java.io.StringWriter

// ██╗ ██████╗░
// ██║ ██╔══██╗
// ██║ ██████╔╝
// ██║ ██╔══██╗
// ██║ ██║░░██║
// ╚═╝ ╚═╝░░╚═╝

/*≡==--==≡≡≡≡==--=≡≡*\
||    ATTRIBUTES    ||
\*≡==---==≡≡==---==≡*/

sealed trait Attribute {
  def name: String
  def prefix: String = "#"
  def custom_verify(): Either[String, Unit] = Right(())
  def custom_print(p: Printer): Unit

  override def toString(): String =
    val out = StringWriter()
    val p = Printer(p = PrintWriter(out))
    custom_print(p)
    p.flush()
    out.toString()

}

trait TypeAttribute extends Attribute {
  override def prefix: String = "!"
}

// TODO: Think about this; probably not the best design
extension (x: Seq[Attribute] | Attribute)

  def custom_print(p: Printer): Unit = x match {
    case attr: Attribute => attr.custom_print(p)
    case x: Seq[_]       =>
      p.printList(
        x.asInstanceOf[Seq[Attribute]],
        "[",
        ", ",
        "]"
      )
      x.asInstanceOf[Seq[Attribute]]
        .map(_.custom_print(p))
        .mkString("[", ", ", "]")
  }

abstract class ParametrizedAttribute() extends Attribute {

  def parameters: Seq[Attribute | Seq[Attribute]]

  override def custom_print(p: Printer) =
    p.print(prefix, name)(using indentLevel = 0)
    if parameters.size > 0 then
      p.printListF(
        parameters,
        (x: Attribute | Seq[Attribute]) => x.custom_print(p),
        "<",
        ", ",
        ">"
      )

  override def equals(attr: Any): Boolean = {
    attr match {
      case x: ParametrizedAttribute =>
        x.name == this.name &&
        x.getClass == this.getClass &&
        x.parameters.length == this.parameters.length &&
        (for ((i, j) <- x.parameters zip this.parameters)
          yield i == j).foldLeft(true)((i, j) => i && j)
      case _ => false
    }
  }

}

object DataAttribute {
  // Make all DataAttributes implicitely convertible to their held data.
  given [D]: Conversion[DataAttribute[D], D] = _.data
}

abstract class DataAttribute[D](
    override val name: String,
    val data: D
) extends Attribute {

  override def custom_print(p: Printer) =
    p.print(prefix, name, "<", data.toString, ">")(using indentLevel = 0)

  override def equals(attr: Any): Boolean = {
    attr match {
      case x: DataAttribute[_] =>
        x.name == this.name &&
        x.getClass == this.getClass &&
        x.data == this.data
      case _ => false
    }
  }

}

trait AttributeCompanion {
  def name: String
  def parse[$: P](p: AttrParser): P[Attribute]
}

trait AttributeCompanionI[T] extends AttributeCompanion {
  extension (op: T) def AttributeTrait = this
}
