package scair.ir

import fastparse.*
import scair.AttrParser
import scair.Parser.*

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
  def custom_print: String
}

trait TypeAttribute extends Attribute {
  override def prefix: String = "!"
}

// TODO: Think about this; probably not the best design
extension (x: Seq[Attribute] | Attribute)

  def custom_print: String = x match {
    case x: Seq[_] =>
      x.asInstanceOf[Seq[Attribute]]
        .map(_.custom_print)
        .mkString("[", ", ", "]")
    case attr: Attribute => attr.custom_print
  }

abstract class ParametrizedAttribute(
    override val name: String,
    val parameters: Seq[Attribute | Seq[Attribute]] = Seq()
) extends Attribute {

  override def custom_print =
    s"${prefix}${name}${
        if parameters.size > 0 then parameters.map(x => x.custom_print).mkString("<", ", ", ">")
        else ""
      }"

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
  override def custom_print = s"$prefix$name<$data>"

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

trait AttributeObject {
  def name: String
  def parse[$: P](p: AttrParser): P[Attribute]
}

trait AttributeTraitI[T] extends AttributeObject {
  extension (op: T) def AttributeTrait = this
}
