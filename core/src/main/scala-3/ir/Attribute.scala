package scair.ir

import fastparse._

import scair.AttrParser
import scair.Parser._

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
  def custom_verify(): Unit = ()
  def custom_print: String
}

trait TypeAttribute extends Attribute {
  override def prefix: String = "!"
}

// TODO: Think about this; probably not the best design
extension (x: Seq[Attribute] | Attribute)

  def custom_print: String = x match {
    case seq: Seq[Attribute] => seq.map(_.custom_print).mkString("[", ", ", "]")
    case attr: Attribute     => attr.custom_print
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

abstract class DataAttribute[D](
    override val name: String,
    val data: D
) extends Attribute {
  override def custom_print = data.toString

  override def equals(attr: Any): Boolean = {
    attr match {
      case x: DataAttribute[D] =>
        x.name == this.name &&
        x.getClass == this.getClass &&
        x.data == this.data
      case _ => false
    }
  }

    }

    trait AttributeObject {
    def name: String
    type FactoryType = (Seq[Attribute]) => Attribute
    def factory: FactoryType = ???

    def parser[$: P](p: AttrParser): P[Seq[Attribute]] =
        P(("<" ~/ p.Type.rep(sep = ",") ~ ">").orElse(Seq()))

    def parse[$: P](p: AttrParser): P[Attribute] =
        parser(p).map(factory(_))

}