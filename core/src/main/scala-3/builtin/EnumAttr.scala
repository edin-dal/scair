package scair.EnumAttr

import fastparse.*
import scair.dialects.builtin.I32
import scair.dialects.builtin.I64
import scair.dialects.builtin.IntegerType
import scair.ir.*

// ███████╗ ███╗░░██╗ ██╗░░░██╗ ███╗░░░███╗
// ██╔════╝ ████╗░██║ ██║░░░██║ ████╗░████║
// █████╗░░ ██╔██╗██║ ██║░░░██║ ██╔████╔██║
// ██╔══╝░░ ██║╚████║ ██║░░░██║ ██║╚██╔╝██║
// ███████╗ ██║░╚███║ ╚██████╔╝ ██║░╚═╝░██║
// ╚══════╝ ╚═╝░░╚══╝ ░╚═════╝░ ╚═╝░░░░░╚═╝

// ░█████╗░ ████████╗ ████████╗ ██████╗░
// ██╔══██╗ ╚══██╔══╝ ╚══██╔══╝ ██╔══██╗
// ███████║ ░░░██║░░░ ░░░██║░░░ ██████╔╝
// ██╔══██║ ░░░██║░░░ ░░░██║░░░ ██╔══██╗
// ██║░░██║ ░░░██║░░░ ░░░██║░░░ ██║░░██║
// ╚═╝░░╚═╝ ░░░╚═╝░░░ ░░░╚═╝░░░ ╚═╝░░╚═╝

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||  ENUM ATTRIBUTE INHERITANCE  ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

abstract class EnumAttrCase[T <: Attribute](
    val symbol: String,
    val typ: T
) extends ParametrizedAttribute(symbol, Seq(typ)) {
  def parse[$: P]: P[Attribute] = P(symbol.!).map(_ => this)
  override def custom_print = symbol
}

abstract class EnumAttr[T <: Attribute](
    val name: String,
    val cases: Seq[EnumAttrCase[T]],
    val typ: T
) {

  def parser[$: P](seq: Seq[EnumAttrCase[T]]): P[Attribute] = seq match {
    case x +: xs => P(x.parse | parser(xs))
    case Nil     => P(Fail)
  }

  def caseParser[$: P]: P[Attribute] = {
    // we want to order here by length in descending order to ensure
    // we hit all cases:
    //  if "v = P("asc" | "ascc")"
    //  then "parse("ascc", v(_))"
    //  returns "Success("asc")"
    //  but we need "Success("ascc")"
    parser(cases.sortBy(_.symbol.length)(using Ordering[Int].reverse))
  }

}

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
|| SPECIALIZED CASE INHERITANCE ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

abstract class I32EnumAttrCase(override val symbol: String)
    extends EnumAttrCase[IntegerType](symbol, I32)

abstract class I32EnumAttr(
    override val name: String,
    override val cases: Seq[I32EnumAttrCase]
) extends EnumAttr[IntegerType](name, cases, I32)

abstract class I64EnumAttrCase(override val symbol: String)
    extends EnumAttrCase[IntegerType](symbol, I64)

abstract class I64EnumAttr(
    override val name: String,
    override val cases: Seq[I64EnumAttrCase]
) extends EnumAttr[IntegerType](name, cases, I64)
