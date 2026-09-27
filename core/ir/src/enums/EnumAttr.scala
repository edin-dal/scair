package scair.enums

import fastparse.*
import scair.dialects.builtin.I32
import scair.dialects.builtin.I64
import scair.dialects.builtin.IntData
import scair.dialects.builtin.IntegerAttr
import scair.ir.AttributeCompanion
import scair.ir.EnumAttr
import scair.ir.IntegerEnumAttr
import scair.parse.Parser
import scair.parse.bareIdP

// ███████╗ ███╗░░██╗ ██╗░░░██╗ ███╗░░░███╗
// ██╔════╝ ████╗░██║ ██║░░░██║ ████╗░████║
// █████╗░░ ██╔██╗██║ ██║░░░██║ ██╔████╔██║
// ██╔══╝░░ ██║╚████║ ██║░░░██║ ██║╚██╔╝██║
// ███████╗ ██║░╚███║ ╚██████╔╝ ██║░╚═╝░██║
// ╚══════╝ ╚═╝░░╚══╝ ░╚═════╝░ ╚═╝░░░░░╚═╝
//
// ░█████╗░ ████████╗ ████████╗ ██████╗░
// ██╔══██╗ ╚══██╔══╝ ╚══██╔══╝ ██╔══██╗
// ███████║ ░░░██║░░░ ░░░██║░░░ ██████╔╝
// ██╔══██║ ░░░██║░░░ ░░░██║░░░ ██╔══██╗
// ██║░░██║ ░░░██║░░░ ░░░██║░░░ ██║░░██║
// ╚═╝░░╚═╝ ░░░╚═╝░░░ ░░░╚═╝░░░ ╚═╝░░╚═╝

abstract class I32Enum(override val name: String)
    extends IntegerEnumAttr
    with scala.reflect.Enum:
  def ordinalIntAttr: IntegerAttr = IntegerAttr(IntData(this.ordinal), I32)

abstract class I64Enum(override val name: String)
    extends IntegerEnumAttr
    with scala.reflect.Enum:
  def ordinalIntAttr: IntegerAttr = IntegerAttr(IntData(this.ordinal), I64)

class EnumAttrCompanion[E <: EnumAttr](val values: Seq[E])
    extends AttributeCompanion[E]:
  override val name: String = values.head.name

  private val byCaseName: Map[String, E] =
    values.map(value => value.caseName -> value).toMap

  override def parse[$: P](using Parser): P[E] =
    P("<" ~/ bareIdP.flatMap { caseName =>
      byCaseName.get(caseName) match
        case Some(value) => Pass(value)
        case None        =>
          Fail(
            s"expected one of ${values.map(_.caseName).mkString(", ")} " +
              s"for enum $name, but got '$caseName'"
          )
    } ~ ">")
