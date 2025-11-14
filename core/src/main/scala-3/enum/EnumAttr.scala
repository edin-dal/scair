package scair.enums.enumattr

import scair.dialects.builtin.I32
import scair.dialects.builtin.I64
import scair.dialects.builtin.IntData
import scair.dialects.builtin.IntegerAttr
import scair.ir.IntegerEnumAttr

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
