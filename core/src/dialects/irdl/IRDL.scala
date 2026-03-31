package scair.dialects.irdl

import scair.clair.*
import scair.dialects.builtin.*
import scair.ir.*

// ██╗ ██████╗░ ██████╗░ ██╗░░░░░
// ██║ ██╔══██╗ ██╔══██╗ ██║░░░░░
// ██║ ██████╔╝ ██║░░██║ ██║░░░░░
// ██║ ██╔══██╗ ██║░░██║ ██║░░░░░
// ██║ ██║░░██║ ██████╔╝ ███████╗
// ╚═╝ ╚═╝░░╚═╝ ╚═════╝░ ╚══════╝

final case class Dialect(
    sym_name: StringData,
    body: Region,
) extends DerivedOperation["irdl.dialect"] derives OpDefs

final case class Operation(
    sym_name: StringData,
    body: Region,
) extends DerivedOperation["irdl.operation"] derives OpDefs

final case class Attribute(
    sym_name: StringData,
    body: Region,
) extends DerivedOperation["irdl.attribute"] derives OpDefs

final case class Type(
    sym_name: StringData,
    body: Region,
) extends DerivedOperation["irdl.type"] derives OpDefs

final case class Parameters(
    args: Seq[Operand[AttributeType]],
    names: ArrayAttribute[StringData],
) extends DerivedOperation["irdl.parameters"] derives OpDefs

final case class Operands(
    args: Seq[Operand[AttributeType]],
    names: ArrayAttribute[StringData],
) extends DerivedOperation["irdl.operands"] derives OpDefs

final case class Results(
    args: Seq[Operand[AttributeType]],
    names: ArrayAttribute[StringData],
) extends DerivedOperation["irdl.results"] derives OpDefs

final case class Attributes(
    args: Seq[Operand[AttributeType]],
    attribute_value_names: ArrayAttribute[StringData],
) extends DerivedOperation["irdl.attributes"] derives OpDefs

final case class AttributeType()
    extends DerivedAttribute["irdl.attribute", AttributeType] derives AttrDefs

final case class RegionType()
    extends DerivedAttribute["irdl.region", RegionType] derives AttrDefs

final case class Any(
    output: Result[AttributeType]
) extends DerivedOperation["irdl.any"] derives OpDefs

val IRDL = summonDialect[
  (AttributeType, RegionType),
  (
      Dialect,
      Operation,
      Attribute,
      Type,
      Parameters,
      Operands,
      Attributes,
      Results,
      Any,
  ),
]
