package scair.dialects.irdl

import scair.clair.macros.*
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
    body: Region
) extends DerivedOperation.WithCompanion["irdl.dialect", Dialect]
    derives DerivedOperationCompanion

final case class Operation(
    sym_name: StringData,
    body: Region
) extends DerivedOperation.WithCompanion["irdl.operation", Operation]
    derives DerivedOperationCompanion

final case class Attribute(
    sym_name: StringData,
    body: Region
) extends DerivedOperation.WithCompanion["irdl.attribute", Attribute]
    derives DerivedOperationCompanion

final case class Type(
    sym_name: StringData,
    body: Region
) extends DerivedOperation.WithCompanion["irdl.type", Type]
    derives DerivedOperationCompanion

final case class Parameters(
    args: Seq[Operand[AttributeType]],
    names: ArrayAttribute[StringData]
) extends DerivedOperation.WithCompanion["irdl.parameters", Parameters]
    derives DerivedOperationCompanion

final case class Operands(
    args: Seq[Operand[AttributeType]],
    names: ArrayAttribute[StringData]
) extends DerivedOperation.WithCompanion["irdl.operands", Operands]
    derives DerivedOperationCompanion

final case class Results(
    args: Seq[Operand[AttributeType]],
    names: ArrayAttribute[StringData]
) extends DerivedOperation.WithCompanion["irdl.results", Results]
    derives DerivedOperationCompanion

final case class Attributes(
    args: Seq[Operand[AttributeType]],
    attribute_value_names: ArrayAttribute[StringData]
) extends DerivedOperation.WithCompanion["irdl.attributes", Attributes]
    derives DerivedOperationCompanion

final case class AttributeType()
    extends DerivedAttribute["irdl.attribute", AttributeType]

final case class RegionType()
    extends DerivedAttribute["irdl.region", RegionType]

final case class Any(
    output: Result[AttributeType]
) extends DerivedOperation.WithCompanion["irdl.any", Any]
    derives DerivedOperationCompanion

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
      Any
  )
]()
