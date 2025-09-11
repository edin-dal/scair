package scair.dialects.irdl

import scair.clair.macros.*
import scair.dialects.builtin.*
import scair.ir.*

final case class Dialect(
    sym_name: StringData,
    body: Region
) extends DerivedOperation["irdl.dialect", Dialect]

final case class Operation(
    sym_name: StringData,
    body: Region
) extends DerivedOperation["irdl.operation", Operation]

final case class Attribute(
    sym_name: StringData,
    body: Region
) extends DerivedOperation["irdl.attribute", Attribute]

final case class Type(
    sym_name: StringData,
    body: Region
) extends DerivedOperation["irdl.type", Type]

final case class Parameters(
    args: Seq[Operand[AttributeType]],
    names: ArrayAttribute[StringData]
) extends DerivedOperation["irdl.parameters", Parameters]

final case class Operands(
    args: Seq[Operand[AttributeType]],
    names: ArrayAttribute[StringData]
) extends DerivedOperation["irdl.operands", Operands]

final case class Results(
    args: Seq[Operand[AttributeType]],
    names: ArrayAttribute[StringData]
) extends DerivedOperation["irdl.results", Results]

final case class AttributeType()
    extends DerivedAttribute["irdl.attribute", AttributeType]

final case class RegionType()
    extends DerivedAttribute["irdl.region", RegionType]

final case class Any(
    output: Result[AttributeType]
) extends DerivedOperation["irdl.any", Any]

val IRDL = summonDialect[
  (AttributeType, RegionType),
  (Dialect, Operation, Attribute, Type, Parameters, Operands, Results, Any)
]()
