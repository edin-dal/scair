package scair.dialects.affinegen

import scair.clair.mirrored.*
import scair.dialects.builtin.*
import scair.ir.Attribute
import scair.scairdl.irdef.ScaIRDLDialect

// ░█████╗░ ███████╗ ███████╗ ██╗ ███╗░░██╗ ███████╗
// ██╔══██╗ ██╔════╝ ██╔════╝ ██║ ████╗░██║ ██╔════╝
// ███████║ █████╗░░ █████╗░░ ██║ ██╔██╗██║ █████╗░░
// ██╔══██║ ██╔══╝░░ ██╔══╝░░ ██║ ██║╚████║ ██╔══╝░░
// ██║░░██║ ██║░░░░░ ██║░░░░░ ██║ ██║░╚███║ ███████╗
// ╚═╝░░╚═╝ ╚═╝░░░░░ ╚═╝░░░░░ ╚═╝ ╚═╝░░╚══╝ ╚══════╝

// ░█████╗░ ██████╗░ ███████╗ ██████╗░ ░█████╗░ ████████╗ ██╗ ░█████╗░ ███╗░░██╗ ░██████╗
// ██╔══██╗ ██╔══██╗ ██╔════╝ ██╔══██╗ ██╔══██╗ ╚══██╔══╝ ██║ ██╔══██╗ ████╗░██║ ██╔════╝
// ██║░░██║ ██████╔╝ █████╗░░ ██████╔╝ ███████║ ░░░██║░░░ ██║ ██║░░██║ ██╔██╗██║ ╚█████╗░
// ██║░░██║ ██╔═══╝░ ██╔══╝░░ ██╔══██╗ ██╔══██║ ░░░██║░░░ ██║ ██║░░██║ ██║╚████║ ░╚═══██╗
// ╚█████╔╝ ██║░░░░░ ███████╗ ██║░░██║ ██║░░██║ ░░░██║░░░ ██║ ╚█████╔╝ ██║░╚███║ ██████╔╝
// ░╚════╝░ ╚═╝░░░░░ ╚══════╝ ╚═╝░░╚═╝ ╚═╝░░╚═╝ ░░░╚═╝░░░ ╚═╝ ░╚════╝░ ╚═╝░░╚══╝ ╚═════╝░

/*≡==---==≡≡≡≡==---=≡≡*\
||      APPLY OP      ||
\*≡==----==≡≡==----==≡*/

case class Apply(
    mapOperands: Variadic[Operand[IndexType.type]],
    res: Result[IndexType.type],
    map: Property[AffineMapAttr]
) extends OperationFE

/*≡==---=≡≡≡≡=---=≡≡*\
||      FOR OP      ||
\*≡==----=≡≡=----==≡*/

case class For(
    lowerBoundOperands: Variadic[Operand[IndexType.type]],
    upperBoundOperands: Variadic[Operand[IndexType.type]],
    inits: Variadic[Operand[Attribute]],
    res: Variadic[Result[Attribute]],
    lowerBoundMap: Property[AffineMapAttr],
    upperBoundMap: Property[AffineMapAttr],
    step: Property[IntegerAttr],
    body: Region
) extends OperationFE

/*≡==---==≡≡≡≡≡==---=≡≡*\
||     PARALLEL OP     ||
\*≡==----==≡≡≡==----==≡*/

case class Parallel(
    map_operands: Variadic[Operand[IndexType.type]],
    // TODO: Should be ArrayAttribute[StringData]
    // Not supported yet
    reductions: Property[AnyAttribute],
    lowerBoundsMap: Property[AffineMapAttr],
    lowerBoundsGroups: Property[DenseIntOrFPElementsAttr],
    upperBoundsMap: Property[AffineMapAttr],
    upperBoundsGroups: Property[DenseIntOrFPElementsAttr],
    res: Variadic[Result[Attribute]],
    body: Region
) extends OperationFE

/*≡==--=≡≡≡=--=≡≡*\
||     IF OP     ||
\*≡==---=≡=---==≡*/

case class If(
    args: Variadic[Operand[Attribute]],
    res: Variadic[Result[Attribute]],
    condition: Attr[AffineSetAttr],
    then_region: Region,
    else_region: Region
) extends OperationFE

/*≡==--=≡≡≡≡=--=≡≡*\
||    STORE OP    ||
\*≡==--==≡≡==--==≡*/

case class Store(
    value: Operand[Attribute],
    memref: Operand[MemrefType],
    indices: Variadic[Operand[IndexType.type]],
    map: Property[AffineMapAttr]
) extends OperationFE

/*≡==---=≡≡≡=---=≡≡*\
||     LOAD OP     ||
\*≡==----=≡=----==≡*/

case class Load(
    memref: Operand[MemrefType],
    indices: Variadic[Operand[IndexType.type]],
    result: Result[Attribute],
    map: Property[AffineMapAttr]
) extends OperationFE

/*≡==--=≡≡≡≡=--=≡≡*\
||     MIN OP     ||
\*≡==---=≡≡=---==≡*/

case class Min(
    arguments: Variadic[Operand[IndexType.type]],
    result: Result[IndexType.type],
    map: Property[AffineMapAttr]
) extends OperationFE

/*≡==--=≡≡≡≡=--=≡≡*\
||    YIELD OP    ||
\*≡==---=≡≡=---==≡*/

case class Yield(
    arguments: Variadic[Operand[Attribute]]
) extends OperationFE

object AffineGen
    extends ScaIRDLDialect(
      summonDialect[
        (Apply, For, Parallel, If, Store, Load, Min, Yield)
      ]("Affine")
    )
