package scair.dialects.affine

import scair.dialects.builtin.*
import scair.ir.Attribute
import scair.clair.codegen.*
import scair.clair.mirrored.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.clair.macros._

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
) extends MLIRName["affine.apply"]
    derives MLIRTrait

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
) extends MLIRName["affine.for"]
    derives MLIRTrait

/*≡==---==≡≡≡≡≡==---=≡≡*\
||     PARALLEL OP     ||
\*≡==----==≡≡≡==----==≡*/

case class Parallel(
    map_operands: Variadic[Operand[IndexType.type]],
    // TODO: Should be ArrayAttribute[StringData]
    // Not supported yet
    reductions: Property[Attribute],
    lowerBoundsMap: Property[AffineMapAttr],
    lowerBoundsGroups: Property[DenseIntOrFPElementsAttr],
    upperBoundsMap: Property[AffineMapAttr],
    upperBoundsGroups: Property[DenseIntOrFPElementsAttr],
    res: Variadic[Result[Attribute]],
    body: Region
) extends MLIRName["affine.parallel"]
    derives MLIRTrait

/*≡==--=≡≡≡=--=≡≡*\
||     IF OP     ||
\*≡==---=≡=---==≡*/

case class If(
    args: Variadic[Operand[Attribute]],
    res: Variadic[Result[Attribute]],
    condition: Property[AffineSetAttr],
    then_region: Region,
    else_region: Region
) extends MLIRName["affine.if"]
    derives MLIRTrait

/*≡==--=≡≡≡≡=--=≡≡*\
||    STORE OP    ||
\*≡==--==≡≡==--==≡*/

case class Store(
    value: Operand[Attribute],
    memref: Operand[MemrefType],
    indices: Variadic[Operand[IndexType.type]],
    map: Property[AffineMapAttr]
) extends MLIRName["affine.store"]
    derives MLIRTrait

/*≡==---=≡≡≡=---=≡≡*\
||     LOAD OP     ||
\*≡==----=≡=----==≡*/

case class Load(
    memref: Operand[MemrefType],
    indices: Variadic[Operand[IndexType.type]],
    result: Result[Attribute],
    map: Property[AffineMapAttr]
) extends MLIRName["affine.load"]
    derives MLIRTrait

/*≡==--=≡≡≡≡=--=≡≡*\
||     MIN OP     ||
\*≡==---=≡≡=---==≡*/

case class Min(
    arguments: Variadic[Operand[IndexType.type]],
    result: Result[IndexType.type],
    map: Property[AffineMapAttr]
) extends MLIRName["affine.min"]
    derives MLIRTrait

/*≡==--=≡≡≡≡=--=≡≡*\
||    YIELD OP    ||
\*≡==---=≡≡=---==≡*/

case class Yield(
    arguments: Variadic[Operand[Attribute]]
) extends MLIRName["affine.yield"]
    derives MLIRTrait

val AffineDialect = summonDialect[
  EmptyTuple,
  (Apply, For, Parallel, If, Store, Load, Min, Yield)
](Seq())
