package scair.dialects.affine

import scair.clair.codegen.*
import scair.clair.macros.*
import scair.dialects.builtin.*
import scair.ir.*

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
    mapOperands: Seq[Operand[IndexType.type]],
    res: Result[IndexType.type],
    map: AffineMapAttr
) extends MLIRName["affine.apply"]
    derives MLIRTrait

/*≡==---=≡≡≡≡=---=≡≡*\
||      FOR OP      ||
\*≡==----=≡≡=----==≡*/

case class For(
    lowerBoundOperands: Seq[Operand[IndexType.type]],
    upperBoundOperands: Seq[Operand[IndexType.type]],
    inits: Seq[Operand[Attribute]],
    res: Seq[Result[Attribute]],
    lowerBoundMap: AffineMapAttr,
    upperBoundMap: AffineMapAttr,
    step: IntegerAttr,
    body: Region
) extends MLIRName["affine.for"]
    derives MLIRTrait

/*≡==---==≡≡≡≡≡==---=≡≡*\
||     PARALLEL OP     ||
\*≡==----==≡≡≡==----==≡*/

case class Parallel(
    map_operands: Seq[Operand[IndexType.type]],
    // TODO: Should be ArrayAttribute[StringData]
    // Not supported yet
    reductions: Attribute,
    lowerBoundsMap: AffineMapAttr,
    lowerBoundsGroups: DenseIntOrFPElementsAttr,
    upperBoundsMap: AffineMapAttr,
    upperBoundsGroups: DenseIntOrFPElementsAttr,
    res: Seq[Result[Attribute]],
    body: Region
) extends MLIRName["affine.parallel"]
    derives MLIRTrait

/*≡==--=≡≡≡=--=≡≡*\
||     IF OP     ||
\*≡==---=≡=---==≡*/

case class If(
    args: Seq[Operand[Attribute]],
    res: Seq[Result[Attribute]],
    condition: AffineSetAttr,
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
    indices: Seq[Operand[IndexType.type]],
    map: AffineMapAttr
) extends MLIRName["affine.store"]
    derives MLIRTrait

/*≡==---=≡≡≡=---=≡≡*\
||     LOAD OP     ||
\*≡==----=≡=----==≡*/

case class Load(
    memref: Operand[MemrefType],
    indices: Seq[Operand[IndexType.type]],
    result: Result[Attribute],
    map: AffineMapAttr
) extends MLIRName["affine.load"]
    derives MLIRTrait

/*≡==--=≡≡≡≡=--=≡≡*\
||     MIN OP     ||
\*≡==---=≡≡=---==≡*/

case class Min(
    arguments: Seq[Operand[IndexType.type]],
    result: Result[IndexType.type],
    map: AffineMapAttr
) extends MLIRName["affine.min"]
    derives MLIRTrait

/*≡==--=≡≡≡≡=--=≡≡*\
||    YIELD OP    ||
\*≡==---=≡≡=---==≡*/

case class Yield(
    arguments: Seq[Operand[Attribute]]
) extends MLIRName["affine.yield"]
    derives MLIRTrait

val AffineDialect = summonDialect[
  EmptyTuple,
  (Apply, For, Parallel, If, Store, Load, Min, Yield)
](Seq())
