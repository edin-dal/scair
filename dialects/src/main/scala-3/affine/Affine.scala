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
    mapOperands: Seq[Operand[IndexType]],
    res: Result[IndexType],
    map: AffineMapAttr
) extends DerivedOperation["affine.apply", Apply]
    with NoMemoryEffect derives DerivedOperationCompanion

/*≡==---=≡≡≡≡=---=≡≡*\
||      FOR OP      ||
\*≡==----=≡≡=----==≡*/

case class For(
    lowerBoundOperands: Seq[Operand[IndexType]],
    upperBoundOperands: Seq[Operand[IndexType]],
    inits: Seq[Operand[Attribute]],
    res: Seq[Result[Attribute]],
    lowerBoundMap: AffineMapAttr,
    upperBoundMap: AffineMapAttr,
    step: IntegerAttr,
    body: Region
) extends DerivedOperation["affine.for", For] derives DerivedOperationCompanion

/*≡==---==≡≡≡≡≡==---=≡≡*\
||     PARALLEL OP     ||
\*≡==----==≡≡≡==----==≡*/

case class Parallel(
    map_operands: Seq[Operand[IndexType]],
    steps: Option[ArrayAttribute[IntegerAttr]],
    reductions: Attribute,
    lowerBoundsMap: AffineMapAttr,
    lowerBoundsGroups: DenseIntOrFPElementsAttr,
    upperBoundsMap: AffineMapAttr,
    upperBoundsGroups: DenseIntOrFPElementsAttr,
    res: Seq[Result[Attribute]],
    body: Region
) extends DerivedOperation["affine.parallel", Parallel]
    derives DerivedOperationCompanion

/*≡==--=≡≡≡=--=≡≡*\
||     IF OP     ||
\*≡==---=≡=---==≡*/

case class If(
    args: Seq[Operand[Attribute]],
    res: Seq[Result[Attribute]],
    condition: AffineSetAttr,
    then_region: Region,
    else_region: Region
) extends DerivedOperation["affine.if", If] derives DerivedOperationCompanion

/*≡==--=≡≡≡≡=--=≡≡*\
||    STORE OP    ||
\*≡==--==≡≡==--==≡*/

case class Store(
    value: Operand[Attribute],
    memref: Operand[MemrefType],
    indices: Seq[Operand[IndexType]],
    map: AffineMapAttr
) extends DerivedOperation["affine.store", Store]
    derives DerivedOperationCompanion

/*≡==---=≡≡≡=---=≡≡*\
||     LOAD OP     ||
\*≡==----=≡=----==≡*/

case class Load(
    memref: Operand[MemrefType],
    indices: Seq[Operand[IndexType]],
    result: Result[Attribute],
    map: AffineMapAttr
) extends DerivedOperation["affine.load", Load]
    derives DerivedOperationCompanion

/*≡==--=≡≡≡≡=--=≡≡*\
||     MIN OP     ||
\*≡==---=≡≡=---==≡*/

case class Min(
    arguments: Seq[Operand[IndexType]],
    result: Result[IndexType],
    map: AffineMapAttr
) extends DerivedOperation["affine.min", Min]
    with NoMemoryEffect derives DerivedOperationCompanion

/*≡==--=≡≡≡≡=--=≡≡*\
||    YIELD OP    ||
\*≡==---=≡≡=---==≡*/

case class Yield(
    arguments: Seq[Operand[Attribute]]
) extends DerivedOperation["affine.yield", Yield]
    with IsTerminator
    with NoMemoryEffect derives DerivedOperationCompanion

val AffineDialect = summonDialect[
  EmptyTuple,
  (Apply, For, Parallel, If, Store, Load, Min, Yield)
]
