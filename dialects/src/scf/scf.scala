package scair.dialects.scf

import scair.clair.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.utils.*

// ░██████╗ ░█████╗░ ███████╗
// ██╔════╝ ██╔══██╗ ██╔════╝
// ╚█████╗░ ██║░░╚═╝ █████╗░░
// ░╚═══██╗ ██║░░██╗ ██╔══╝░░
// ██████╔╝ ╚█████╔╝ ██║░░░░░
// ╚═════╝░ ░╚════╝░ ╚═╝░░░░░

// ██████╗░ ██╗ ░█████╗░ ██╗░░░░░ ███████╗ ░░██████╗░ ████████╗
// ██╔══██╗ ██║ ██╔══██╗ ██║░░░░░ ██╔════╝ ░██ ╔══██╗ ╚══██╔══╝
// ██║░░██║ ██║ ███████║ ██║░░░░░ █████╗░░ ░██ ║░░╚═╝ ░░░██║░░░
// ██║░░██║ ██║ ██╔══██║ ██║░░░░░ ██╔══╝░░ ░██ ║░░██╗ ░░░██║░░░
// ██████╔╝ ██║ ██║░░██║ ███████╗ ███████╗ ░░██████╔╝ ░░░██║░░░
// ╚═════╝░ ╚═╝ ╚═╝░░╚═╝ ╚══════╝ ╚══════╝ ░░ ╚════╝░ ░░░╚═╝░░░

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||  TYPES AND CONSTRAINTS  ||
\*≡==---==≡≡≡≡≡≡≡≡≡==---==≡*/

// TODO: this needs to be constrained specifically to an I1 integer
type I1 = IntegerType
// TODO: this needs to be a signless integer type specifically
type AnySignlessIntegerOrIndex = IntegerType | IndexType
type Index = IndexType

trait AllTypesMatch(values: Attribute*) extends Operation:

  override def traitVerify(): OK[Operation] =
    if values.isEmpty then OK(this)
    else
      val firstClass = values.head.getClass
      if values.tail.forall(_.getClass == firstClass) then OK(this)
      else
        Err(
          "All parameters of AllTypesMatch must be of the same type in operation " +
            this.name
        )

/*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
||  OPERATION DEFINTION  ||
\*≡==---==≡≡≡≡≡≡≡==---==≡*/

case class Condition(
    condition: Operand[I1],
    args: Seq[Operand[Attribute]],
) extends DerivedOperation["scf.condition"]
    with NoMemoryEffect
    with IsTerminator derives OpDefs

case class ExecuteRegionOp(
    region: Region,
    result: Seq[Result[Attribute]],
) extends DerivedOperation["scf.execute_region"] derives OpDefs

// TODO: this should also contain a SingleBlockImplicitTerminator<"scf::YieldOp">,
case class ForOp(
    lowerBound: Operand[AnySignlessIntegerOrIndex],
    upperBound: Operand[AnySignlessIntegerOrIndex],
    step: Operand[AnySignlessIntegerOrIndex],
    initArgs: Seq[Operand[Attribute]],
    region: Region,
    resultss: Seq[Result[Attribute]],
) extends DerivedOperation["scf.for"]
    with AllTypesMatch(lowerBound.typ, upperBound.typ, step.typ) derives OpDefs

case class ForallOp(
    dynamicLowerBound: Seq[Operand[Index]],
    dynamicUpperBound: Seq[Operand[Index]],
    dynamicStep: Seq[Operand[Index]],
    staticLowerBound: DenseArrayAttr,
    staticUpperBound: DenseArrayAttr,
    staticStep: DenseArrayAttr,
    outputs: Seq[Operand[RankedTensorType]],
    // TODO: Should be array of "DeviceMappingAttribute", but we're not interested yet.
    mapping: Option[ArrayAttribute[Attribute]],
    region: Region,
    resultss: Seq[Result[Attribute]],
) extends DerivedOperation["scf.forall"] derives OpDefs

case class InParallelOp(
    region: Region
) extends DerivedOperation["scf.forall.in_parallel"]
    with IsTerminator
    with NoMemoryEffect derives OpDefs

case class IfOp(
    condition: Operand[I1],
    thenRegion: Region,
    elseRegion: Region,
    resultss: Seq[Result[Attribute]],
) extends DerivedOperation["scf.if"] derives OpDefs

case class ParallelOp(
    lowerBound: Seq[Operand[Index]],
    upperBound: Seq[Operand[Index]],
    step: Seq[Operand[Index]],
    initVals: Seq[Operand[Attribute]],
    region: Region,
    resultss: Seq[Result[Attribute]],
) extends DerivedOperation["scf.parallel"] derives OpDefs

case class ReduceOp(
    operandss: Seq[Operand[Attribute]],
    // TODO: variadic regions
    reductions: Seq[Region],
) extends DerivedOperation["scf.reduce"]
    with AssemblyFormat[
      "(`(` $operandss^ `:` type($operandss) `)`)? $reductions attr-dict"
    ]
    with IsTerminator derives OpDefs

case class ReduceReturnOp(
    result: Operand[Attribute]
) extends DerivedOperation["scf.reduce.return"]
    with AssemblyFormat["$result attr-dict `:` type($result)"]
    with IsTerminator
    with NoMemoryEffect derives OpDefs

case class WhileOp(
    inits: Seq[Operand[Attribute]],
    before: Region,
    after: Region,
    resultss: Seq[Result[Attribute]],
) extends DerivedOperation["scf.while"] derives OpDefs

case class IndexSwitchOp(
    arg: Operand[Index],
    cases: DenseArrayAttr,
    defaultRegion: Region,
    // TODO: variadic regions
    caseRegions: Region,
    resultss: Seq[Result[Attribute]],
) extends DerivedOperation["scf.index_switch"] derives OpDefs

case class YieldOp(
    resultss: Seq[Operand[Attribute]]
) extends DerivedOperation["scf.yield"]
    with AssemblyFormat["attr-dict ($resultss^ `:` type($resultss))?"]
    with IsTerminator
    with NoMemoryEffect derives OpDefs

val SCFDialect =
  summonDialect[EmptyTuple](
    Condition,
    ExecuteRegionOp,
    ForOp,
    ForallOp,
    InParallelOp,
    IfOp,
    ParallelOp,
    ReduceOp,
    ReduceReturnOp,
    WhileOp,
    IndexSwitchOp,
    YieldOp,
  )
