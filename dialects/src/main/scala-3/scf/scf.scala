package scair.dialects.scf

import scair.clair.codegen.*
import scair.clair.macros.*
import scair.dialects.builtin.*
import scair.ir.*

type I1 = IntegerType
type AnySignlessIntegerOrIndex = IntegerType | IndexType
type Index = IndexType

case class Condition(
    condition: Operand[I1],
    args: Seq[Operand[Attribute]]
) extends DerivedOperation["scf.condition", Condition]
    derives DerivedOperationCompanion

case class ExecuteRegionOp(
    region: Region,
    result: Seq[Result[Attribute]]
) extends DerivedOperation["scf.execute_region", ExecuteRegionOp]
    derives DerivedOperationCompanion

case class ForOp(
    lowerBound: Operand[AnySignlessIntegerOrIndex],
    upperBound: Operand[AnySignlessIntegerOrIndex],
    step: Operand[AnySignlessIntegerOrIndex],
    initArgs: Operand[Attribute],
    region: Region,
    resultss: Seq[Result[Attribute]]
) extends DerivedOperation["scf.for", ForOp]
    derives DerivedOperationCompanion

case class ForallOp(
    dynamicLowerBound: Operand[Index],
    dynamicUpperBound: Operand[Index],
    dynamicStep: Operand[Index],
    staticLowerBound: Operand[DenseArrayAttr],
    staticUpperBound: Operand[DenseArrayAttr],
    staticStep: Operand[DenseArrayAttr],
    outputs: Seq[Operand[RankedTensorType]],
    // TODO: should be optional
    mapping: Seq[Operand[RankedTensorType]],
    region: Region,
    resultss: Seq[Result[Attribute]]
) extends DerivedOperation["scf.forall", ForallOp]
    derives DerivedOperationCompanion

case class InParallelOp(
    region: Region
) extends DerivedOperation["scf.forall.in_parallel", InParallelOp]
    derives DerivedOperationCompanion

case class IfOp(
    condition: Operand[I1],
    thenRegion: Region,
    elseRegion: Region,
    resultss: Seq[Result[Attribute]]
) extends DerivedOperation["scf.if", IfOp]
    derives DerivedOperationCompanion

case class ParallelOp(
    lowerBound: Seq[Operand[Index]],
    upperBound: Seq[Operand[Index]],
    step: Seq[Operand[Index]],
    initVals: Seq[Operand[Attribute]],
    region: Region,
    resultss: Seq[Result[Attribute]]
) extends DerivedOperation["scf.parallel", ParallelOp]
    derives DerivedOperationCompanion

case class ReduceOp(
    operandss: Seq[Operand[Attribute]],
    // TODO: variadic regions
    reductions: Region
) extends DerivedOperation["scf.reduce", ReduceOp]
    derives DerivedOperationCompanion

case class ReduceReturnOp(
    resultss: Result[Attribute]
) extends DerivedOperation["scf.reduce.return", ReduceReturnOp]
    derives DerivedOperationCompanion

case class WhileOp(
    inits: Seq[Operand[Attribute]],
    before: Region,
    after: Region,
    resultss: Seq[Result[Attribute]]
) extends DerivedOperation["scf.while", WhileOp]
    derives DerivedOperationCompanion

case class IndexSwitchOp(
    arg: Operand[Index],
    cases: Operand[DenseArrayAttr],
    defaultRegion: Region,
    // TODO: variadic regions
    caseRegions: Region,
    resultss: Seq[Result[Attribute]]
) extends DerivedOperation["scf.index_switch", IndexSwitchOp]
    derives DerivedOperationCompanion

case class YieldOp(
    resultss: Seq[Result[Attribute]]
) extends DerivedOperation["scf.yield", YieldOp]
    derives DerivedOperationCompanion

val SCFDialect =
  summonDialect[
    EmptyTuple,
    (
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
        YieldOp
    )
  ](Seq())
