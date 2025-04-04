package scair.dialects.scf

import scair.clair.codegen.*
import scair.clair.macros.*
import scair.dialects.builtin.*
import scair.ir.*

type I1 = IntegerType
type AnySignlessIntegerOrIndex = IntegerType | IndexType.type
type Index = IndexType.type

case class Condition(
    condition: Operand[I1],
    args: Seq[Operand[Attribute]]
) extends MLIRName["scf.condition"]
    derives MLIRTrait

case class ExecuteRegionOp(
    region: Region,
    result: Seq[Result[Attribute]]
) extends MLIRName["scf.execute_region"]
    derives MLIRTrait

case class ForOp(
    lowerBound: Operand[AnySignlessIntegerOrIndex],
    upperBound: Operand[AnySignlessIntegerOrIndex],
    step: Operand[AnySignlessIntegerOrIndex],
    initArgs: Operand[Attribute],
    region: Region,
    results: Seq[Result[Attribute]]
) extends MLIRName["scf.for"]
    derives MLIRTrait

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
    results: Seq[Result[Attribute]]
) extends MLIRName["scf.forall"]
    derives MLIRTrait

case class InParallelOp(
    region: Region
) extends MLIRName["scf.forall.in_parallel"]
    derives MLIRTrait

case class IfOp(
    condition: Operand[I1],
    thenRegion: Region,
    elseRegion: Region,
    results: Seq[Result[Attribute]]
) extends MLIRName["scf.if"]
    derives MLIRTrait

case class ParallelOp(
    lowerBound: Seq[Operand[Index]],
    upperBound: Seq[Operand[Index]],
    step: Seq[Operand[Index]],
    initVals: Seq[Operand[Attribute]],
    region: Region,
    results: Seq[Result[Attribute]]
) extends MLIRName["scf.parallel"]
    derives MLIRTrait

case class ReduceOp(
    operands: Seq[Operand[Attribute]],
    // TODO: variadic regions
    reductions: Region
) extends MLIRName["scf.reduce"]
    derives MLIRTrait

case class ReduceReturnOp(
    results: Result[Attribute]
) extends MLIRName["scf.reduce.return"]
    derives MLIRTrait

case class WhileOp(
    inits: Seq[Operand[Attribute]],
    before: Region,
    after: Region,
    results: Seq[Result[Attribute]]
) extends MLIRName["scf.while"]
    derives MLIRTrait

case class IndexSwitchOp(
    arg: Operand[Index],
    cases: Operand[DenseArrayAttr],
    defaultRegion: Region,
    // TODO: variadic regions
    caseRegions: Region,
    results: Seq[Result[Attribute]]
) extends MLIRName["scf.index_switch"]
    derives MLIRTrait

case class YieldOp(
    results: Seq[Result[Attribute]]
) extends MLIRName["scf.yield"]
    derives MLIRTrait

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
