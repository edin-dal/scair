package scair.dialects.scf

import scair.clair.codegen.*
import scair.clair.macros.*
import scair.dialects.builtin.*
import scair.ir.*
import javax.print.attribute.standard.Destination

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

/*≡=--==≡≡≡≡==--=≡*\
||   ATTRIBUTES   ||
\*≡=---==≡≡==---=≡*/

type DenseI64ArrayAttr = DenseArrayAttr
type AnyRankedTensor = RankedTensorType
// TODO: Should be array of "DeviceMappingAttribute", but we're not interested yet.
type DeviceMappingArrayAttr = ArrayAttribute[Attribute]

/*≡=--==≡≡≡≡≡==--=≡*\
||   CONSTRAINTS   ||
\*≡=---==≡≡≡==---=≡*/

// TODO: this needs to be constrained specifically to an I1 integer
type I1 = IntegerType
// TODO: this needs to be a signless integer type specifically
type AnySignlessIntegerOrIndex = IntegerType | IndexType
type Index = IndexType
type SizedRegion[V <: Int] = Region
type MaxSizedRegion[V <: Int] = Region
// TODO: Variadic regions
type VariadicRegion[v <: Region] = Region

/*≡=--==≡≡≡≡==--=≡*\
||   INTERFACES   ||
\*≡=---==≡≡==---=≡*/

trait AllTypesMatch(values: Attribute*) extends Operation {

  override def trait_verify(): Either[String, Operation] = {
    if (values.isEmpty) Right(this)
    else {
      val firstClass = values.head.getClass
      if (values.tail.forall(_.getClass == firstClass)) Right(this)
      else
        Left(
          "All parameters of AllTypesMatch must be of the same type in operation " + this.name
        )
    }
  }

}

// def GraphRegionNoTerminator : TraitList<[
//     NoTerminator,
//     SingleBlock,
//     RegionKindInterface,
//     HasOnlyGraphRegion
//   ]>;

trait HasParent[P <: Operation]
trait Pure
trait RegionBranchTerminatorOpInterface
trait RegionBranchOpInterface
trait AutomaticAllocationScope
trait LoopLikeOpInterface
trait ConditionallySpeculable
trait SingleBlockImplicitTerminator[T <: Operation]
trait RecursiveMemoryEffects
trait DestinationStyleOpInterface
trait HasParallelRegion
trait AttrSizedOperandSegments
trait ParallelCombiningOpInterface
trait RecursivelySpeculatable
trait NoRegionArguments
trait InferTypeOpAdaptor
trait SingleBlock
trait ReturnLike
trait ParentOneOf[Operations]

/*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
||  OPERATION DEFINTION  ||
\*≡==---==≡≡≡≡≡≡≡==---==≡*/

case class Condition(
    val condition: Operand[I1],
    val args: Seq[Operand[Attribute]]
) extends DerivedOperation["scf.condition", Condition],
      NoMemoryEffect,
      HasParent[WhileOp],
      RegionBranchTerminatorOpInterface,
      IsTerminator derives DerivedOperationCompanion

case class ExecuteRegionOp(
    val region: Region,
    val result: Seq[Result[TypeAttribute]]
) extends DerivedOperation["scf.execute_region", ExecuteRegionOp],
      RegionBranchOpInterface derives DerivedOperationCompanion

case class ForOp(
    val lowerBound: Operand[AnySignlessIntegerOrIndex],
    val upperBound: Operand[AnySignlessIntegerOrIndex],
    val step: Operand[AnySignlessIntegerOrIndex],
    val initArgs: Seq[Operand[TypeAttribute]],
    val _results: Seq[Result[TypeAttribute]],
    val region: SizedRegion[1]
) extends DerivedOperation["scf.for", ForOp],
      AutomaticAllocationScope,
      LoopLikeOpInterface,
      AllTypesMatch(lowerBound.typ, upperBound.typ, step.typ),
      ConditionallySpeculable,
      RegionBranchOpInterface,
      SingleBlockImplicitTerminator[YieldOp],
      RecursiveMemoryEffects derives DerivedOperationCompanion

case class ForallOp(
    val dynamicLowerBound: Seq[Operand[Index]],
    val dynamicUpperBound: Seq[Operand[Index]],
    val dynamicStep: Seq[Operand[Index]],
    val staticLowerBound: DenseI64ArrayAttr,
    val staticUpperBound: DenseI64ArrayAttr,
    val staticStep: DenseI64ArrayAttr,
    val outputs: Seq[Operand[AnyRankedTensor]],
    val mapping: Option[DeviceMappingArrayAttr],
    val _results: Seq[Result[TypeAttribute]],
    val region: SizedRegion[1]
) extends DerivedOperation["scf.forall", ForallOp],
      AttrSizedOperandSegments,
      AutomaticAllocationScope,
      LoopLikeOpInterface,
      RecursiveMemoryEffects,
      SingleBlockImplicitTerminator[InParallelOp],
      RegionBranchOpInterface,
      DestinationStyleOpInterface,
      HasParallelRegion derives DerivedOperationCompanion

// TODO: GraphRegionNoTerminator
case class InParallelOp(
    val region: SizedRegion[1]
) extends DerivedOperation["scf.forall.in_parallel", InParallelOp],
      IsTerminator,
      NoMemoryEffect,
      ParallelCombiningOpInterface,
      HasParent[ForallOp] derives DerivedOperationCompanion

case class IfOp(
    val condition: Operand[I1],
    val thenRegion: SizedRegion[1],
    val elseRegion: MaxSizedRegion[1],
    val _results: Seq[Result[TypeAttribute]]
) extends DerivedOperation["scf.if", IfOp],
      RegionBranchOpInterface,
      InferTypeOpAdaptor,
      SingleBlockImplicitTerminator[YieldOp],
      RecursiveMemoryEffects,
      RecursivelySpeculatable,
      NoRegionArguments derives DerivedOperationCompanion

case class ParallelOp(
    val lowerBound: Seq[Operand[Index]],
    val upperBound: Seq[Operand[Index]],
    val step: Seq[Operand[Index]],
    val initVals: Seq[Operand[TypeAttribute]],
    val _results: Seq[Result[TypeAttribute]],
    val region: SizedRegion[1]
) extends DerivedOperation["scf.parallel", ParallelOp],
      AutomaticAllocationScope,
      AttrSizedOperandSegments,
      LoopLikeOpInterface,
      RecursiveMemoryEffects,
      RegionBranchOpInterface,
      SingleBlockImplicitTerminator[ReduceOp],
      HasParallelRegion derives DerivedOperationCompanion

case class ReduceOp(
    val _operands: Seq[Operand[Attribute]],
    val reductions: VariadicRegion[SizedRegion[1]]
) extends DerivedOperation["scf.reduce", ReduceOp],
      HasParent[ParallelOp],
      RecursiveMemoryEffects,
      RegionBranchTerminatorOpInterface,
      IsTerminator derives DerivedOperationCompanion

case class ReduceReturnOp(
    val _results: Result[TypeAttribute]
) extends DerivedOperation["scf.reduce.return", ReduceReturnOp],
      HasParent[ReduceOp],
      NoMemoryEffect,
      IsTerminator derives DerivedOperationCompanion

case class WhileOp(
    val inits: Seq[Operand[TypeAttribute]],
    val before: SizedRegion[1],
    val after: SizedRegion[1],
    val _results: Seq[Result[TypeAttribute]]
) extends DerivedOperation["scf.while", WhileOp],
      RegionBranchOpInterface,
      LoopLikeOpInterface,
      RecursiveMemoryEffects,
      SingleBlock derives DerivedOperationCompanion

case class IndexSwitchOp(
    val arg: Operand[Index],
    val cases: DenseI64ArrayAttr,
    val defaultRegion: SizedRegion[1],
    val caseRegions: VariadicRegion[SizedRegion[1]],
    val _results: Seq[Result[TypeAttribute]]
) extends DerivedOperation["scf.index_switch", IndexSwitchOp],
      RecursiveMemoryEffects,
      SingleBlockImplicitTerminator[YieldOp],
      RegionBranchOpInterface derives DerivedOperationCompanion

case class YieldOp(
    val _results: Seq[Operand[TypeAttribute]]
) extends DerivedOperation["scf.yield", YieldOp],
      ParentOneOf[(ExecuteRegionOp, ForOp, IfOp, IndexSwitchOp, WhileOp)],
      ReturnLike,
      IsTerminator,
      NoMemoryEffect derives DerivedOperationCompanion

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
