package scair.passes.convert_arith_to_llvm

import scair.MLContext
import scair.dialects.arith
import scair.dialects.builtin.*
import scair.dialects.func
import scair.dialects.llvm
import scair.ir.*
import scair.transformations.GreedyRewritePatternApplier
import scair.transformations.PatternRewriteWalker
import scair.transformations.WalkerPass
import scair.transformations.pattern

import scala.collection.mutable

private val llvmIndexType: IntegerType = I64

private def asLLVMIndex(v: Value[Attribute]): Operand[IntegerType | IndexType] =
  v.asInstanceOf[Operand[IntegerType | IndexType]]

private def asFloat(v: Value[Attribute]): Operand[FloatType] =
  v.asInstanceOf[Operand[FloatType]]

private def convertLLVMValueType(attr: Attribute): Attribute =
  attr match
    case _: IndexType => llvmIndexType
    case other        => other

private def convertLLVMIntegerType(attr: Attribute): IntegerType | IndexType =
  attr match
    case _: IndexType       => llvmIndexType
    case other: IntegerType => other

private def convertLLVMConstantAttr(attr: Attribute): Attribute =
  attr match
    case IntegerAttr(IntData(v), _: IndexType) =>
      IntegerAttr(IntData(v), llvmIndexType)
    case other => other

// This builder performs a whole-function rebuild so arithmetic conversion can
// preserve block order and SSA remapping without relying on full conversion
// infrastructure.
private final class Builder(val funcOp: func.Func):
  val blockMap = mutable.Map.empty[Block, Block]
  val valueMap = mutable.Map.empty[Value[Attribute], Value[Attribute]]

  private def remap(v: Value[Attribute]): Value[Attribute] =
    valueMap.getOrElse(v, v)

  private def lowerConstant(op: arith.Constant, block: Block): Operation =
    val lowered = llvm.Constant(
      convertLLVMConstantAttr(op.value),
      Result(convertLLVMValueType(op.result.typ)),
    )
    valueMap(op.result) = lowered.res
    lowered

  private def lowerOp(op: Operation): Seq[Operation] =
    op match
      case c: arith.Constant =>
        Seq(
          llvm.Constant(
            convertLLVMConstantAttr(c.value),
            Result(convertLLVMValueType(c.result.typ)),
          )
        )
      case add: arith.AddI =>
        val lowered = llvm.Add(
          asLLVMIndex(remap(add.lhs)),
          asLLVMIndex(remap(add.rhs)),
          Result(convertLLVMIntegerType(add.result.typ)),
        )
        valueMap(add.result) = lowered.res
        Seq(lowered)
      case mul: arith.MulI =>
        val lowered = llvm.Mul(
          asLLVMIndex(remap(mul.lhs)),
          asLLVMIndex(remap(mul.rhs)),
          Result(convertLLVMIntegerType(mul.result.typ)),
        )
        valueMap(mul.result) = lowered.res
        Seq(lowered)
      case add: arith.AddF =>
        val lowered = llvm.FAdd(
          asFloat(remap(add.lhs)),
          asFloat(remap(add.rhs)),
          Result(add.result.typ),
        )
        valueMap(add.result) = lowered.res
        Seq(lowered)
      case mul: arith.MulF =>
        val lowered = llvm.FMul(
          asFloat(remap(mul.lhs)),
          asFloat(remap(mul.rhs)),
          Result(mul.result.typ),
        )
        valueMap(mul.result) = lowered.res
        Seq(lowered)
      case cmp: llvm.ICmp =>
        val lowered = llvm.ICmp(
          asLLVMIndex(remap(cmp.lhs)),
          asLLVMIndex(remap(cmp.rhs)),
          Result(cmp.res.typ),
          cmp.predicate,
        )
        valueMap(cmp.res) = lowered.res
        Seq(lowered)
      case other =>
        val copied = other.deepCopy(using blockMap, valueMap)
        valueMap.addAll(other.results.zip(copied.results))
        Seq(copied)

  def lower(): func.Func =
    val newBlocks = funcOp.body.blocks.map { oldBlock =>
      val nb = Block(oldBlock.arguments.map(_.typ), Seq.empty)
      blockMap(oldBlock) = nb
      valueMap.addAll(oldBlock.arguments.zip(nb.arguments))
      nb
    }
    funcOp.body.blocks.zip(newBlocks).foreach { case (oldBlock, newBlock) =>
      // Constants are emitted in source order within each block to keep the
      // lowering deterministic without introducing a pass-local ordering policy.
      val constants = oldBlock.operations.collect { case c: arith.Constant =>
        c
      }.toSeq
      constants.foreach { c =>
        val lowered = lowerConstant(c, newBlock)
        newBlock.addOp(lowered)
      }
      oldBlock.operations.foreach {
        case _: arith.Constant => ()
        case other             => newBlock.addOps(lowerOp(other))
      }
    }
    val lowered = func.Func(
      funcOp.sym_name,
      funcOp.function_type,
      funcOp.sym_visibility,
      Region(newBlocks),
    )
    lowered.attributes.addAll(funcOp.attributes)
    lowered

private val LowerFunc = pattern {
  case op: func.Func if op.body.blocks.exists(_.operations.exists {
        case _: arith.Constant | _: arith.AddI | _: arith.MulI | _: arith.AddF |
            _: arith.MulF =>
          true
        case _ => false
      }) =>
    Builder(op).lower()
}

// Converts scalar arithmetic to LLVM arithmetic.
// Example: `arith.constant` / `arith.addi` / `arith.muli`
//   -> `llvm.constant` / `llvm.add` / `llvm.mul`.
final class ConvertArithToLLVM(ctx: MLContext) extends WalkerPass(ctx):
  override val name: String = "convert-arith-to-llvm"

  override val walker: PatternRewriteWalker =
    PatternRewriteWalker(GreedyRewritePatternApplier(Seq(LowerFunc)))
