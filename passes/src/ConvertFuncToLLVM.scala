package scair.passes.convert_func_to_llvm

import scair.MLContext
import scair.dialects.func
import scair.dialects.llvm
import scair.ir.*
import scair.transformations.GreedyRewritePatternApplier
import scair.transformations.PatternRewriteWalker
import scair.transformations.WalkerPass
import scair.transformations.pattern

import scala.collection.mutable

private final class Builder(val funcOp: func.Func):
  private val blockMap = mutable.Map.empty[Block, Block]
  private val valueMap = mutable.Map.empty[Value[Attribute], Value[Attribute]]

  private def remap(v: Value[Attribute]): Value[Attribute] =
    valueMap.getOrElse(v, v)

  def lower(): llvm.Func =
    val newBlocks = funcOp.body.blocks.map { oldBlock =>
      val newBlock = Block(oldBlock.arguments.map(_.typ), Seq.empty)
      blockMap(oldBlock) = newBlock
      valueMap.addAll(oldBlock.arguments.zip(newBlock.arguments))
      newBlock
    }

    funcOp.body.blocks.zip(newBlocks).foreach { case (oldBlock, newBlock) =>
      oldBlock.operations.foreach {
        case call: func.Call =>
          val lowered = llvm.Call(
            call.callee,
            call._operands.map(v => remap(v).asInstanceOf[Operand[Attribute]]),
            call._results.map(r => Result(r.typ.asInstanceOf[Attribute])),
          )
          newBlock.addOp(lowered)
          valueMap.addAll(call.results.zip(lowered.results))
        case ret: func.Return =>
          newBlock
            .addOp(
              llvm
                .Return(
                  ret._operands
                    .map(v => remap(v).asInstanceOf[Operand[Attribute]])
                )
            )
        case other =>
          val copied = other
            .deepCopy(using mutable.Map.from(blockMap), valueMap)
          newBlock.addOp(copied)
          valueMap.addAll(other.results.zip(copied.results))
      }
    }

    val lowered = llvm.Func(
      funcOp.sym_name,
      funcOp.function_type,
      if funcOp.body.blocks.isEmpty then None else funcOp.sym_visibility,
      Region(newBlocks),
    )
    lowered.attributes.addAll(funcOp.attributes)
    lowered

private val LowerFunc = pattern { case op: func.Func =>
  Builder(op).lower()
}

final class ConvertFuncToLLVM(ctx: MLContext) extends WalkerPass(ctx):
  override val name: String = "convert-func-to-llvm"

  override val walker: PatternRewriteWalker =
    PatternRewriteWalker(GreedyRewritePatternApplier(Seq(LowerFunc)))
