package scair.passes.convert_arith_to_llvm

import scair.MLContext
import scair.dialects.arith
import scair.dialects.builtin.*
import scair.dialects.llvm
import scair.ir.*
import scair.transformations.GreedyRewritePatternApplier
import scair.transformations.PatternRewriteWalker
import scair.transformations.WalkerPass
import scair.transformations.pattern

// TODO: This is platform-dependent and shouldn't be hardcoded
private val llvmIndexType: IntegerType = I64

private def convertLLVMType[A <: Attribute](attr: A): A | IntegerType =
  attr match
    case _: IndexType => llvmIndexType
    case other        => other

private def convertLLVMConstantAttr(attr: Attribute): Attribute =
  attr match
    case IntegerAttr(IntData(v), _: IndexType) =>
      IntegerAttr(IntData(v), llvmIndexType)
    case other => other

private val LowerConstant = pattern { case c: arith.Constant =>
  llvm.Constant(
    convertLLVMConstantAttr(c.value),
    Result(convertLLVMType(c.result.typ)),
  )
}

private val LowerAddI = pattern { case add: arith.AddI =>
  llvm.Add(add.lhs, add.rhs, Result(convertLLVMType(add.result.typ)))
}

private val LowerMulI = pattern { case mul: arith.MulI =>
  llvm.Mul(mul.lhs, mul.rhs, Result(convertLLVMType(mul.result.typ)))
}

private val LowerAddF = pattern { case add: arith.AddF =>
  llvm.FAdd(add.lhs, add.rhs, Result(add.result.typ))
}

private val LowerMulF = pattern { case mul: arith.MulF =>
  llvm.FMul(mul.lhs, mul.rhs, Result(mul.result.typ))
}

// Converts scalar arithmetic to LLVM arithmetic.
// Example: `arith.constant` / `arith.addi` / `arith.muli`
//   -> `llvm.constant` / `llvm.add` / `llvm.mul`.
final class ConvertArithToLLVM(ctx: MLContext) extends WalkerPass(ctx):
  override val name: String = "convert-arith-to-llvm"

  override val walker: PatternRewriteWalker =
    PatternRewriteWalker(
      GreedyRewritePatternApplier(
        Seq(LowerConstant, LowerAddI, LowerMulI, LowerAddF, LowerMulF)
      )
    )
