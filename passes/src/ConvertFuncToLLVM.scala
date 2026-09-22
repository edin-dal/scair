package scair.passes.convert_func_to_llvm

import scair.MLContext
import scair.dialects.func
import scair.dialects.llvm
import scair.ir.*
import scair.transformations.*

private val LowerFunc = conversionPattern { case op: func.Func =>
  val lowered = llvm.Func(
    op.sym_name,
    op.function_type,
    // A declaration has no visibility to carry over.
    if op.body.blocks.isEmpty then None else op.sym_visibility,
    op.body,
  )
  lowered.attributes ++= op.attributes
  lowered
}

private val LowerCall = conversionPattern { case op: func.Call =>
  llvm.Call(
    op.callee,
    adaptor.operands,
    op._results.map(r => Result(convertType(r.typ))),
  )
}

private val LowerReturn = conversionPattern { case _: func.Return =>
  llvm.Return(adaptor.operands)
}

final class ConvertFuncToLLVM(ctx: MLContext) extends ConversionPass(ctx):
  override val name: String = "convert-func-to-llvm"

  override val typeConverter: TypeConverter = TypeConverter(Seq.empty)

  override val patterns: Seq[ConversionPattern] =
    Seq(LowerFunc, LowerCall, LowerReturn)
