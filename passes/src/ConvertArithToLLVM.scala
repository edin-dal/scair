package scair.passes.convert_arith_to_llvm

import scair.MLContext
import scair.dialects.arith
import scair.dialects.builtin.*
import scair.dialects.llvm
import scair.ir.*
import scair.transformations.*

private val llvmIndexType: IntegerType = I64

private val IndexToLLVM = typeConversion { case _: IndexType => llvmIndexType }

// `Value`'s parameter is covariant, so a converted operand - which the adaptor
// hands over as a `Value[Attribute]` - has to be ascribed to the type the
// operation it is built into expects, as does its result type.
private def as[T <: Attribute](v: Value[Attribute]): Operand[T] =
  v.asInstanceOf[Operand[T]]

private def res[T <: Attribute](attr: Attribute): Result[T] =
  Result(attr.asInstanceOf[T])

/** The driver converts the types of values, not those an attribute carries. */
private def convertConstantAttr(attr: Attribute): Attribute =
  attr match
    case IntegerAttr(IntData(v), _: IndexType) =>
      IntegerAttr(IntData(v), llvmIndexType)
    case other => other

private val LowerConstant = conversionPattern { case op: arith.Constant =>
  llvm.Constant(
    convertConstantAttr(op.value),
    Result(convertType(op.result.typ)),
  )
}

private val LowerAddI = conversionPattern { case op: arith.AddI =>
  llvm.Add(as(adaptor(0)), as(adaptor(1)), res(convertType(op.result.typ)))
}

private val LowerMulI = conversionPattern { case op: arith.MulI =>
  llvm.Mul(as(adaptor(0)), as(adaptor(1)), res(convertType(op.result.typ)))
}

private val LowerAddF = conversionPattern { case op: arith.AddF =>
  llvm.FAdd(as(adaptor(0)), as(adaptor(1)), res(convertType(op.result.typ)))
}

private val LowerMulF = conversionPattern { case op: arith.MulF =>
  llvm.FMul(as(adaptor(0)), as(adaptor(1)), res(convertType(op.result.typ)))
}

// `llvm.icmp` is already lowered, but its operands are not: it is converted so
// that a comparison of converted indices does not have to cast them back.
private val LowerICmp = conversionPattern { case op: llvm.ICmp =>
  llvm.ICmp(as(adaptor(0)), as(adaptor(1)), Result(op.res.typ), op.predicate)
}

// Converts scalar arithmetic to LLVM arithmetic.
// Example: `arith.constant` / `arith.addi` / `arith.muli`
//   -> `llvm.constant` / `llvm.add` / `llvm.mul`.
final class ConvertArithToLLVM(ctx: MLContext) extends ConversionPass(ctx):
  override val name: String = "convert-arith-to-llvm"

  override val typeConverter: TypeConverter = TypeConverter(Seq(IndexToLLVM))

  override val patterns: Seq[ConversionPattern] =
    Seq(
      LowerConstant,
      LowerAddI,
      LowerMulI,
      LowerAddF,
      LowerMulF,
      LowerICmp,
    )
