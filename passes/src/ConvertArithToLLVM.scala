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
// operation it is built into expects.
private def asInteger(v: Value[Attribute]): Operand[IntegerType | IndexType] =
  v.asInstanceOf[Operand[IntegerType | IndexType]]

private def asFloat(v: Value[Attribute]): Operand[FloatType] =
  v.asInstanceOf[Operand[FloatType]]

private def integerResult(attr: Attribute): Result[IntegerType | IndexType] =
  Result(attr.asInstanceOf[IntegerType | IndexType])

private def floatResult(attr: Attribute): Result[FloatType] =
  Result(attr.asInstanceOf[FloatType])

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
  llvm.Add(
    asInteger(adaptor(0)),
    asInteger(adaptor(1)),
    integerResult(convertType(op.result.typ)),
  )
}

private val LowerMulI = conversionPattern { case op: arith.MulI =>
  llvm.Mul(
    asInteger(adaptor(0)),
    asInteger(adaptor(1)),
    integerResult(convertType(op.result.typ)),
  )
}

private val LowerAddF = conversionPattern { case op: arith.AddF =>
  llvm.FAdd(
    asFloat(adaptor(0)),
    asFloat(adaptor(1)),
    floatResult(convertType(op.result.typ)),
  )
}

private val LowerMulF = conversionPattern { case op: arith.MulF =>
  llvm.FMul(
    asFloat(adaptor(0)),
    asFloat(adaptor(1)),
    floatResult(convertType(op.result.typ)),
  )
}

// `llvm.icmp` is already lowered, but its operands are not: it is converted so
// that a comparison of converted indices does not have to cast them back.
private val LowerICmp = conversionPattern { case op: llvm.ICmp =>
  llvm.ICmp(
    asInteger(adaptor(0)),
    asInteger(adaptor(1)),
    Result(op.res.typ),
    op.predicate,
  )
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
