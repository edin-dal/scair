package scair.passes.convert_func_to_llvm

import scair.MLContext
import scair.dialects.builtin.*
import scair.dialects.func
import scair.dialects.llvm
import scair.ir.*
import scair.transformations.*

private def resultType(types: Seq[Attribute]): TypeAttribute = types match
  case Seq()                      => llvm.VoidType()
  case Seq(single: TypeAttribute) => single
  case many => llvm.StructType(many.map(_.asInstanceOf[TypeAttribute]))

private def position(index: Int): DenseArrayAttr =
  DenseArrayAttr(I64, Seq(IntegerAttr(IntData(index), I64)))

private val LowerFunc = conversionPattern { case op: func.Func =>
  val lowered = llvm.Func(
    op.sym_name,
    llvm.LLVMFunctionType(
      op.function_type.inputs.map(_.asInstanceOf[TypeAttribute]),
      resultType(op.function_type.outputs),
    ),
    // A declaration has no visibility to carry over.
    if op.body.blocks.isEmpty then None else op.sym_visibility,
    op.body.detached,
  )
  lowered.attributes ++= op.attributes
  lowered
}

private val LowerCall = conversionPattern { case op: func.Call =>
  val types = op._results.map(r => convertType(r.typ))
  val packedType = resultType(types)
  val call = llvm.Call(
    Some(op.callee),
    adaptor.operands,
    if types.isEmpty then Seq.empty else Seq(Result(packedType)),
  )
  if types.size <= 1 then call
  else
    val container = call.result.head.asInstanceOf[Operand[llvm.StructType]]
    val extracts = types.zipWithIndex.map { (typ, index) =>
      llvm.ExtractValue(container, position(index), Result(typ))
    }
    (call +: extracts, extracts.map(_.res))
}

private val LowerReturn = conversionPattern { case _: func.Return =>
  val args = adaptor.operands
  if args.size <= 1 then llvm.Return(args)
  else
    val aggregate = resultType(args.map(_.typ)).asInstanceOf[llvm.StructType]
    val poison = llvm.Poison(Result(aggregate))
    var container = poison.res
      .asInstanceOf[Operand[llvm.StructType | llvm.ArrayType]]
    val inserts = args.zipWithIndex.map { (value, index) =>
      val insert = llvm
        .InsertValue(container, value, position(index), Result(aggregate))
      container = insert.res
      insert
    }
    Seq(poison) ++ inserts :+ llvm.Return(Seq(container))
}

final case class ConvertFuncToLLVM() extends ConversionPass
    derives PassCompanion:
  override val typeConverter: TypeConverter = TypeConverter(Seq.empty)

  override val patterns: Seq[ConversionPattern] =
    Seq(LowerFunc, LowerCall, LowerReturn)
