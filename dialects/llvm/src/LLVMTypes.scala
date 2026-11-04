package scair.dialects.llvm

import fastparse.*
import scair.clair.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.parse.*
import scair.parse.given
import scair.print.Printer

case class VoidType() extends DerivedAttribute["llvm.void"] with TypeAttribute
    derives AttrDefs

case class LLVMFunctionType(
    inputs: ArrayAttribute[TypeAttribute],
    returnType: TypeAttribute = VoidType(),
    variadic: IntData = IntData(0),
) extends DerivedAttribute["llvm.func"]
    with TypeAttribute derives AttrDefs:

  override def printParameters(p: Printer): Unit =
    p.print("<", returnType, " (")
    p.printListF(inputs, p.print, sep = ", ")
    if variadic.data != 0 then
      if inputs.nonEmpty then p.print(", ")
      p.print("...")
    p.print(")>")

// MLIR abbreviates LLVM types nested inside another LLVM type (ptr, void, ...).
private[llvm] def llvmTypeP[$: P](using Parser): P[TypeAttribute] =
  P(
    typeP.map(_.asInstanceOf[TypeAttribute]) | ("void".map(_ => VoidType())) |
      ("ptr" ~ summon[AttributeCustomParser[Ptr]].parse) |
      ("array" ~ summon[AttributeCustomParser[ArrayType]].parse) |
      ("struct" ~ summon[AttributeCustomParser[StructType]].parse) |
      ("func" ~ llvmFunctionParametersP)
  )

private def llvmFunctionParametersP[$: P](using Parser): P[LLVMFunctionType] =
  P(
    "<" ~ llvmTypeP ~ "(" ~ llvmTypeP.rep(sep = ",") ~ ((",".? ~ "...").!.?) ~
      ")" ~ ">"
  ).map { (result, args, vararg) =>
    LLVMFunctionType(args, result, IntData(if vararg.nonEmpty then 1 else 0))
  }

given AttributeCustomParser[LLVMFunctionType]:

  override def parse[$: P](using Parser): P[LLVMFunctionType] =
    llvmFunctionParametersP
