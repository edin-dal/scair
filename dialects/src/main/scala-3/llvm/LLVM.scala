package scair.dialects.llvm

import scair.clair.codegen.*
import scair.clair.macros.*
import scair.dialects.builtin.*
import scair.ir.*

case class Ptr() extends DerivedAttribute["llvm.ptr", Ptr] with TypeAttribute

case class Load(
    addr: Operand[Ptr],
    res: Result[Attribute]
) extends DerivedOperation["llvm.load", Load]
    with AssemblyFormat["$addr attr-dict `:` type($addr) `->` type($res)"]

case class GetElementPtr(
    base: Operand[Ptr],
    dynamicIndices: Seq[Operand[IntegerType]],
    res: Result[Ptr],
    rawConstantIndices: DenseArrayAttr,
    elem_type: Attribute
) extends DerivedOperation["llvm.getelementptr", GetElementPtr]
    with NoMemoryEffect

val LLVMDialect = summonDialect[Tuple1[Ptr], (Load, GetElementPtr)](Seq())
