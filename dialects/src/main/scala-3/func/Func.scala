package scair.dialects.func

import scair.dialects.builtin.*
import scair.ir.Attribute
import scair.ir.*
import scair.clairV2.codegen.*
import scair.clairV2.mirrored.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.clairV2.macros._

case class Call(
    callee: SymbolRefAttr,
    _operands: Seq[Operand[Attribute]],
    _results: Seq[Result[Attribute]]
) extends MLIRName["func.call"]
    derives MLIRTrait

case class Func(
    sym_name: StringData,
    function_type: FunctionType,
    // TODO: This needs optional
    // sym_visibility: StringData,
    body: Region
) extends MLIRName["func.func"]
    derives MLIRTrait

case class Return(
    _operands: Seq[Operand[Attribute]]
) extends MLIRName["func.return"]
    derives MLIRTrait

val FuncDialect = summonDialect[(Call, Func, Return)](Seq())
