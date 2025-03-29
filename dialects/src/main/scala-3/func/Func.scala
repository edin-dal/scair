package scair.dialects.func

import scair.dialects.builtin.*
import scair.ir.Attribute
import scair.ir.*
import scair.clair.codegen.*
import scair.clair.mirrored.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.clair.macros._

case class Call(
    callee: Property[SymbolRefAttr],
    _operands: Variadic[Operand[Attribute]],
    _results: Variadic[Result[Attribute]]
) extends MLIRName["func.call"]
    derives MLIRTrait

case class Func(
    sym_name: Property[StringData],
    function_type: Property[FunctionType],
    // TODO: This needs optional
    // sym_visibility: Property[StringData],
    body: Region
) extends MLIRName["func.func"]
    derives MLIRTrait

case class Return(
    _operands: Variadic[Operand[Attribute]]
) extends MLIRName["func.return"]
    derives MLIRTrait

val FuncDialect = summonDialect[(Call, Func, Return)](Seq())
