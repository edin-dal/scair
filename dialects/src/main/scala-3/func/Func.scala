package scair.dialects.func

import scair.clair.codegen.*
import scair.clair.macros.*
import scair.dialects.builtin.*
import scair.ir.*

case class Call(
    callee: SymbolRefAttr,
    _operands: Seq[Operand[Attribute]],
    _results: Seq[Result[Attribute]]
) extends DerivedOperation["func.call", Call]
    derives DerivedOperationCompanion

case class Func(
    sym_name: StringData,
    function_type: FunctionType,
    // TODO: This needs optional
    // sym_visibility: StringData,
    body: Region
) extends DerivedOperation["func.func", Func]
    derives DerivedOperationCompanion

case class Return(
    _operands: Seq[Operand[Attribute]]
) extends DerivedOperation["func.return", Return]
    derives DerivedOperationCompanion

val FuncDialect = summonDialect[EmptyTuple, (Call, Func, Return)](Seq())
