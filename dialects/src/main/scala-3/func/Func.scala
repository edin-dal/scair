package scair.dialects.funcgen

import scair.clair.mirrored.*
import scair.dialects.builtin.*
import scair.ir.Attribute
import scair.scairdl.irdef.ScaIRDLDialect

case class Call(
    callee: Property[SymbolRefAttr],
    _operands: Variadic[Operand[Attribute]],
    _results: Variadic[Result[Attribute]]
) extends OperationFE
case class Func(
    sym_name: Property[StringData],
    function_type: Property[FunctionType],
    sym_visibility: Property[StringData],
    body: Region
) extends OperationFE
case class Return(
    _operands: Variadic[Operand[Attribute]]
) extends OperationFE

object FuncGen
    extends ScaIRDLDialect(
      summonDialect[
        (
            Call,
            Func,
            Return
        )
      ](
        "_Func",
        Seq(),
        Seq()
      )
    )
