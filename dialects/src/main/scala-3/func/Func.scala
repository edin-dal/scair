package scair.dialects.funcgen

import scair.clair.mirrored.*
import scair.dialects.builtin.*
import scair.scairdl.irdef.ScaIRDLDialect
import scair.ir.Attribute

case class CallOp(
    callee: Property[SymbolRefAttr],
    _operands: Variadic[Operand[Attribute]],
    _results: Variadic[Result[Attribute]]
) extends OperationFE
case class FuncOp(
    sym_name: Property[StringData],
    function_type: Property[FunctionType],
    sym_visibility: Property[StringData],
    body: Region
) extends OperationFE
case class ReturnOp(
    _operands: Variadic[Operand[Attribute]]
) extends OperationFE

object FuncGen
    extends ScaIRDLDialect(
      summonDialect[
        (
            CallOp,
            FuncOp,
            ReturnOp
        )
      ](
        "Func",
        Seq(),
        Seq()
      )
    )
