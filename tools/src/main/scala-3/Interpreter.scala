package scair.tools

import scair.dialects.builtin.ModuleOp
import scair.dialects.builtin.IntegerAttr
import scair.ir.*
import scair.dialects.arith

class Interpreter {
    // TODO: BinOp helper
  def interpret(op: Operation): Attribute = {
    op match {
      // TODO: vectors?
      case constant: arith.Constant => constant.value
      // TODO: handling block arguments for all arithmetic operations
      case addIOp: arith.AddI => 
        interpretBinOp(addIOp.lhs, addIOp.rhs, "AddI")(_ + _)
      case subIOp: arith.SubI => 
        interpretBinOp(subIOp.lhs, subIOp.rhs, "SubI")(_ - _)
      case mulIOp: arith.MulI => 
        interpretBinOp(mulIOp.lhs, mulIOp.rhs, "MulI")(_ * _)
      case divSIOp: arith.DivSI => 
        interpretBinOp(divSIOp.lhs, divSIOp.rhs, "DivSI")(_.divSI(_))
      case divUIOp: arith.DivUI => 
        interpretBinOp(divUIOp.lhs, divUIOp.rhs, "DivUI")(_.divUI(_))
      case _ => throw new Exception("Unsupported operation")
    }
  }

  def interpretBinOp(lhs: Value[Attribute], rhs: Value[Attribute], name: String)(combine: (IntegerAttr, IntegerAttr) => IntegerAttr): Attribute = {
    (lhs.owner.get, rhs.owner.get) match {
      case (l: Operation, r: Operation) =>
        (interpret(l), interpret(r)) match {
          case (li: IntegerAttr, ri: IntegerAttr) =>
            combine(li, ri)
          case other => throw new Exception("Unsupported operand types for $name, got: $other")
        }
      case other => throw new Exception("Expected operations as operands for $name, got: $other")
    }
  }
}
