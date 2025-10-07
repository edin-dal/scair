package scair.tools

import scair.ir.*
import scair.dialects.arith
import scair.dialects.func
import scair.dialects.builtin


// TODO: do flow analysis to find correct return op
class Interpreter {

  // TODO: generic interpret op, probably will become block/function evaluator soon
  def interpret(op: Operation): Attribute = {
    // presuming main function for now
    op match {
      case func_op: func.Func =>
        val return_op = func_op.body.blocks.head.operations.last
        interpret_op(return_op)
      case other => throw new Exception("Expecting function")
    }

    
  }

  def interpret_op(op: Operation): Attribute = {
    op match {

      // Function Return
      // TODO: multiple return vals, environments
      case funcReturnOp: func.Return =>
        val return_root = funcReturnOp.operands.head.owner.getOrElse(0).asInstanceOf[Operation]
        interpret_op(return_root)

      // Constants
      // TODO: vectors?
      case constant: arith.Constant => constant.value

      // TODO: handling block arguments for all arithmetic operations
      // Binary Operations
      case addIOp: arith.AddI => 
        interpret_bin_op(addIOp.lhs, addIOp.rhs, "AddI")(_ + _)
      case subIOp: arith.SubI => 
        interpret_bin_op(subIOp.lhs, subIOp.rhs, "SubI")(_ - _)
      case mulIOp: arith.MulI => 
        interpret_bin_op(mulIOp.lhs, mulIOp.rhs, "MulI")(_ * _)
      case divSIOp: arith.DivSI => 
        interpret_bin_op(divSIOp.lhs, divSIOp.rhs, "DivSI")(_.divSI(_))
      case divUIOp: arith.DivUI => 
        interpret_bin_op(divUIOp.lhs, divUIOp.rhs, "DivUI")(_.divUI(_))
      case andIOp: arith.AndI =>
        interpret_bin_op(andIOp.lhs, andIOp.rhs, "AndI")(_ & _)
      case orIOp: arith.OrI =>
        interpret_bin_op(orIOp.lhs, orIOp.rhs, "OrI")(_ | _)
      case xorIOp: arith.XOrI =>
        interpret_bin_op(xorIOp.lhs, xorIOp.rhs, "XorI")(_ ^ _)

        // Shift Operations
        // TODO: convert to unsigned?
      case shliOp: arith.ShLI =>
        interpret_bin_op(shliOp.lhs, shliOp.rhs, "ShLI")(_ << _)
      case shrsiOp: arith.ShRSI =>
        interpret_bin_op(shrsiOp.lhs, shrsiOp.rhs, "ShRSI")(_ >> _)
      case shruiOp: arith.ShRUI =>
        interpret_bin_op(shruiOp.lhs, shruiOp.rhs, "ShRUI")(_ >>> _)

      // Comparison Operation
      case cmpiOp: arith.CmpI =>
        interpret_cmp_op(cmpiOp.lhs, cmpiOp.rhs, cmpiOp.predicate.value.toInt)
      case _ => throw new Exception("Unsupported operation")

      // Select Operation
    }
  }

  def interpret_bin_op(lhs: Value[Attribute], rhs: Value[Attribute], name: String)
  (combine: (IntegerAttr, IntegerAttr) => IntegerAttr): Attribute = {
    (lhs.owner.get, rhs.owner.get) match {
      case (l: Operation, r: Operation) =>
        (interpret_op(l), interpret_op(r)) match {
          case (li: IntegerAttr, ri: IntegerAttr) =>
            combine(li, ri)
          case other => throw new Exception("Unsupported operand types for $name, got: $other")
        }
      case other => throw new Exception("Expected operations as operands for $name, got: $other")
    }
  }

  def interpret_cmp_op(lhs: Value[Attribute], rhs: Value[Attribute], predicate: Int): Attribute = {
    (lhs.owner.get, rhs.owner.get) match {
      case (l: Operation, r: Operation) =>
        (interpret_op(l), interpret_op(r)) match {
          case (li: IntegerAttr, ri: IntegerAttr) =>
            predicate match {
              case 0 => // EQ
                if (li.value == ri.value) IntegerAttr(IntData(1), li.typ) else IntegerAttr(IntData(0), li.typ)
              case 1 => // NE
                if (li.value != ri.value) IntegerAttr(IntData(1), li.typ) else IntegerAttr(IntData(0), li.typ)
              // case 2 => // SLT

              case _ => throw new Exception("Unknown comparison predicate")
            }
          case other => throw new Exception("Unsupported operand types for cmpi, got: $other")
        }
      case other => throw new Exception("Expected operations as operands for cmpi, got: $other")
    }
  }

//  def convert_sign(intAttr: IntegerAttr, sign: Signedness): IntegerAttr = {
//    (intAttr.typ.asInstanceOf[IntegerType].sign, sign) match {
//      case (Signed, Signed) | (Unsigned, Unsigned) | (Signless, Signless) => intAttr
//      case (Signed, Unsigned) =>
//        val intValue = intAttr.value.value.toUInt
//    }
//  } 
}
