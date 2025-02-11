package test.generated

import test.newir.*
import ValueConversions.{opToVal, valToOp, resToVal, valToRes}
import scair.ir.{Value, Attribute, ListType, DictType}
import scair.scairdl.constraints.{BaseAttr, ConstraintContext}
import scair.ir.Value
import scair.dialects.builtin.IntegerAttr

import test.testing.ExampleWithGenedCode.{Mul, Norm}

val x = new MLIRRealm[Mul] {

  def unverify(op: Mul): RegisteredOp[Mul] = {
    val op1 = RegisteredOp[Mul](
      name = "mul",
      operands = ListType(op.lhs, op.rhs)
    )

    op1.results.clear()
    op1.results.addAll(ListType(op.result))
    op1
  }

  def verify(op: RegisteredOp[_]): Mul = {
    if (op.operands.length != 2) then
      throw new Exception(s"Expected 2 operands, got ${op.operands.length}")
    if (op.results.length != 1) then
      throw new Exception(s"Expected 1 results, got ${op.results.length}")

    BaseAttr[scair.dialects.builtin.IntegerAttr]()
      .verify(op.operands(0).typ, new ConstraintContext())
    BaseAttr[scair.dialects.builtin.IntegerAttr]()
      .verify(op.operands(1).typ, new ConstraintContext())
    BaseAttr[scair.dialects.builtin.IntegerAttr]()
      .verify(op.results(0).typ, new ConstraintContext())

    Mul(
      lhs =
        op.operands(0).asInstanceOf[Value[scair.dialects.builtin.IntegerAttr]],
      rhs =
        op.operands(1).asInstanceOf[Value[scair.dialects.builtin.IntegerAttr]],
      result =
        op.results(0).asInstanceOf[Value[scair.dialects.builtin.IntegerAttr]]
    )
  }

}

val y = new MLIRRealm[Norm] {

  def unverify(op: Norm): RegisteredOp[Norm] = {
    val op1 = RegisteredOp[Norm](
      name = "norm",
      operands = ListType(op.norm)
    )

    op1.results.clear()
    op1.results.addAll(ListType(op.result))
    op1
  }

  def verify(op: RegisteredOp[_]): Norm = {
    if (op.operands.length != 1) then
      throw new Exception(s"Expected 1 operands, got ${op.operands.length}")
    if (op.results.length != 1) then
      throw new Exception(s"Expected 1 results, got ${op.results.length}")

    BaseAttr[scair.dialects.builtin.IntegerAttr]()
      .verify(op.operands(0).typ, new ConstraintContext())
    BaseAttr[scair.dialects.builtin.IntegerAttr]()
      .verify(op.results(0).typ, new ConstraintContext())

    Norm(
      norm =
        op.operands(0).asInstanceOf[Value[scair.dialects.builtin.IntegerAttr]],
      result =
        op.results(0).asInstanceOf[Value[scair.dialects.builtin.IntegerAttr]]
    )
  }

}
