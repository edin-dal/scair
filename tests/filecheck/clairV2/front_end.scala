// RUN: scala full-classpath %s - | filecheck %s

import scair.clairV2.mirrored._
import scair.dialects.builtin.*
import scair.scairdl.irdef._
import scair.ir.*

object Main {

  case class Mul(
      lhs: Operand[IntegerAttr],
      rhs: Operand[IntegerAttr],
      result: Result[IntegerAttr],
      randProp: Property[StringData],
      randAttr: Attr[StringData]
  ) extends ADTOperation

  case class Norm(
      norm: Operand[IntegerAttr],
      result: Result[IntegerAttr]
  ) extends ADTOperation

  def main(args: Array[String]): Unit = {
    val mlirOpDef = summonMLIROps[(Mul, Norm)]

    println(mlirOpDef.print)
  }

}

// CHECK:       import scair.ir.*
// CHECK-NEXT:  import scair.ir.ValueConversions.{resToVal, valToRes}
// CHECK-NEXT:  import scair.scairdl.constraints.{BaseAttr, ConstraintContext}
// CHECK-NEXT:  import scair.dialects.builtin.IntegerAttr
// CHECK-NEXT:  import scair.dialects.builtin.StringData

// CHECK:       object MulHelper extends ADTCompanion {
// CHECK-NEXT:    val getMLIRRealm: MLIRRealm[Mul] = new MLIRRealm[Mul] {

// CHECK:           def constructUnverifiedOp(
// CHECK-NEXT:          name: String,
// CHECK-NEXT:          operands: ListType[Value[Attribute]] = ListType(),
// CHECK-NEXT:          successors: ListType[Block] = ListType(),
// CHECK-NEXT:          results_types: ListType[Attribute] = ListType(),
// CHECK-NEXT:          regions: ListType[Region] = ListType(),
// CHECK-NEXT:          dictionaryProperties: DictType[String, Attribute] = DictType.empty[String, Attribute],
// CHECK-NEXT:          dictionaryAttributes: DictType[String, Attribute] = DictType.empty[String, Attribute]
// CHECK-NEXT:      ): UnverifiedOp[Mul] = {
// CHECK-NEXT:        UnverifiedOp[Mul](
// CHECK-NEXT:          name = name,
// CHECK-NEXT:          operands = operands,
// CHECK-NEXT:          successors = successors,
// CHECK-NEXT:          results_types = results_types,
// CHECK-NEXT:          regions = regions,
// CHECK-NEXT:          dictionaryProperties = dictionaryProperties,
// CHECK-NEXT:          dictionaryAttributes = dictionaryAttributes
// CHECK-NEXT:        )
// CHECK-NEXT:      }

// CHECK:           def unverify(op: Mul): UnverifiedOp[Mul] = {
// CHECK-NEXT:        val op1 = UnverifiedOp[Mul](
// CHECK-NEXT:          name = "mul",
// CHECK-NEXT:          operands = ListType(op.lhs, op.rhs),
// CHECK-NEXT:          dictionaryProperties = DictType(("randProp", op.randProp.typ)),
// CHECK-NEXT:          dictionaryAttributes = DictType(("randAttr", op.randAttr.typ))
// CHECK-NEXT:        )

// CHECK:             op1.results.clear()
// CHECK-NEXT:        op1.results.addAll(ListType(op.result))
// CHECK-NEXT:        op1
// CHECK-NEXT:      }

// CHECK:           def verify(op: UnverifiedOp[Mul]): Mul = {

// CHECK:             if (op.operands.size != 2) then throw new Exception(s"Expected 2 operands, got ${op.operands.size}")
// CHECK-NEXT:        if (op.results.size != 1) then throw new Exception(s"Expected 1 results, got ${op.results.size}")
// CHECK-NEXT:        if (op.dictionaryProperties.size != 1) then throw new Exception(s"Expected 1 dictionaryProperties, got ${op.dictionaryProperties.size}")
// CHECK-NEXT:        if (op.dictionaryAttributes.size != 1) then throw new Exception(s"Expected 1 dictionaryAttributes, got ${op.dictionaryAttributes.size}")

// CHECK:             if !(op.dictionaryProperties.contains("randProp") &&
// CHECK-NEXT:        op.dictionaryAttributes.contains("randAttr"))
// CHECK-NEXT:        then throw new Exception("Expected specific properties and attributes")

// CHECK:             BaseAttr[scair.dialects.builtin.IntegerAttr]().verify(op.operands(0).typ, new ConstraintContext())
// CHECK-NEXT:        BaseAttr[scair.dialects.builtin.IntegerAttr]().verify(op.operands(1).typ, new ConstraintContext())
// CHECK-NEXT:        BaseAttr[scair.dialects.builtin.IntegerAttr]().verify(op.results(0).typ, new ConstraintContext())
// CHECK-NEXT:        BaseAttr[scair.dialects.builtin.StringData]().verify(op.dictionaryProperties("randProp"), new ConstraintContext())
// CHECK-NEXT:        BaseAttr[scair.dialects.builtin.StringData]().verify(op.dictionaryAttributes("randAttr"), new ConstraintContext())

// CHECK:             Mul(
// CHECK-NEXT:          lhs = op.operands(0).asInstanceOf[Value[scair.dialects.builtin.IntegerAttr]],
// CHECK-NEXT:          rhs = op.operands(1).asInstanceOf[Value[scair.dialects.builtin.IntegerAttr]],
// CHECK-NEXT:          result = op.results(0).asInstanceOf[Value[scair.dialects.builtin.IntegerAttr]],
// CHECK-NEXT:          randProp = Property(op.dictionaryProperties("randProp").asInstanceOf[scair.dialects.builtin.StringData]),
// CHECK-NEXT:          randAttr = Attr(op.dictionaryAttributes("randAttr").asInstanceOf[scair.dialects.builtin.StringData])
// CHECK-NEXT:        )
// CHECK-NEXT:      }

// CHECK:         }
// CHECK-NEXT:  }

// CHECK:       object NormHelper extends ADTCompanion {
// CHECK-NEXT:    val getMLIRRealm: MLIRRealm[Norm] = new MLIRRealm[Norm] {

// CHECK:           def constructUnverifiedOp(
// CHECK-NEXT:          name: String,
// CHECK-NEXT:          operands: ListType[Value[Attribute]] = ListType(),
// CHECK-NEXT:          successors: ListType[Block] = ListType(),
// CHECK-NEXT:          results_types: ListType[Attribute] = ListType(),
// CHECK-NEXT:          regions: ListType[Region] = ListType(),
// CHECK-NEXT:          dictionaryProperties: DictType[String, Attribute] = DictType.empty[String, Attribute],
// CHECK-NEXT:          dictionaryAttributes: DictType[String, Attribute] = DictType.empty[String, Attribute]
// CHECK-NEXT:      ): UnverifiedOp[Norm] = {
// CHECK-NEXT:        UnverifiedOp[Norm](
// CHECK-NEXT:          name = name,
// CHECK-NEXT:          operands = operands,
// CHECK-NEXT:          successors = successors,
// CHECK-NEXT:          results_types = results_types,
// CHECK-NEXT:          regions = regions,
// CHECK-NEXT:          dictionaryProperties = dictionaryProperties,
// CHECK-NEXT:          dictionaryAttributes = dictionaryAttributes
// CHECK-NEXT:        )
// CHECK-NEXT:      }

// CHECK:           def unverify(op: Norm): UnverifiedOp[Norm] = {
// CHECK-NEXT:        val op1 = UnverifiedOp[Norm](
// CHECK-NEXT:          name = "norm",
// CHECK-NEXT:          operands = ListType(op.norm)
// CHECK-NEXT:        )

// CHECK:             op1.results.clear()
// CHECK-NEXT:        op1.results.addAll(ListType(op.result))
// CHECK-NEXT:        op1
// CHECK-NEXT:      }

// CHECK:           def verify(op: UnverifiedOp[Norm]): Norm = {

// CHECK:             if (op.operands.size != 1) then throw new Exception(s"Expected 1 operands, got ${op.operands.size}")
// CHECK-NEXT:        if (op.results.size != 1) then throw new Exception(s"Expected 1 results, got ${op.results.size}")

// CHECK:             BaseAttr[scair.dialects.builtin.IntegerAttr]().verify(op.operands(0).typ, new ConstraintContext())
// CHECK-NEXT:        BaseAttr[scair.dialects.builtin.IntegerAttr]().verify(op.results(0).typ, new ConstraintContext())

// CHECK:             Norm(
// CHECK-NEXT:          norm = op.operands(0).asInstanceOf[Value[scair.dialects.builtin.IntegerAttr]],
// CHECK-NEXT:          result = op.results(0).asInstanceOf[Value[scair.dialects.builtin.IntegerAttr]]
// CHECK-NEXT:        )
// CHECK-NEXT:      }

// CHECK:         }
// CHECK-NEXT:  }
