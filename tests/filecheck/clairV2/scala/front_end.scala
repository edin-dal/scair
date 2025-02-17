// RUN: scala full-classpath %s - | filecheck %s

import scair.clairV2.mirrored._
import scair.dialects.builtin.*
import scair.scairdl.irdef._
import scair.ir.*

object Main {

  case class Mul(
      operand1: Operand[IntegerAttr],
      operand2: Operand[IntegerAttr],
      result1: Result[IntegerAttr],
      result2: Result[IntegerAttr],
      randProp1: Property[StringData],
      randProp2: Property[StringData],
      randAttr1: Attr[StringData],
      randAttr2: Attr[StringData],
      reg1: Region,
      reg2: Region,
      succ1: Successor,
      succ2: Successor
  ) extends ADTOperation

  case class Norm(
      norm: Operand[IntegerAttr],
      result: Result[IntegerAttr]
  ) extends ADTOperation

  def main(args: Array[String]): Unit = {
    val mlirOpDef = summonMLIROps[(Mul, Norm)]("CMath")

    println(mlirOpDef.print)
  }

}

// CHECK:       package Main

// CHECK:       import scair.ir.*
// CHECK-NEXT:  import scair.ir.ValueConversions.{resToVal, valToRes}
// CHECK-NEXT:  import scair.scairdl.constraints.{BaseAttr, ConstraintContext}
// CHECK-NEXT:  import scair.dialects.builtin.IntegerAttr
// CHECK-NEXT:  import scair.dialects.builtin.StringData

// CHECK:       object MulHelper extends ADTCompanion {
// CHECK-NEXT:    val getName: String = "cmath.mul"

// CHECK:         val getMLIRRealm: MLIRRealm[Mul] = new MLIRRealm[Mul] {

// CHECK:           def constructUnverifiedOp(
// CHECK-NEXT:          operands: ListType[Value[Attribute]] = ListType(),
// CHECK-NEXT:          successors: ListType[Block] = ListType(),
// CHECK-NEXT:          results_types: ListType[Attribute] = ListType(),
// CHECK-NEXT:          regions: ListType[Region] = ListType(),
// CHECK-NEXT:          dictionaryProperties: DictType[String, Attribute] = DictType.empty[String, Attribute],
// CHECK-NEXT:          dictionaryAttributes: DictType[String, Attribute] = DictType.empty[String, Attribute]
// CHECK-NEXT:      ): UnverifiedOp[Mul] = {
// CHECK-NEXT:        UnverifiedOp[Mul](
// CHECK-NEXT:          name = "cmath.mul",
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
// CHECK-NEXT:          name = "cmath.mul",
// CHECK-NEXT:          operands = ListType(op.operand1, op.operand2),
// CHECK-NEXT:          successors = ListType(op.succ1, op.succ2),
// CHECK-NEXT:          regions = ListType(op.reg1, op.reg2),
// CHECK-NEXT:          dictionaryProperties = DictType(("randProp1", op.randProp1.typ), ("randProp2", op.randProp2.typ)),
// CHECK-NEXT:          dictionaryAttributes = DictType(("randAttr1", op.randAttr1.typ), ("randAttr2", op.randAttr2.typ))
// CHECK-NEXT:        )

// CHECK:             op1.results.clear()
// CHECK-NEXT:        op1.results.addAll(ListType(op.result1, op.result2))
// CHECK-NEXT:        op1
// CHECK-NEXT:      }

// CHECK:           def verify(op: UnverifiedOp[Mul]): Mul = {

// CHECK:             if (op.operands.size != 2) then throw new Exception(s"Expected 2 operands, got ${op.operands.size}")
// CHECK-NEXT:        if (op.results.size != 2) then throw new Exception(s"Expected 2 results, got ${op.results.size}")
// CHECK-NEXT:        if (op.regions.size != 2) then throw new Exception(s"Expected 2 regions, got ${op.regions.size}")
// CHECK-NEXT:        if (op.successors.size != 2) then throw new Exception(s"Expected 2 successors, got ${op.successors.size}")
// CHECK-NEXT:        if (op.dictionaryProperties.size != 2) then throw new Exception(s"Expected 2 dictionaryProperties, got ${op.dictionaryProperties.size}")
// CHECK-NEXT:        if (op.dictionaryAttributes.size != 2) then throw new Exception(s"Expected 2 dictionaryAttributes, got ${op.dictionaryAttributes.size}")

// CHECK:             if !(op.dictionaryProperties.contains("randProp1") &&
// CHECK-NEXT:        op.dictionaryProperties.contains("randProp2") &&
// CHECK-NEXT:        op.dictionaryAttributes.contains("randAttr1") &&
// CHECK-NEXT:        op.dictionaryAttributes.contains("randAttr2"))
// CHECK-NEXT:        then throw new Exception("Expected specific properties and attributes")

// CHECK:             BaseAttr[scair.dialects.builtin.IntegerAttr]().verify(op.operands(0).typ, new ConstraintContext())
// CHECK-NEXT:        BaseAttr[scair.dialects.builtin.IntegerAttr]().verify(op.operands(1).typ, new ConstraintContext())
// CHECK-NEXT:        BaseAttr[scair.dialects.builtin.IntegerAttr]().verify(op.results(0).typ, new ConstraintContext())
// CHECK-NEXT:        BaseAttr[scair.dialects.builtin.IntegerAttr]().verify(op.results(1).typ, new ConstraintContext())
// CHECK-NEXT:        BaseAttr[scair.dialects.builtin.StringData]().verify(op.dictionaryProperties("randProp1"), new ConstraintContext())
// CHECK-NEXT:        BaseAttr[scair.dialects.builtin.StringData]().verify(op.dictionaryProperties("randProp2"), new ConstraintContext())
// CHECK-NEXT:        BaseAttr[scair.dialects.builtin.StringData]().verify(op.dictionaryAttributes("randAttr1"), new ConstraintContext())
// CHECK-NEXT:        BaseAttr[scair.dialects.builtin.StringData]().verify(op.dictionaryAttributes("randAttr2"), new ConstraintContext())

// CHECK:             Mul(
// CHECK-NEXT:          operand1 = op.operands(0).asInstanceOf[Value[scair.dialects.builtin.IntegerAttr]],
// CHECK-NEXT:          operand2 = op.operands(1).asInstanceOf[Value[scair.dialects.builtin.IntegerAttr]],
// CHECK-NEXT:          result1 = op.results(0).asInstanceOf[Value[scair.dialects.builtin.IntegerAttr]],
// CHECK-NEXT:          result2 = op.results(1).asInstanceOf[Value[scair.dialects.builtin.IntegerAttr]],
// CHECK-NEXT:          reg1 = op.regions(0),
// CHECK-NEXT:          reg2 = op.regions(1),
// CHECK-NEXT:          succ1 = op.successors(0),
// CHECK-NEXT:          succ2 = op.successors(1),
// CHECK-NEXT:          randProp1 = Property(op.dictionaryProperties("randProp1").asInstanceOf[scair.dialects.builtin.StringData]),
// CHECK-NEXT:          randProp2 = Property(op.dictionaryProperties("randProp2").asInstanceOf[scair.dialects.builtin.StringData]),
// CHECK-NEXT:          randAttr1 = Attr(op.dictionaryAttributes("randAttr1").asInstanceOf[scair.dialects.builtin.StringData]),
// CHECK-NEXT:          randAttr2 = Attr(op.dictionaryAttributes("randAttr2").asInstanceOf[scair.dialects.builtin.StringData])
// CHECK-NEXT:        )
// CHECK-NEXT:      }

// CHECK:         }
// CHECK-NEXT:  }

// CHECK:       object NormHelper extends ADTCompanion {
// CHECK-NEXT:    val getName: String = "cmath.norm"

// CHECK:         val getMLIRRealm: MLIRRealm[Norm] = new MLIRRealm[Norm] {

// CHECK:           def constructUnverifiedOp(
// CHECK-NEXT:          operands: ListType[Value[Attribute]] = ListType(),
// CHECK-NEXT:          successors: ListType[Block] = ListType(),
// CHECK-NEXT:          results_types: ListType[Attribute] = ListType(),
// CHECK-NEXT:          regions: ListType[Region] = ListType(),
// CHECK-NEXT:          dictionaryProperties: DictType[String, Attribute] = DictType.empty[String, Attribute],
// CHECK-NEXT:          dictionaryAttributes: DictType[String, Attribute] = DictType.empty[String, Attribute]
// CHECK-NEXT:      ): UnverifiedOp[Norm] = {
// CHECK-NEXT:        UnverifiedOp[Norm](
// CHECK-NEXT:          name = "cmath.norm",
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
// CHECK-NEXT:          name = "cmath.norm",
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

// CHECK:       val CMathDialect: DialectV2 =
// CHECK-NEXT:    new DialectV2(
// CHECK-NEXT:      operations = List(
// CHECK-NEXT:        MulHelper,
// CHECK-NEXT:        NormHelper
// CHECK-NEXT:      )
// CHECK-NEXT:    )
