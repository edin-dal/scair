// RUN: scala -classpath ../../target/scala-3.3.1/classes/ %s | filecheck %s

import scair.scairdl.constraints._
import scair.clair.mirrored._

enum CMath extends Dialect:
  case Norm(
      e1: Operand[AnyAttr.type],
      e2: Result[AnyAttr.type],
      e3: Region
  )
  case Mul(
      e1: Operand[AnyAttr.type],
      e2: Result[AnyAttr.type]
  )

object CMath {
  val generator = summonDialect[CMath]
}

def main(args: Array[String]): Unit = {
  println(CMath.generator.print(0))
}


// CHECK:       import scair.ir._
// CHECK-NEXT:  import scair.dialects.builtin._
// CHECK-NEXT:  import scair.scairdl.constraints._
// CHECK-NEXT:  import scair.scairdl.constraints.attr2constraint

// CHECK:       object Norm extends DialectOperation {
// CHECK-NEXT:    override def name = "cmath.norm"
// CHECK-NEXT:    override def factory = Norm.apply
// CHECK-NEXT:  }

// CHECK:       case class Norm(
// CHECK-NEXT:      override val operands: ListType[Value[Attribute]] = ListType(),
// CHECK-NEXT:      override val successors: ListType[Block] = ListType(),
// CHECK-NEXT:      override val results: ListType[Value[Attribute]] = ListType(),
// CHECK-NEXT:      override val regions: ListType[Region] = ListType(),
// CHECK-NEXT:      override val dictionaryProperties: DictType[String, Attribute] =
// CHECK-NEXT:        DictType.empty[String, Attribute],
// CHECK-NEXT:      override val dictionaryAttributes: DictType[String, Attribute] =
// CHECK-NEXT:        DictType.empty[String, Attribute]
// CHECK-NEXT:  ) extends RegisteredOperation(name = "cmath.norm") {

// CHECK:         def e1: Value[Attribute] = operands(0)
// CHECK-NEXT:    def e1_=(new_operand: Value[Attribute]): Unit = {operands(0) = new_operand}

// CHECK:         def e2: Value[Attribute] = results(0)
// CHECK-NEXT:    def e2_=(new_result: Value[Attribute]): Unit = {results(0) = new_result}

// CHECK:         def e3: Region = regions(0)
// CHECK-NEXT:    def e3_=(new_region: Region): Unit = {regions(0) = new_region}

// CHECK:         val e1_constr = AnyAttr
// CHECK-NEXT:    val e2_constr = AnyAttr

// CHECK:         override def custom_verify(): Unit =
// CHECK-NEXT:      val verification_context = new ConstraintContext()

// CHECK:           if (operands.length != 1) then throw new Exception(s"Expected 1 operands, got ${operands.length}")
// CHECK-NEXT:      if (results.length != 1) then throw new Exception(s"Expected 1 results, got ${results.length}")
// CHECK-NEXT:      if (regions.length != 1) then throw new Exception(s"Expected 1 regions, got ${regions.length}")
// CHECK-NEXT:      if (successors.length != 0) then throw new Exception(s"Expected 0 successors, got ${successors.length}")
// CHECK-NEXT:      if (dictionaryProperties.size != 0) then throw new Exception("Expected 0 properties, got dictionaryProperties.size")
// CHECK-NEXT:      if (dictionaryAttributes.size != 0) then throw new Exception("Expected 0 attributes, got dictionaryAttributes.size")

// CHECK:           e1_constr.verify(e1.typ, verification_context)
// CHECK-NEXT:      e2_constr.verify(e2.typ, verification_context)

// CHECK:       }

// CHECK:       object Mul extends DialectOperation {
// CHECK-NEXT:    override def name = "cmath.mul"
// CHECK-NEXT:    override def factory = Mul.apply
// CHECK-NEXT:  }

// CHECK:       case class Mul(
// CHECK-NEXT:      override val operands: ListType[Value[Attribute]] = ListType(),
// CHECK-NEXT:      override val successors: ListType[Block] = ListType(),
// CHECK-NEXT:      override val results: ListType[Value[Attribute]] = ListType(),
// CHECK-NEXT:      override val regions: ListType[Region] = ListType(),
// CHECK-NEXT:      override val dictionaryProperties: DictType[String, Attribute] =
// CHECK-NEXT:        DictType.empty[String, Attribute],
// CHECK-NEXT:      override val dictionaryAttributes: DictType[String, Attribute] =
// CHECK-NEXT:        DictType.empty[String, Attribute]
// CHECK-NEXT:  ) extends RegisteredOperation(name = "cmath.mul") {

// CHECK:         def e1: Value[Attribute] = operands(0)
// CHECK-NEXT:    def e1_=(new_operand: Value[Attribute]): Unit = {operands(0) = new_operand}

// CHECK:         def e2: Value[Attribute] = results(0)
// CHECK-NEXT:    def e2_=(new_result: Value[Attribute]): Unit = {results(0) = new_result}

// CHECK:         val e1_constr = AnyAttr
// CHECK-NEXT:    val e2_constr = AnyAttr

// CHECK:         override def custom_verify(): Unit =
// CHECK-NEXT:      val verification_context = new ConstraintContext()

// CHECK:           if (operands.length != 1) then throw new Exception(s"Expected 1 operands, got ${operands.length}")
// CHECK-NEXT:      if (results.length != 1) then throw new Exception(s"Expected 1 results, got ${results.length}")
// CHECK-NEXT:      if (regions.length != 0) then throw new Exception(s"Expected 0 regions, got ${regions.length}")
// CHECK-NEXT:      if (successors.length != 0) then throw new Exception(s"Expected 0 successors, got ${successors.length}")
// CHECK-NEXT:      if (dictionaryProperties.size != 0) then throw new Exception("Expected 0 properties, got dictionaryProperties.size")
// CHECK-NEXT:      if (dictionaryAttributes.size != 0) then throw new Exception("Expected 0 attributes, got dictionaryAttributes.size")

// CHECK:           e1_constr.verify(e1.typ, verification_context)
// CHECK-NEXT:      e2_constr.verify(e2.typ, verification_context)

// CHECK:       }

// CHECK:       val cmath: Dialect = new Dialect(
// CHECK-NEXT:    operations = Seq(Norm, Mul),
// CHECK-NEXT:    attributes = Seq()
// CHECK-NEXT:  )