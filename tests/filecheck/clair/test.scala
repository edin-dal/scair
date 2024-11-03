// RUN: scala -classpath ../../target/scala-3.3.1/classes/ %s | filecheck %s

import scair.clair.ir._
import scair.dialects.builtin.IntData
import scair.scairdl.constraints._

object Main {
  def main(args: Array[String]) = {
    val dialect = DialectDef(
      "test",
      ListType(
        OperationDef(
          "test.no_variadics",
          "NoVariadicsOp",
          operands = List(
            OperandDef("sing_op1", BaseAttr[IntData]()),
            OperandDef(
              "sing_op2",
              AnyOf(Seq(EqualAttr(IntData(5)), EqualAttr(IntData(6))))
            )
          ),
          regions = List(RegionDef("region1")),
          successors = List(SuccessorDef("successor1"))
        ),
        OperationDef(
          "test.variadic_operand",
          "VariadicOperandOp",
          operands = List(
            OperandDef("sing_op1", BaseAttr[IntData]()),
            OperandDef("var_op1", EqualAttr(IntData(5)), Variadicity.Variadic),
            OperandDef(
              "sing_op2",
              AnyOf(Seq(EqualAttr(IntData(5)), EqualAttr(IntData(6))))
            )
          ),
          regions = List(RegionDef("region1")),
          successors = List(SuccessorDef("successor1"))
        )
      ),
      ListType(AttributeDef("test.type", "TypeAttr", typee = 1))
    )

    println(dialect.print(0))
  }
}

// CHECK:       import scair.ir._
// CHECK-NEXT:  import scair.dialects.builtin._
// CHECK-NEXT:  import scair.scairdl.constraints._

// CHECK:       object NoVariadicsOp extends DialectOperation {
// CHECK-NEXT:    override def name = "test.no_variadics"
// CHECK-NEXT:    override def factory = NoVariadicsOp.apply
// CHECK-NEXT:  }

// CHECK:       case class NoVariadicsOp(
// CHECK-NEXT:      override val operands: ListType[Value[Attribute]] = ListType(),
// CHECK-NEXT:      override val successors: ListType[Block] = ListType(),
// CHECK-NEXT:      override val results: ListType[Value[Attribute]] = ListType(),
// CHECK-NEXT:      override val regions: ListType[Region] = ListType(),
// CHECK-NEXT:      override val dictionaryProperties: DictType[String, Attribute] =
// CHECK-NEXT:        DictType.empty[String, Attribute],
// CHECK-NEXT:      override val dictionaryAttributes: DictType[String, Attribute] =
// CHECK-NEXT:        DictType.empty[String, Attribute]
// CHECK-NEXT:  ) extends RegisteredOperation(name = "test.no_variadics") {

// CHECK:         def sing_op1: Value[Attribute] = operands(0)
// CHECK-NEXT:    def sing_op1_=(value: Value[Attribute]): Unit = {operands(0) = value}

// CHECK:         def sing_op2: Value[Attribute] = operands(1)
// CHECK-NEXT:    def sing_op2_=(value: Value[Attribute]): Unit = {operands(1) = value}

// CHECK:         def region1: Region = regions(0)
// CHECK-NEXT:    def region1_=(value: Region): Unit = {regions(0) = value}

// CHECK:         def successor1: Block = successors(0)
// CHECK-NEXT:    def successor1_=(value: Block): Unit = {successors(0) = value}

// CHECK:         val sing_op1_constr = BaseAttr[scair.dialects.builtin.IntData]()
// CHECK-NEXT:    val sing_op2_constr = AnyOf(List(EqualAttr(IntData(5)), EqualAttr(IntData(6))))

// CHECK:         override def custom_verify(): Unit =
// CHECK-NEXT:      val verification_context = new ConstraintContext()

// CHECK:           if (operands.length != 2) then throw new Exception(s"Expected 2 operands, got ${operands.length}")
// CHECK-NEXT:      if (results.length != 0) then throw new Exception("Expected 0 results, got results.length")
// CHECK-NEXT:      if (regions.length != 1) then throw new Exception("Expected 1 regions, got regions.length")
// CHECK-NEXT:      if (successors.length != 1) then throw new Exception("Expected 1 successors, got successors.length")
// CHECK-NEXT:      if (dictionaryProperties.size != 0) then throw new Exception("Expected 0 properties, got dictionaryProperties.size")
// CHECK-NEXT:      if (dictionaryAttributes.size != 0) then throw new Exception("Expected 0 attributes, got dictionaryAttributes.size")

// CHECK:           sing_op1_constr.verify(sing_op1.typ, verification_context)
// CHECK-NEXT:      sing_op2_constr.verify(sing_op2.typ, verification_context)

// CHECK:       }
// CHECK:       object VariadicOperandOp extends DialectOperation {

// CHECK-NEXT:    override def name = "test.variadic_operand"
// CHECK-NEXT:    override def factory = VariadicOperandOp.apply
// CHECK-NEXT:  }

// CHECK:       case class VariadicOperandOp(
// CHECK-NEXT:      override val operands: ListType[Value[Attribute]] = ListType(),
// CHECK-NEXT:      override val successors: ListType[Block] = ListType(),
// CHECK-NEXT:      override val results: ListType[Value[Attribute]] = ListType(),
// CHECK-NEXT:      override val regions: ListType[Region] = ListType(),
// CHECK-NEXT:      override val dictionaryProperties: DictType[String, Attribute] =
// CHECK-NEXT:        DictType.empty[String, Attribute],
// CHECK-NEXT:      override val dictionaryAttributes: DictType[String, Attribute] =
// CHECK-NEXT:        DictType.empty[String, Attribute]
// CHECK-NEXT:  ) extends RegisteredOperation(name = "test.variadic_operand") {

// CHECK:         def sing_op1: Value[Attribute] = operands(0)
// CHECK-NEXT:    def sing_op1_=(value: Value[Attribute]): Unit = {operands(0) = value}

// CHECK:         def var_op1: Seq[Value[Attribute]] = operands.slice(1, operands.length - 1).toSeq
// CHECK-NEXT:    def var_op1_=(values: Seq[Value[Attribute]]): Unit = {
// CHECK-NEXT:      val diff = values.length - (operands.length - 2)
// CHECK-NEXT:      for (value, i) <- (values ++ operands.slice(1, operands.length)).zipWithIndex do
// CHECK-NEXT:        operands(i + 1) = value
// CHECK-NEXT:      if (diff < 0)
// CHECK-NEXT:        operands.trimEnd(-diff)
// CHECK-NEXT:    }

// CHECK:         def sing_op2: Value[Attribute] = operands(operands.length - 1)
// CHECK-NEXT:    def sing_op2_=(value: Value[Attribute]): Unit = {operands(operands.length - 1) = value}

// CHECK:         def region1: Region = regions(0)
// CHECK-NEXT:    def region1_=(value: Region): Unit = {regions(0) = value}

// CHECK:         def successor1: Block = successors(0)
// CHECK-NEXT:    def successor1_=(value: Block): Unit = {successors(0) = value}

// CHECK:         val sing_op1_constr = BaseAttr[scair.dialects.builtin.IntData]()
// CHECK-NEXT:    val var_op1_constr = EqualAttr(IntData(5))
// CHECK-NEXT:    val sing_op2_constr = AnyOf(List(EqualAttr(IntData(5)), EqualAttr(IntData(6))))

// CHECK:         override def custom_verify(): Unit =
// CHECK-NEXT:      val verification_context = new ConstraintContext()

// CHECK:           if (operands.length < 2) then throw new Exception(s"Expected at least 2 operands, got ${operands.length}")
// CHECK-NEXT:      if (results.length != 0) then throw new Exception("Expected 0 results, got results.length")
// CHECK-NEXT:      if (regions.length != 1) then throw new Exception("Expected 1 regions, got regions.length")
// CHECK-NEXT:      if (successors.length != 1) then throw new Exception("Expected 1 successors, got successors.length")
// CHECK-NEXT:      if (dictionaryProperties.size != 0) then throw new Exception("Expected 0 properties, got dictionaryProperties.size")
// CHECK-NEXT:      if (dictionaryAttributes.size != 0) then throw new Exception("Expected 0 attributes, got dictionaryAttributes.size")

// CHECK:           sing_op1_constr.verify(sing_op1.typ, verification_context)
// CHECK-NEXT:      var_op1_constr.verify(var_op1.typ, verification_context)
// CHECK-NEXT:      sing_op2_constr.verify(sing_op2.typ, verification_context)

// CHECK:       }

// CHECK:       object TypeAttr extends DialectAttribute {
// CHECK-NEXT:    override def name = "test.type"
// CHECK-NEXT:    override def factory = TypeAttr.apply
// CHECK-NEXT:  }

// CHECK:       case class TypeAttr(override val parameters: Seq[Attribute]) extends ParametrizedAttribute(name = "test.type", parameters = parameters) with TypeAttribute {
// CHECK-NEXT:    override def custom_verify(): Unit =
// CHECK-NEXT:      if (parameters.length != 0) then throw new Exception("Expected 0 parameters, got parameters.length")
// CHECK-NEXT:  }

// CHECK:       val test: Dialect = new Dialect(
// CHECK-NEXT:    operations = Seq(NoVariadicsOp, VariadicOperandOp),
// CHECK-NEXT:    attributes = Seq(TypeAttr)
// CHECK-NEXT:  )
