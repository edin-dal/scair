// RUN: scala full-classpath %s | filecheck %s

import scair.scairdl.irdef._
import scair.dialects.builtin.IntData
import scair.scairdl.constraints._
import scair.scairdl.constraints.attr2constraint

object Main {

  def main(args: Array[String]) = {
    val dialect = DialectDef(
      "test",
      Seq(
        OperationDef(
          "test.no_variadics",
          "NoVariadicsOp",
          operands = List(
            OperandDef("sing_op1", BaseAttr[IntData]()),
            OperandDef(
              "sing_op2",
              IntData(5) || IntData(6)
            )
          ),
          results = List(
            ResultDef("sing_res1"),
            ResultDef("sing_res2")
          ),
          regions = List(RegionDef("region1"), RegionDef("region2")),
          successors =
            List(SuccessorDef("successor1"), SuccessorDef("successor2"))
        ),
        OperationDef(
          "test.variadic_operand",
          "VariadicOperandOp",
          operands = List(
            OperandDef("sing_op1", BaseAttr[IntData]()),
            OperandDef("var_op1", IntData(5), Variadicity.Variadic),
            OperandDef(
              "sing_op2",
              IntData(5) || IntData(6)
            )
          ),
          results = List(
            ResultDef("sing_res1"),
            ResultDef("var_res1", variadicity = Variadicity.Variadic),
            ResultDef("sing_res2")
          ),
          regions = List(
            RegionDef("region1"),
            RegionDef("var_region1", variadicity = Variadicity.Variadic),
            RegionDef("region2")
          ),
          successors = List(
            SuccessorDef("successor1"),
            SuccessorDef("var_successor", variadicity = Variadicity.Variadic),
            SuccessorDef("successor2")
          )
        ),
        OperationDef(
          "test.multi_variadic_operand",
          "MultiVariadicOperandOp",
          operands = List(
            OperandDef("sing_op1", BaseAttr[IntData]()),
            OperandDef("var_op1", IntData(5), Variadicity.Variadic),
            OperandDef("var_op2", IntData(5), Variadicity.Variadic),
            OperandDef(
              "sing_op2",
              IntData(5) || IntData(6)
            )
          ),
          results = List(
            ResultDef("sing_res1"),
            ResultDef("var_res1", variadicity = Variadicity.Variadic),
            ResultDef("var_res2", variadicity = Variadicity.Variadic),
            ResultDef("sing_res2")
          ),
          regions = List(
            RegionDef("region1"),
            RegionDef("var_region1", variadicity = Variadicity.Variadic),
            RegionDef("region2")
          ),
          successors = List(
            SuccessorDef("successor1"),
            SuccessorDef("var_successor", variadicity = Variadicity.Variadic),
            SuccessorDef("successor2")
          )
        )
      ),
      Seq(AttributeDef("test.type", "TypeAttr", typee = 1))
    )

    println(dialect.print(0))
  }

}

// CHECK:       package scair.dialects.test

// CHECK:       import scair.ir._
// CHECK-NEXT:  import scair.Parser
// CHECK-NEXT:  import scair.Parser.whitespace

// CHECK:       import scair.dialects.builtin._
// CHECK-NEXT:  import scair.scairdl.constraints._
// CHECK-NEXT:  import scair.scairdl.constraints.attr2constraint

// CHECK:       object NoVariadicsOp extends MLIROperationObject {
// CHECK-NEXT:    override def name = "test.no_variadics"
// CHECK-NEXT:    override def factory = NoVariadicsOp.apply

// CHECK:       }

// CHECK:       case class NoVariadicsOp(
// CHECK-NEXT:      override val operands: ListType[Value[Attribute]] = ListType(),
// CHECK-NEXT:      override val successors: ListType[Block] = ListType(),
// CHECK-NEXT:      results_types: ListType[Attribute] = ListType(),
// CHECK-NEXT:      override val regions: ListType[Region] = ListType(),
// CHECK-NEXT:      override val dictionaryProperties: DictType[String, Attribute] =
// CHECK-NEXT:        DictType.empty[String, Attribute],
// CHECK-NEXT:      override val dictionaryAttributes: DictType[String, Attribute] =
// CHECK-NEXT:        DictType.empty[String, Attribute]
// CHECK-NEXT:  ) extends RegisteredOperation(name = "test.no_variadics", operands, successors, results_types, regions, dictionaryProperties, dictionaryAttributes) {

// CHECK:         def sing_op1: Value[Attribute] = operands(0)
// CHECK-NEXT:    def sing_op1_=(new_operand: Value[Attribute]): Unit = {operands(0) = new_operand}

// CHECK:         def sing_op2: Value[Attribute] = operands(1)
// CHECK-NEXT:    def sing_op2_=(new_operand: Value[Attribute]): Unit = {operands(1) = new_operand}

// CHECK:         def sing_res1: Value[Attribute] = results(0)
// CHECK-NEXT:    def sing_res1_=(new_result: Value[Attribute]): Unit = {results(0) = new_result}

// CHECK:         def sing_res2: Value[Attribute] = results(1)
// CHECK-NEXT:    def sing_res2_=(new_result: Value[Attribute]): Unit = {results(1) = new_result}

// CHECK:         def region1: Region = regions(0)
// CHECK-NEXT:    def region1_=(new_region: Region): Unit = {regions(0) = new_region}

// CHECK:         def region2: Region = regions(1)
// CHECK-NEXT:    def region2_=(new_region: Region): Unit = {regions(1) = new_region}

// CHECK:         def successor1: Block = successors(0)
// CHECK-NEXT:    def successor1_=(new_successor: Block): Unit = {successors(0) = new_successor}

// CHECK:         def successor2: Block = successors(1)
// CHECK-NEXT:    def successor2_=(new_successor: Block): Unit = {successors(1) = new_successor}

// CHECK:         val sing_op1_constr = BaseAttr[scair.dialects.builtin.IntData]()
// CHECK-NEXT:    val sing_op2_constr = IntData(5) || IntData(6)
// CHECK-NEXT:    val sing_res1_constr = AnyAttr
// CHECK-NEXT:    val sing_res2_constr = AnyAttr

// CHECK:         override def custom_verify(): Unit =
// CHECK-NEXT:      val verification_context = new ConstraintContext()

// CHECK:           if (operands.length != 2) then throw new Exception(s"Expected 2 operands, got ${operands.length}")
// CHECK-NEXT:      if (results.length != 2) then throw new Exception(s"Expected 2 results, got ${results.length}")
// CHECK-NEXT:      if (regions.length != 2) then throw new Exception(s"Expected 2 regions, got ${regions.length}")
// CHECK-NEXT:      if (successors.length != 2) then throw new Exception(s"Expected 2 successors, got ${successors.length}")

// CHECK:           sing_op1_constr.verify(sing_op1.typ, verification_context)
// CHECK-NEXT:      sing_op2_constr.verify(sing_op2.typ, verification_context)
// CHECK-NEXT:      sing_res1_constr.verify(sing_res1.typ, verification_context)
// CHECK-NEXT:      sing_res2_constr.verify(sing_res2.typ, verification_context)

// CHECK:       }

// CHECK:       object VariadicOperandOp extends MLIROperationObject {
// CHECK-NEXT:    override def name = "test.variadic_operand"
// CHECK-NEXT:    override def factory = VariadicOperandOp.apply

// CHECK:       }

// CHECK:       case class VariadicOperandOp(
// CHECK-NEXT:      override val operands: ListType[Value[Attribute]] = ListType(),
// CHECK-NEXT:      override val successors: ListType[Block] = ListType(),
// CHECK-NEXT:      results_types: ListType[Attribute] = ListType(),
// CHECK-NEXT:      override val regions: ListType[Region] = ListType(),
// CHECK-NEXT:      override val dictionaryProperties: DictType[String, Attribute] =
// CHECK-NEXT:        DictType.empty[String, Attribute],
// CHECK-NEXT:      override val dictionaryAttributes: DictType[String, Attribute] =
// CHECK-NEXT:        DictType.empty[String, Attribute]
// CHECK-NEXT:  ) extends RegisteredOperation(name = "test.variadic_operand", operands, successors, results_types, regions, dictionaryProperties, dictionaryAttributes) {

// CHECK:         def sing_op1: Value[Attribute] = operands(0)
// CHECK-NEXT:    def sing_op1_=(new_operand: Value[Attribute]): Unit = {operands(0) = new_operand}

// CHECK:         def var_op1: Seq[Value[Attribute]] = {
// CHECK-NEXT:        val from = 1
// CHECK-NEXT:        val to = operands.length - 1
// CHECK-NEXT:        operands.slice(from, to).toSeq
// CHECK-NEXT:    }
// CHECK-NEXT:    def var_op1_=(new_operands: Seq[Value[Attribute]]): Unit = {
// CHECK-NEXT:      val from = 1
// CHECK-NEXT:      val to = operands.length - 1
// CHECK-NEXT:      val diff = new_operands.length - (to - from)
// CHECK-NEXT:      for (operand, i) <- (new_operands ++ operands.slice(to, operands.length)).zipWithIndex do
// CHECK-NEXT:        operands(from + i) = operand
// CHECK-NEXT:      if (diff < 0)
// CHECK-NEXT:        operands.trimEnd(-diff)
// CHECK-NEXT:    }

// CHECK:         def sing_op2: Value[Attribute] = operands(operands.length - 1)
// CHECK-NEXT:    def sing_op2_=(new_operand: Value[Attribute]): Unit = {operands(operands.length - 1) = new_operand}

// CHECK:         def sing_res1: Value[Attribute] = results(0)
// CHECK-NEXT:    def sing_res1_=(new_result: Value[Attribute]): Unit = {results(0) = new_result}

// CHECK:         def var_res1: Seq[Value[Attribute]] = {
// CHECK-NEXT:        val from = 1
// CHECK-NEXT:        val to = results.length - 1
// CHECK-NEXT:        results.slice(from, to).toSeq
// CHECK-NEXT:    }
// CHECK-NEXT:    def var_res1_=(new_results: Seq[Value[Attribute]]): Unit = {
// CHECK-NEXT:      val from = 1
// CHECK-NEXT:      val to = results.length - 1
// CHECK-NEXT:      val diff = new_results.length - (to - from)
// CHECK-NEXT:      for (new_results, i) <- (new_results ++ results.slice(to, results.length)).zipWithIndex do
// CHECK-NEXT:        results(from + i) = new_results
// CHECK-NEXT:      if (diff < 0)
// CHECK-NEXT:        results.trimEnd(-diff)
// CHECK-NEXT:    }

// CHECK:         def sing_res2: Value[Attribute] = results(results.length - 1)
// CHECK-NEXT:    def sing_res2_=(new_result: Value[Attribute]): Unit = {results(results.length - 1) = new_result}

// CHECK:         def region1: Region = regions(0)
// CHECK-NEXT:    def region1_=(new_region: Region): Unit = {regions(0) = new_region}

// CHECK:         def var_region1: Seq[Region] = {
// CHECK-NEXT:        val from = 1
// CHECK-NEXT:        val to = regions.length - 1
// CHECK-NEXT:        regions.slice(from, to).toSeq
// CHECK-NEXT:    }
// CHECK-NEXT:    def var_region1_=(new_regions: Seq[Region]): Unit = {
// CHECK-NEXT:      val from = 1
// CHECK-NEXT:      val to = regions.length - 1
// CHECK-NEXT:      val diff = new_regions.length - (to - from)
// CHECK-NEXT:      for (region, i) <- (new_regions ++ regions.slice(to, regions.length)).zipWithIndex do
// CHECK-NEXT:        regions(from + i) = region
// CHECK-NEXT:      if (diff < 0)
// CHECK-NEXT:        regions.trimEnd(-diff)
// CHECK-NEXT:    }

// CHECK:         def region2: Region = regions(regions.length - 1)
// CHECK-NEXT:    def region2_=(new_region: Region): Unit = {regions(regions.length - 1) = new_region}

// CHECK:         def successor1: Block = successors(0)
// CHECK-NEXT:    def successor1_=(new_successor: Block): Unit = {successors(0) = new_successor}

// CHECK:         def var_successor: Seq[Block] = {
// CHECK-NEXT:        val from = 1
// CHECK-NEXT:        val to = successors.length - 1
// CHECK-NEXT:        successors.slice(from, to).toSeq
// CHECK-NEXT:    }
// CHECK-NEXT:    def var_successor_=(new_successors: Seq[Block]): Unit = {
// CHECK-NEXT:      val from = 1
// CHECK-NEXT:      val to = successors.length - 1
// CHECK-NEXT:      val diff = new_successors.length - (to - from)
// CHECK-NEXT:      for (successor, i) <- (new_successors ++ successors.slice(to, successors.length)).zipWithIndex do
// CHECK-NEXT:        successors(from + i) = successor
// CHECK-NEXT:      if (diff < 0)
// CHECK-NEXT:        successors.trimEnd(-diff)
// CHECK-NEXT:    }

// CHECK:         def successor2: Block = successors(successors.length - 1)
// CHECK-NEXT:    def successor2_=(new_successor: Block): Unit = {successors(successors.length - 1) = new_successor}

// CHECK:         val sing_op1_constr = BaseAttr[scair.dialects.builtin.IntData]()
// CHECK-NEXT:    val var_op1_constr = IntData(5)
// CHECK-NEXT:    val sing_op2_constr = IntData(5) || IntData(6)
// CHECK-NEXT:    val sing_res1_constr = AnyAttr
// CHECK-NEXT:    val var_res1_constr = AnyAttr
// CHECK-NEXT:    val sing_res2_constr = AnyAttr

// CHECK:         override def custom_verify(): Unit =
// CHECK-NEXT:      val verification_context = new ConstraintContext()

// CHECK:           if (operands.length < 2) then throw new Exception(s"Expected at least 2 operands, got ${operands.length}")
// CHECK-NEXT:      if (results.length < 2) then throw new Exception(s"Expected at least 2 results, got ${results.length}")
// CHECK-NEXT:      if (regions.length < 2) then throw new Exception(s"Expected at least 2 regions, got ${regions.length}")
// CHECK-NEXT:      if (successors.length < 2) then throw new Exception(s"Expected at least 2 successors, got ${successors.length}")

// CHECK:           sing_op1_constr.verify(sing_op1.typ, verification_context)
// CHECK-NEXT:      var_op1_constr.verify(var_op1.typ, verification_context)
// CHECK-NEXT:      sing_op2_constr.verify(sing_op2.typ, verification_context)
// CHECK-NEXT:      sing_res1_constr.verify(sing_res1.typ, verification_context)
// CHECK-NEXT:      var_res1_constr.verify(var_res1.typ, verification_context)
// CHECK-NEXT:      sing_res2_constr.verify(sing_res2.typ, verification_context)

// CHECK:       }

// CHECK:       object MultiVariadicOperandOp extends MLIROperationObject {
// CHECK-NEXT:    override def name = "test.multi_variadic_operand"
// CHECK-NEXT:    override def factory = MultiVariadicOperandOp.apply

// CHECK:       }

// CHECK:       case class MultiVariadicOperandOp(
// CHECK-NEXT:      override val operands: ListType[Value[Attribute]] = ListType(),
// CHECK-NEXT:      override val successors: ListType[Block] = ListType(),
// CHECK-NEXT:      results_types: ListType[Attribute] = ListType(),
// CHECK-NEXT:      override val regions: ListType[Region] = ListType(),
// CHECK-NEXT:      override val dictionaryProperties: DictType[String, Attribute] =
// CHECK-NEXT:        DictType.empty[String, Attribute],
// CHECK-NEXT:      override val dictionaryAttributes: DictType[String, Attribute] =
// CHECK-NEXT:        DictType.empty[String, Attribute]
// CHECK-NEXT:  ) extends RegisteredOperation(name = "test.multi_variadic_operand", operands, successors, results_types, regions, dictionaryProperties, dictionaryAttributes) {

// CHECK:         def operandSegmentSizes: Seq[Int] =
// CHECK-NEXT:      if (!dictionaryProperties.contains("operandSegmentSizes")) then throw new Exception("Expected operandSegmentSizes property")
// CHECK-NEXT:      val operandSegmentSizes_attr = dictionaryProperties("operandSegmentSizes") match {
// CHECK-NEXT:        case right: DenseArrayAttr => right
// CHECK-NEXT:        case _ => throw new Exception("Expected operandSegmentSizes to be a DenseArrayAttr")
// CHECK-NEXT:      }
// CHECK-NEXT:      ParametrizedAttrConstraint[scair.dialects.builtin.DenseArrayAttr](List(IntegerType(IntData(32),Signless), BaseAttr[scair.dialects.builtin.IntegerAttr]() && ParametrizedAttrConstraint[scair.dialects.builtin.IntegerAttr](List(BaseAttr[scair.dialects.builtin.IntData](), IntegerType(IntData(32),Signless))))).verify(operandSegmentSizes_attr, ConstraintContext())
// CHECK-NEXT:      if (operandSegmentSizes_attr.length != 4) then throw new Exception(s"Expected operandSegmentSizes to have 4 elements, got ${operandSegmentSizes_attr.length}")

// CHECK:           for (s <- operandSegmentSizes_attr) yield s match {
// CHECK-NEXT:        case right: IntegerAttr => right.value.data.toInt
// CHECK-NEXT:        case _ => throw new Exception("Unreachable exception as per above constraint check.")
// CHECK-NEXT:      }

// CHECK:         def resultSegmentSizes: Seq[Int] =
// CHECK-NEXT:      if (!dictionaryProperties.contains("resultSegmentSizes")) then throw new Exception("Expected resultSegmentSizes property")
// CHECK-NEXT:      val resultSegmentSizes_attr = dictionaryProperties("resultSegmentSizes") match {
// CHECK-NEXT:        case right: DenseArrayAttr => right
// CHECK-NEXT:        case _ => throw new Exception("Expected resultSegmentSizes to be a DenseArrayAttr")
// CHECK-NEXT:      }
// CHECK-NEXT:      ParametrizedAttrConstraint[scair.dialects.builtin.DenseArrayAttr](List(IntegerType(IntData(32),Signless), BaseAttr[scair.dialects.builtin.IntegerAttr]() && ParametrizedAttrConstraint[scair.dialects.builtin.IntegerAttr](List(BaseAttr[scair.dialects.builtin.IntData](), IntegerType(IntData(32),Signless))))).verify(resultSegmentSizes_attr, ConstraintContext())
// CHECK-NEXT:      if (resultSegmentSizes_attr.length != 4) then throw new Exception(s"Expected resultSegmentSizes to have 4 elements, got ${resultSegmentSizes_attr.length}")

// CHECK:           for (s <- resultSegmentSizes_attr) yield s match {
// CHECK-NEXT:        case right: IntegerAttr => right.value.data.toInt
// CHECK-NEXT:        case _ => throw new Exception("Unreachable exception as per above constraint check.")
// CHECK-NEXT:      }

// CHECK:         def sing_op1: Value[Attribute] = operands(operandSegmentSizes.slice(0, 0).fold(0)(_ + _))
// CHECK-NEXT:    def sing_op1_=(new_operand: Value[Attribute]): Unit = {operands(operandSegmentSizes.slice(0, 0).fold(0)(_ + _)) = new_operand}

// CHECK:         def var_op1: Seq[Value[Attribute]] = {
// CHECK-NEXT:        val from = operandSegmentSizes.slice(0, 1).fold(0)(_ + _)
// CHECK-NEXT:        val to = from + operandSegmentSizes(1)
// CHECK-NEXT:        operands.slice(from, to).toSeq
// CHECK-NEXT:    }
// CHECK-NEXT:    def var_op1_=(new_operands: Seq[Value[Attribute]]): Unit = {
// CHECK-NEXT:      val from = operandSegmentSizes.slice(0, 1).fold(0)(_ + _)
// CHECK-NEXT:      val to = from + operandSegmentSizes(1)
// CHECK-NEXT:      val diff = new_operands.length - (to - from)
// CHECK-NEXT:      for (operand, i) <- (new_operands ++ operands.slice(to, operands.length)).zipWithIndex do
// CHECK-NEXT:        operands(from + i) = operand
// CHECK-NEXT:      if (diff < 0)
// CHECK-NEXT:        operands.trimEnd(-diff)
// CHECK-NEXT:    }

// CHECK:         def var_op2: Seq[Value[Attribute]] = {
// CHECK-NEXT:        val from = operandSegmentSizes.slice(0, 2).fold(0)(_ + _)
// CHECK-NEXT:        val to = from + operandSegmentSizes(2)
// CHECK-NEXT:        operands.slice(from, to).toSeq
// CHECK-NEXT:    }
// CHECK-NEXT:    def var_op2_=(new_operands: Seq[Value[Attribute]]): Unit = {
// CHECK-NEXT:      val from = operandSegmentSizes.slice(0, 2).fold(0)(_ + _)
// CHECK-NEXT:      val to = from + operandSegmentSizes(2)
// CHECK-NEXT:      val diff = new_operands.length - (to - from)
// CHECK-NEXT:      for (operand, i) <- (new_operands ++ operands.slice(to, operands.length)).zipWithIndex do
// CHECK-NEXT:        operands(from + i) = operand
// CHECK-NEXT:      if (diff < 0)
// CHECK-NEXT:        operands.trimEnd(-diff)
// CHECK-NEXT:    }

// CHECK:         def sing_op2: Value[Attribute] = operands(operandSegmentSizes.slice(0, 3).fold(0)(_ + _))
// CHECK-NEXT:    def sing_op2_=(new_operand: Value[Attribute]): Unit = {operands(operandSegmentSizes.slice(0, 3).fold(0)(_ + _)) = new_operand}

// CHECK:         def sing_res1: Value[Attribute] = results(resultSegmentSizes.slice(0, 0).fold(0)(_ + _))
// CHECK-NEXT:    def sing_res1_=(new_result: Value[Attribute]): Unit = {results(resultSegmentSizes.slice(0, 0).fold(0)(_ + _)) = new_result}

// CHECK:         def var_res1: Seq[Value[Attribute]] = {
// CHECK-NEXT:        val from = resultSegmentSizes.slice(0, 1).fold(0)(_ + _)
// CHECK-NEXT:        val to = from + resultSegmentSizes(1)
// CHECK-NEXT:        results.slice(from, to).toSeq
// CHECK-NEXT:    }
// CHECK-NEXT:    def var_res1_=(new_results: Seq[Value[Attribute]]): Unit = {
// CHECK-NEXT:      val from = resultSegmentSizes.slice(0, 1).fold(0)(_ + _)
// CHECK-NEXT:      val to = from + resultSegmentSizes(1)
// CHECK-NEXT:      val diff = new_results.length - (to - from)
// CHECK-NEXT:      for (new_results, i) <- (new_results ++ results.slice(to, results.length)).zipWithIndex do
// CHECK-NEXT:        results(from + i) = new_results
// CHECK-NEXT:      if (diff < 0)
// CHECK-NEXT:        results.trimEnd(-diff)
// CHECK-NEXT:    }

// CHECK:         def var_res2: Seq[Value[Attribute]] = {
// CHECK-NEXT:        val from = resultSegmentSizes.slice(0, 2).fold(0)(_ + _)
// CHECK-NEXT:        val to = from + resultSegmentSizes(2)
// CHECK-NEXT:        results.slice(from, to).toSeq
// CHECK-NEXT:    }
// CHECK-NEXT:    def var_res2_=(new_results: Seq[Value[Attribute]]): Unit = {
// CHECK-NEXT:      val from = resultSegmentSizes.slice(0, 2).fold(0)(_ + _)
// CHECK-NEXT:      val to = from + resultSegmentSizes(2)
// CHECK-NEXT:      val diff = new_results.length - (to - from)
// CHECK-NEXT:      for (new_results, i) <- (new_results ++ results.slice(to, results.length)).zipWithIndex do
// CHECK-NEXT:        results(from + i) = new_results
// CHECK-NEXT:      if (diff < 0)
// CHECK-NEXT:        results.trimEnd(-diff)
// CHECK-NEXT:    }

// CHECK:         def sing_res2: Value[Attribute] = results(resultSegmentSizes.slice(0, 3).fold(0)(_ + _))
// CHECK-NEXT:    def sing_res2_=(new_result: Value[Attribute]): Unit = {results(resultSegmentSizes.slice(0, 3).fold(0)(_ + _)) = new_result}

// CHECK:         def region1: Region = regions(0)
// CHECK-NEXT:    def region1_=(new_region: Region): Unit = {regions(0) = new_region}

// CHECK:         def var_region1: Seq[Region] = {
// CHECK-NEXT:        val from = 1
// CHECK-NEXT:        val to = regions.length - 1
// CHECK-NEXT:        regions.slice(from, to).toSeq
// CHECK-NEXT:    }
// CHECK-NEXT:    def var_region1_=(new_regions: Seq[Region]): Unit = {
// CHECK-NEXT:      val from = 1
// CHECK-NEXT:      val to = regions.length - 1
// CHECK-NEXT:      val diff = new_regions.length - (to - from)
// CHECK-NEXT:      for (region, i) <- (new_regions ++ regions.slice(to, regions.length)).zipWithIndex do
// CHECK-NEXT:        regions(from + i) = region
// CHECK-NEXT:      if (diff < 0)
// CHECK-NEXT:        regions.trimEnd(-diff)
// CHECK-NEXT:    }

// CHECK:         def region2: Region = regions(regions.length - 1)
// CHECK-NEXT:    def region2_=(new_region: Region): Unit = {regions(regions.length - 1) = new_region}

// CHECK:         def successor1: Block = successors(0)
// CHECK-NEXT:    def successor1_=(new_successor: Block): Unit = {successors(0) = new_successor}

// CHECK:         def var_successor: Seq[Block] = {
// CHECK-NEXT:        val from = 1
// CHECK-NEXT:        val to = successors.length - 1
// CHECK-NEXT:        successors.slice(from, to).toSeq
// CHECK-NEXT:    }
// CHECK-NEXT:    def var_successor_=(new_successors: Seq[Block]): Unit = {
// CHECK-NEXT:      val from = 1
// CHECK-NEXT:      val to = successors.length - 1
// CHECK-NEXT:      val diff = new_successors.length - (to - from)
// CHECK-NEXT:      for (successor, i) <- (new_successors ++ successors.slice(to, successors.length)).zipWithIndex do
// CHECK-NEXT:        successors(from + i) = successor
// CHECK-NEXT:      if (diff < 0)
// CHECK-NEXT:        successors.trimEnd(-diff)
// CHECK-NEXT:    }

// CHECK:         def successor2: Block = successors(successors.length - 1)
// CHECK-NEXT:    def successor2_=(new_successor: Block): Unit = {successors(successors.length - 1) = new_successor}

// CHECK:         val sing_op1_constr = BaseAttr[scair.dialects.builtin.IntData]()
// CHECK-NEXT:    val var_op1_constr = IntData(5)
// CHECK-NEXT:    val var_op2_constr = IntData(5)
// CHECK-NEXT:    val sing_op2_constr = IntData(5) || IntData(6)
// CHECK-NEXT:    val sing_res1_constr = AnyAttr
// CHECK-NEXT:    val var_res1_constr = AnyAttr
// CHECK-NEXT:    val var_res2_constr = AnyAttr
// CHECK-NEXT:    val sing_res2_constr = AnyAttr

// CHECK:         override def custom_verify(): Unit =
// CHECK-NEXT:      val verification_context = new ConstraintContext()

// CHECK:           val operandSegmentSizesSum = operandSegmentSizes.fold(0)(_ + _)
// CHECK-NEXT:      if (operandSegmentSizesSum != operands.length) then throw new Exception(s"Expected ${operandSegmentSizesSum} operands, got ${operands.length}")
// CHECK-NEXT:      if operandSegmentSizes(0) != 1 then throw new Exception(s"operand segment size expected to be 1 for singular operand sing_op1 at index 0, got ${operandSegmentSizes(0)}")
// CHECK-NEXT:      if operandSegmentSizes(3) != 1 then throw new Exception(s"operand segment size expected to be 1 for singular operand sing_op2 at index 3, got ${operandSegmentSizes(3)}")
// CHECK-NEXT:      val resultSegmentSizesSum = resultSegmentSizes.fold(0)(_ + _)
// CHECK-NEXT:      if (resultSegmentSizesSum != results.length) then throw new Exception(s"Expected ${resultSegmentSizesSum} results, got ${results.length}")
// CHECK-NEXT:      if resultSegmentSizes(0) != 1 then throw new Exception(s"result segment size expected to be 1 for singular result sing_res1 at index 0, got ${resultSegmentSizes(0)}")
// CHECK-NEXT:      if resultSegmentSizes(3) != 1 then throw new Exception(s"result segment size expected to be 1 for singular result sing_res2 at index 3, got ${resultSegmentSizes(3)}")
// CHECK-NEXT:      if (regions.length < 2) then throw new Exception(s"Expected at least 2 regions, got ${regions.length}")
// CHECK-NEXT:      if (successors.length < 2) then throw new Exception(s"Expected at least 2 successors, got ${successors.length}")

// CHECK:           sing_op1_constr.verify(sing_op1.typ, verification_context)
// CHECK-NEXT:      var_op1_constr.verify(var_op1.typ, verification_context)
// CHECK-NEXT:      var_op2_constr.verify(var_op2.typ, verification_context)
// CHECK-NEXT:      sing_op2_constr.verify(sing_op2.typ, verification_context)
// CHECK-NEXT:      sing_res1_constr.verify(sing_res1.typ, verification_context)
// CHECK-NEXT:      var_res1_constr.verify(var_res1.typ, verification_context)
// CHECK-NEXT:      var_res2_constr.verify(var_res2.typ, verification_context)
// CHECK-NEXT:      sing_res2_constr.verify(sing_res2.typ, verification_context)

// CHECK:       }

// CHECK:       object TypeAttr extends AttributeObject {
// CHECK-NEXT:    override def name = "test.type"
// CHECK-NEXT:    override def factory = TypeAttr.apply
// CHECK-NEXT:  }

// CHECK:       case class TypeAttr(override val parameters: Seq[Attribute]) extends ParametrizedAttribute(name = "test.type", parameters = parameters) with TypeAttribute {
// CHECK-NEXT:    override def custom_verify(): Unit =
// CHECK-NEXT:      if (parameters.length != 0) then throw new Exception(s"Expected 0 parameters, got ${parameters.length}")
// CHECK-NEXT:  }

// CHECK:       val testDialect: Dialect = new Dialect(
// CHECK-NEXT:    operations = Seq(NoVariadicsOp, VariadicOperandOp, MultiVariadicOperandOp),
// CHECK-NEXT:    attributes = Seq(TypeAttr)
// CHECK-NEXT:  )
