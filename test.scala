
import scair.ir._
import scair.dialects.builtin._
import scair.scairdl.constraints._
  
object NoVariadicsOp extends DialectOperation {
  override def name = "test.no_variadics"
  override def factory = NoVariadicsOp.apply
}

case class NoVariadicsOp(
    override val operands: ListType[Value[Attribute]] = ListType(),
    override val successors: ListType[Block] = ListType(),
    override val results: ListType[Value[Attribute]] = ListType(),
    override val regions: ListType[Region] = ListType(),
    override val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends RegisteredOperation(name = "test.no_variadics") {


  
  
  
  def sing_op1: Value[Attribute] = operands(0)
  def sing_op1_=(value: Value[Attribute]): Unit = {operands(0) = value}

  def sing_op2: Value[Attribute] = operands(1)
  def sing_op2_=(value: Value[Attribute]): Unit = {operands(1) = value}

  def sing_res1: Value[Attribute] = results(0)
  def sing_res1_=(value: Value[Attribute]): Unit = {results(0) = value}

  def sing_res2: Value[Attribute] = results(1)
  def sing_res2_=(value: Value[Attribute]): Unit = {results(1) = value}

  def region1: Region = regions(0)
  def region1_=(value: Region): Unit = {regions(0) = value}

  def successor1: Block = successors(0)
  def successor1_=(value: Block): Unit = {successors(0) = value}

  val sing_op1_constr = BaseAttr[scair.dialects.builtin.IntData]()
  val sing_op2_constr = AnyOf(List(EqualAttr(IntData(5)), EqualAttr(IntData(6))))
  val sing_res1_constr = AnyAttr
  val sing_res2_constr = AnyAttr

  override def custom_verify(): Unit = 
    val verification_context = new ConstraintContext()
    
    if (operands.length != 2) then throw new Exception(s"Expected 2 operands, got ${operands.length}")
    if (results.length != 2) then throw new Exception(s"Expected 2 results, got ${results.length}")
    if (regions.length != 1) then throw new Exception("Expected 1 regions, got regions.length")
    if (successors.length != 1) then throw new Exception("Expected 1 successors, got successors.length")
    if (dictionaryProperties.size != 0) then throw new Exception("Expected 0 properties, got dictionaryProperties.size")
    if (dictionaryAttributes.size != 0) then throw new Exception("Expected 0 attributes, got dictionaryAttributes.size")

    sing_op1_constr.verify(sing_op1.typ, verification_context)
    sing_op2_constr.verify(sing_op2.typ, verification_context)
    sing_res1_constr.verify(sing_res1.typ, verification_context)
    sing_res2_constr.verify(sing_res2.typ, verification_context)


}


object VariadicOperandOp extends DialectOperation {
  override def name = "test.variadic_operand"
  override def factory = VariadicOperandOp.apply
}

case class VariadicOperandOp(
    override val operands: ListType[Value[Attribute]] = ListType(),
    override val successors: ListType[Block] = ListType(),
    override val results: ListType[Value[Attribute]] = ListType(),
    override val regions: ListType[Region] = ListType(),
    override val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends RegisteredOperation(name = "test.variadic_operand") {


  
  
  
  def sing_op1: Value[Attribute] = operands(0)
  def sing_op1_=(value: Value[Attribute]): Unit = {operands(0) = value}

  def var_op1: Seq[Value[Attribute]] = {
      val from = 1
      val to = operands.length - 1
      operands.slice(from, to).toSeq
  }
  def var_op1_=(values: Seq[Value[Attribute]]): Unit = {
    val from = 1
    val to = operands.length - 1
    val diff = values.length - (to - from)
    for (value, i) <- (values ++ operands.slice(to, operands.length)).zipWithIndex do
      operands(from + i) = value
    if (diff < 0)
      operands.trimEnd(-diff)
  }


  def sing_op2: Value[Attribute] = operands(operands.length - 1)
  def sing_op2_=(value: Value[Attribute]): Unit = {operands(operands.length - 1) = value}

  def sing_res1: Value[Attribute] = results(0)
  def sing_res1_=(value: Value[Attribute]): Unit = {results(0) = value}

  def var_res1: Seq[Value[Attribute]] = {
      val from = 1
      val to = results.length - 1
      results.slice(from, to).toSeq
  }
  def var_res1_=(values: Seq[Value[Attribute]]): Unit = {
    val from = 1
    val to = results.length - 1
    val diff = values.length - (to - from)
    for (value, i) <- (values ++ results.slice(to, results.length)).zipWithIndex do
      results(from + i) = value
    if (diff < 0)
      results.trimEnd(-diff)
  }


  def sing_res2: Value[Attribute] = results(results.length - 1)
  def sing_res2_=(value: Value[Attribute]): Unit = {results(results.length - 1) = value}

  def region1: Region = regions(0)
  def region1_=(value: Region): Unit = {regions(0) = value}

  def successor1: Block = successors(0)
  def successor1_=(value: Block): Unit = {successors(0) = value}

  val sing_op1_constr = BaseAttr[scair.dialects.builtin.IntData]()
  val var_op1_constr = EqualAttr(IntData(5))
  val sing_op2_constr = AnyOf(List(EqualAttr(IntData(5)), EqualAttr(IntData(6))))
  val sing_res1_constr = AnyAttr
  val var_res1_constr = AnyAttr
  val sing_res2_constr = AnyAttr

  override def custom_verify(): Unit = 
    val verification_context = new ConstraintContext()
    
    if (operands.length < 2) then throw new Exception(s"Expected at least 2 operands, got ${operands.length}")
    if (results.length < 2) then throw new Exception(s"Expected at least 2 results, got ${results.length}")
    if (regions.length != 1) then throw new Exception("Expected 1 regions, got regions.length")
    if (successors.length != 1) then throw new Exception("Expected 1 successors, got successors.length")
    if (dictionaryProperties.size != 0) then throw new Exception("Expected 0 properties, got dictionaryProperties.size")
    if (dictionaryAttributes.size != 0) then throw new Exception("Expected 0 attributes, got dictionaryAttributes.size")

    sing_op1_constr.verify(sing_op1.typ, verification_context)
    var_op1_constr.verify(var_op1.typ, verification_context)
    sing_op2_constr.verify(sing_op2.typ, verification_context)
    sing_res1_constr.verify(sing_res1.typ, verification_context)
    var_res1_constr.verify(var_res1.typ, verification_context)
    sing_res2_constr.verify(sing_res2.typ, verification_context)


}


object MultiVariadicOperandOp extends DialectOperation {
  override def name = "test.multi_variadic_operand"
  override def factory = MultiVariadicOperandOp.apply
}

case class MultiVariadicOperandOp(
    override val operands: ListType[Value[Attribute]] = ListType(),
    override val successors: ListType[Block] = ListType(),
    override val results: ListType[Value[Attribute]] = ListType(),
    override val regions: ListType[Region] = ListType(),
    override val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends RegisteredOperation(name = "test.multi_variadic_operand") {


  def operandSegmentSizes: Seq[Int] =
    if (!dictionaryProperties.contains("operandSegmentSizes")) then throw new Exception("Expected operandSegmentSizes property")
    val operandSegmentSizes_attr = dictionaryProperties("operandSegmentSizes") match {
      case right: DenseArrayAttr => right
      case _ => throw new Exception("Expected operandSegmentSizes to be a DenseArrayAttr")
    }
    ParametrizedAttrConstraint[scair.dialects.builtin.DenseArrayAttr](List(EqualAttr(IntegerType(IntData(32),Signless)), AllOf(List(BaseAttr[scair.dialects.builtin.IntegerAttr](), ParametrizedAttrConstraint[scair.dialects.builtin.IntegerAttr](List(BaseAttr[scair.dialects.builtin.IntData](), EqualAttr(IntegerType(IntData(32),Signless)))))))).verify(operandSegmentSizes_attr, ConstraintContext())
    if (operandSegmentSizes_attr.length != 4) then throw new Exception(s"Expected operandSegmentSizes to have 4 elements, got ${operandSegmentSizes_attr.length}")
    
    for (s <- operandSegmentSizes_attr) yield s match {
      case right: IntegerAttr => right.value.data.toInt
      case _ => throw new Exception("Unreachable exception as per above constraint check.")
    }
    
  def resultSegmentSizes: Seq[Int] =
    if (!dictionaryProperties.contains("resultSegmentSizes")) then throw new Exception("Expected resultSegmentSizes property")
    val resultSegmentSizes_attr = dictionaryProperties("resultSegmentSizes") match {
      case right: DenseArrayAttr => right
      case _ => throw new Exception("Expected resultSegmentSizes to be a DenseArrayAttr")
    }
    ParametrizedAttrConstraint[scair.dialects.builtin.DenseArrayAttr](List(EqualAttr(IntegerType(IntData(32),Signless)), AllOf(List(BaseAttr[scair.dialects.builtin.IntegerAttr](), ParametrizedAttrConstraint[scair.dialects.builtin.IntegerAttr](List(BaseAttr[scair.dialects.builtin.IntData](), EqualAttr(IntegerType(IntData(32),Signless)))))))).verify(resultSegmentSizes_attr, ConstraintContext())
    if (resultSegmentSizes_attr.length != 4) then throw new Exception(s"Expected resultSegmentSizes to have 4 elements, got ${resultSegmentSizes_attr.length}")
    
    for (s <- resultSegmentSizes_attr) yield s match {
      case right: IntegerAttr => right.value.data.toInt
      case _ => throw new Exception("Unreachable exception as per above constraint check.")
    }
    
  
  def sing_op1: Value[Attribute] = operands(operandSegmentSizes.slice(0, 0).reduce(_ + _))
  def sing_op1_=(value: Value[Attribute]): Unit = {operands(operandSegmentSizes.slice(0, 0).reduce(_ + _)) = value}

  def var_op1: Seq[Value[Attribute]] = {
      val from = operandSegmentSizes.slice(0, 1).reduce(_ + _)
      val to = from + operandSegmentSizes(1)
      operands.slice(from, to).toSeq
  }
  def var_op1_=(values: Seq[Value[Attribute]]): Unit = {
    val from = operandSegmentSizes.slice(0, 1).reduce(_ + _)
    val to = from + operandSegmentSizes(1)
    val diff = values.length - (to - from)
    for (value, i) <- (values ++ operands.slice(to, operands.length)).zipWithIndex do
      operands(from + i) = value
    if (diff < 0)
      operands.trimEnd(-diff)
  }


  def var_op2: Seq[Value[Attribute]] = {
      val from = operandSegmentSizes.slice(0, 2).reduce(_ + _)
      val to = from + operandSegmentSizes(2)
      operands.slice(from, to).toSeq
  }
  def var_op2_=(values: Seq[Value[Attribute]]): Unit = {
    val from = operandSegmentSizes.slice(0, 2).reduce(_ + _)
    val to = from + operandSegmentSizes(2)
    val diff = values.length - (to - from)
    for (value, i) <- (values ++ operands.slice(to, operands.length)).zipWithIndex do
      operands(from + i) = value
    if (diff < 0)
      operands.trimEnd(-diff)
  }


  def sing_op2: Value[Attribute] = operands(operandSegmentSizes.slice(0, 3).reduce(_ + _))
  def sing_op2_=(value: Value[Attribute]): Unit = {operands(operandSegmentSizes.slice(0, 3).reduce(_ + _)) = value}

  def sing_res1: Value[Attribute] = results(resultSegmentSizes.slice(0, 0).reduce(_ + _))
  def sing_res1_=(value: Value[Attribute]): Unit = {results(resultSegmentSizes.slice(0, 0).reduce(_ + _)) = value}

  def var_res1: Seq[Value[Attribute]] = {
      val from = resultSegmentSizes.slice(0, 1).reduce(_ + _)
      val to = from + resultSegmentSizes(1)
      results.slice(from, to).toSeq
  }
  def var_res1_=(values: Seq[Value[Attribute]]): Unit = {
    val from = resultSegmentSizes.slice(0, 1).reduce(_ + _)
    val to = from + resultSegmentSizes(1)
    val diff = values.length - (to - from)
    for (value, i) <- (values ++ results.slice(to, results.length)).zipWithIndex do
      results(from + i) = value
    if (diff < 0)
      results.trimEnd(-diff)
  }


  def var_res2: Seq[Value[Attribute]] = {
      val from = resultSegmentSizes.slice(0, 2).reduce(_ + _)
      val to = from + resultSegmentSizes(2)
      results.slice(from, to).toSeq
  }
  def var_res2_=(values: Seq[Value[Attribute]]): Unit = {
    val from = resultSegmentSizes.slice(0, 2).reduce(_ + _)
    val to = from + resultSegmentSizes(2)
    val diff = values.length - (to - from)
    for (value, i) <- (values ++ results.slice(to, results.length)).zipWithIndex do
      results(from + i) = value
    if (diff < 0)
      results.trimEnd(-diff)
  }


  def sing_res2: Value[Attribute] = results(resultSegmentSizes.slice(0, 3).reduce(_ + _))
  def sing_res2_=(value: Value[Attribute]): Unit = {results(resultSegmentSizes.slice(0, 3).reduce(_ + _)) = value}

  def region1: Region = regions(0)
  def region1_=(value: Region): Unit = {regions(0) = value}

  def successor1: Block = successors(0)
  def successor1_=(value: Block): Unit = {successors(0) = value}

  val sing_op1_constr = BaseAttr[scair.dialects.builtin.IntData]()
  val var_op1_constr = EqualAttr(IntData(5))
  val var_op2_constr = EqualAttr(IntData(5))
  val sing_op2_constr = AnyOf(List(EqualAttr(IntData(5)), EqualAttr(IntData(6))))
  val sing_res1_constr = AnyAttr
  val var_res1_constr = AnyAttr
  val var_res2_constr = AnyAttr
  val sing_res2_constr = AnyAttr

  override def custom_verify(): Unit = 
    val verification_context = new ConstraintContext()
    
    val operandSegmentSizesSum = operandSegmentSizes.reduce(_ + _)
    if (operandSegmentSizesSum != ${operands.length}) then throw new Exception(s"Expected ${operandSegmentSizesSum} operands, got ${operands.length}")
    if operandSegmentSizes(0) != 1 then throw new Exception("operand segment size expected to be 1 for singular operand sing_op1 at index 0, got ${operandSegmentSizes(0)}")
    if operandSegmentSizes(3) != 1 then throw new Exception("operand segment size expected to be 1 for singular operand sing_op2 at index 3, got ${operandSegmentSizes(3)}")
    val resultSegmentSizesSum = resultSegmentSizes.reduce(_ + _)
    if (resultSegmentSizesSum != ${results.length}) then throw new Exception(s"Expected ${resultSegmentSizesSum} results, got ${results.length}")
    if resultSegmentSizes(0) != 1 then throw new Exception("result segment size expected to be 1 for singular result sing_res1 at index 0, got ${resultSegmentSizes(0)}")
    if resultSegmentSizes(3) != 1 then throw new Exception("result segment size expected to be 1 for singular result sing_res2 at index 3, got ${resultSegmentSizes(3)}")
    if (regions.length != 1) then throw new Exception("Expected 1 regions, got regions.length")
    if (successors.length != 1) then throw new Exception("Expected 1 successors, got successors.length")
    if (dictionaryProperties.size != 0) then throw new Exception("Expected 0 properties, got dictionaryProperties.size")
    if (dictionaryAttributes.size != 0) then throw new Exception("Expected 0 attributes, got dictionaryAttributes.size")

    sing_op1_constr.verify(sing_op1.typ, verification_context)
    var_op1_constr.verify(var_op1.typ, verification_context)
    var_op2_constr.verify(var_op2.typ, verification_context)
    sing_op2_constr.verify(sing_op2.typ, verification_context)
    sing_res1_constr.verify(sing_res1.typ, verification_context)
    var_res1_constr.verify(var_res1.typ, verification_context)
    var_res2_constr.verify(var_res2.typ, verification_context)
    sing_res2_constr.verify(sing_res2.typ, verification_context)


}


object TypeAttr extends DialectAttribute {
  override def name = "test.type"
  override def factory = TypeAttr.apply
}

case class TypeAttr(override val parameters: Seq[Attribute]) extends ParametrizedAttribute(name = "test.type", parameters = parameters) with TypeAttribute {
  override def custom_verify(): Unit = 
    if (parameters.length != 0) then throw new Exception("Expected 0 parameters, got parameters.length")
}
  
val test: Dialect = new Dialect(
  operations = Seq(NoVariadicsOp, VariadicOperandOp, MultiVariadicOperandOp),
  attributes = Seq(TypeAttr)
)
  
