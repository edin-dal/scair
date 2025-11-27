import scair.ir.*
import scair.dialects.builtin.*
import scair.clair.macros.*
import org.scalatest.*
import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*
import scala.collection.mutable.LinkedHashMap

case class RegionOp(
    wowregions: Seq[Region]
) extends DerivedOperation["test.region", RegionOp]
    derives DerivedOperationCompanion

case class Mul(
    lhs: Operand[IntegerType],
    rhs: Operand[IntegerType],
    result: Result[IntegerType],
    randProp: StringData
) extends DerivedOperation["cmath.mul", Mul] derives DerivedOperationCompanion

case class MulSingleVariadic(
    lhs: Operand[IntegerType],
    rhs: Seq[Operand[IntegerType]],
    result: Seq[Result[IntegerType]],
    randProp: StringData
) extends DerivedOperation[
      "cmath.mulsinglevariadic",
      MulSingleVariadic
    ] derives DerivedOperationCompanion

case class MulMultiVariadic(
    lhs: Operand[IntegerType],
    rhs: Seq[Operand[IntegerType]],
    mhs: Seq[Operand[IntegerType]],
    result: Seq[Result[IntegerType]],
    result2: Result[IntegerType],
    result3: Seq[Result[IntegerType]],
    operandSegmentSizes: DenseArrayAttr,
    resultSegmentSizes: DenseArrayAttr
) extends DerivedOperation[
      "cmath.mulmultivariadic",
      MulMultiVariadic
    ] derives DerivedOperationCompanion

case class MulFull(
    operand1: Operand[IntegerType],
    operand2: Operand[IntegerType],
    result1: Result[IntegerType],
    result2: Result[IntegerType],
    randProp1: StringData,
    randProp2: StringData,
    reg1: Region,
    reg2: Region,
    succ1: Successor,
    succ2: Successor
) extends DerivedOperation["cmath.mulfull", MulFull]
    derives DerivedOperationCompanion

case class MulOptional(
    lhs: Option[Operand[IntegerType]],
    rhs: Operand[IntegerType],
    res: Result[IntegerType]
) extends DerivedOperation["cmath.mulopt", MulOptional]
    with AssemblyFormat[
      "($lhs^ `,`)? $rhs attr-dict `:` `(` (type($lhs)^ `,`)? type($rhs) `)` `->` type($res)"
    ] derives DerivedOperationCompanion

case class MulMultiOptional(
    lhs: Option[Operand[IntegerType]],
    rhs: Option[Operand[IntegerType]],
    additional: Option[Operand[IntegerType]],
    res: Result[IntegerType]
) extends DerivedOperation["cmath.mulmultiopt", MulMultiOptional]
    derives DerivedOperationCompanion

case class MultiOptionalPropertyOp(
    prop1: Option[IntegerType],
    prop2: Option[IntegerType],
    prop3: Option[IntegerType]
) extends DerivedOperation[
      "cmath.multpropop",
      MultiOptionalPropertyOp
    ] derives DerivedOperationCompanion

case class MultiOptionalCompositionOp(
    operand: Option[Operand[IntegerType]],
    prop1: Option[IntegerType],
    prop2: IntegerType,
    result: Option[Result[IntegerType]]
) extends DerivedOperation[
      "cmath.multpropcompop",
      MultiOptionalCompositionOp
    ] derives DerivedOperationCompanion

val mulComp = summon[DerivedOperationCompanion[Mul]]
val mulSVComp = summon[DerivedOperationCompanion[MulSingleVariadic]]
val mulMMVComp = summon[DerivedOperationCompanion[MulMultiVariadic]]
val mulOptComp = summon[DerivedOperationCompanion[MulOptional]]
val mulMultiOptComp = summon[DerivedOperationCompanion[MulMultiOptional]]

val multiOptPropOpComp =
  summon[DerivedOperationCompanion[MultiOptionalPropertyOp]]

val multiOptCompOp =
  summon[DerivedOperationCompanion[MultiOptionalCompositionOp]]

class MacrosTest extends AnyFlatSpec with BeforeAndAfter:

  object TestCases:

    def unstrucOp = mulComp.UnstructuredOp(
      operands = Seq(
        Value[Attribute](typ = IntegerType(IntData(5), Unsigned)),
        Value[IntegerType](typ = IntegerType(IntData(5), Unsigned))
      ),
      results =
        Seq[Attribute](IntegerType(IntData(25), Unsigned)).map(Result(_)),
      properties = Map(("randProp" -> StringData("what")))
    )

    def adtMulOp = Mul(
      lhs = Value(IntegerType(IntData(5), Unsigned)),
      rhs = Value(IntegerType(IntData(5), Unsigned)),
      result = Result(IntegerType(IntData(25), Unsigned)),
      randProp = StringData("what")
    )

    def adtMulOpAllFields = MulFull(
      operand1 = Value(IntegerType(IntData(5), Unsigned)),
      operand2 = Value(IntegerType(IntData(5), Unsigned)),
      result1 = Result(IntegerType(IntData(25), Unsigned)),
      result2 = Result(IntegerType(IntData(25), Unsigned)),
      randProp1 = StringData("what1"),
      randProp2 = StringData("what2"),
      reg1 = Region(),
      reg2 = Region(),
      succ1 = scair.ir.Block(),
      succ2 = scair.ir.Block()
    )

    def unstructMulSinVarOp = new mulSVComp.UnstructuredOp(
      operands = Seq(
        Value[Attribute](typ = IntegerType(IntData(5), Unsigned)),
        Value[IntegerType](typ = IntegerType(IntData(5), Unsigned)),
        Value[IntegerType](typ = IntegerType(IntData(5), Unsigned))
      ),
      results = Seq(
        IntegerType(IntData(25), Unsigned),
        IntegerType(IntData(25), Unsigned)
      ).map(Result(_)),
      properties = Map(("randProp" -> StringData("what")))
    )

    def adtMulSinVarOp = MulSingleVariadic(
      lhs = Value(IntegerType(IntData(5), Unsigned)),
      rhs = Seq(
        Value(IntegerType(IntData(5), Unsigned)),
        Value(IntegerType(IntData(5), Unsigned))
      ),
      result = Seq(
        Result(IntegerType(IntData(25), Unsigned)),
        Result(IntegerType(IntData(25), Unsigned))
      ),
      randProp = StringData("what")
    )

    def unstructMulMulVarOp = mulMMVComp.UnstructuredOp(
      operands = Seq(
        Value(typ = IntegerType(IntData(5), Unsigned)),
        Value(typ = IntegerType(IntData(5), Unsigned)),
        Value(typ = IntegerType(IntData(5), Unsigned)),
        Value(typ = IntegerType(IntData(5), Unsigned)),
        Value(typ = IntegerType(IntData(5), Unsigned)),
        Value(typ = IntegerType(IntData(5), Unsigned))
      ),
      results = Seq(
        IntegerType(IntData(25), Unsigned),
        IntegerType(IntData(25), Unsigned),
        IntegerType(IntData(25), Unsigned),
        IntegerType(IntData(25), Unsigned),
        IntegerType(IntData(25), Unsigned),
        IntegerType(IntData(25), Unsigned)
      ).map(Result(_)),
      properties = Map(
        ("operandSegmentSizes" -> DenseArrayAttr(
          IntegerType(IntData(32), Signless),
          Seq[IntegerAttr](
            IntegerAttr(IntData(1), IntegerType(IntData(32), Signless)),
            IntegerAttr(IntData(3), IntegerType(IntData(32), Signless)),
            IntegerAttr(IntData(2), IntegerType(IntData(32), Signless))
          )
        )),
        ("resultSegmentSizes" -> DenseArrayAttr(
          IntegerType(IntData(32), Signless),
          Seq[IntegerAttr](
            IntegerAttr(IntData(3), IntegerType(IntData(32), Signless)),
            IntegerAttr(IntData(1), IntegerType(IntData(32), Signless)),
            IntegerAttr(IntData(2), IntegerType(IntData(32), Signless))
          )
        ))
      )
    )

    def adtMulMulVarOp = MulMultiVariadic(
      lhs = Value(IntegerType(IntData(5), Unsigned)),
      rhs = Seq(
        Value(IntegerType(IntData(5), Unsigned)),
        Value(IntegerType(IntData(5), Unsigned)),
        Value(IntegerType(IntData(5), Unsigned))
      ),
      mhs = Seq(
        Value(IntegerType(IntData(5), Unsigned)),
        Value(IntegerType(IntData(5), Unsigned))
      ),
      result = Seq(
        Result(IntegerType(IntData(5), Unsigned)),
        Result(IntegerType(IntData(5), Unsigned)),
        Result(IntegerType(IntData(5), Unsigned))
      ),
      result2 = Result(IntegerType(IntData(5), Unsigned)),
      result3 = Seq(
        Result(IntegerType(IntData(5), Unsigned)),
        Result(IntegerType(IntData(5), Unsigned))
      ),
      operandSegmentSizes = DenseArrayAttr(
        IntegerType(IntData(32), Signless),
        Seq[IntegerAttr](
          IntegerAttr(IntData(1), IntegerType(IntData(32), Signless)),
          IntegerAttr(IntData(3), IntegerType(IntData(32), Signless)),
          IntegerAttr(IntData(2), IntegerType(IntData(32), Signless))
        )
      ),
      resultSegmentSizes = DenseArrayAttr(
        IntegerType(IntData(32), Signless),
        Seq[IntegerAttr](
          IntegerAttr(IntData(1), IntegerType(IntData(32), Signless)),
          IntegerAttr(IntData(3), IntegerType(IntData(32), Signless)),
          IntegerAttr(IntData(2), IntegerType(IntData(32), Signless))
        )
      )
    )

    def adtMulOptional = MulOptional(
      lhs = Some(Value(IntegerType(IntData(5), Unsigned))),
      rhs = Value(IntegerType(IntData(5), Unsigned)),
      res = Result(IntegerType(IntData(25), Unsigned))
    )

    def unstructMulOptional = mulOptComp.UnstructuredOp(
      operands = Seq(
        Value[IntegerType](typ = IntegerType(IntData(5), Unsigned)),
        Value[IntegerType](typ = IntegerType(IntData(5), Unsigned))
      ),
      results = Seq(IntegerType(IntData(25), Unsigned)).map(Result(_))
    )

    def adtMulMultiOptional = MulMultiOptional(
      lhs = Some(Value(IntegerType(IntData(5), Unsigned))),
      rhs = Some(Value(IntegerType(IntData(5), Unsigned))),
      additional = Some(Value(IntegerType(IntData(5), Unsigned))),
      res = Result(IntegerType(IntData(25), Unsigned))
    )

    def unstructMulMultiOptional = mulMultiOptComp.UnstructuredOp(
      operands = Seq(
        Value[IntegerType](typ = IntegerType(IntData(5), Unsigned)),
        Value[IntegerType](typ = IntegerType(IntData(5), Unsigned)),
        Value[IntegerType](typ = IntegerType(IntData(5), Unsigned))
      ),
      results = Seq(IntegerType(IntData(25), Unsigned)).map(Result(_)),
      properties = Map(
        ("operandSegmentSizes" -> DenseArrayAttr(
          IntegerType(IntData(32), Signless),
          Seq[IntegerAttr](
            IntegerAttr(IntData(1), IntegerType(IntData(32), Signless)),
            IntegerAttr(IntData(1), IntegerType(IntData(32), Signless)),
            IntegerAttr(IntData(1), IntegerType(IntData(32), Signless))
          )
        ))
      )
    )

    def unstructMulMultiMissingOptional = mulMultiOptComp.UnstructuredOp(
      operands = Seq(
        Value[IntegerType](typ = IntegerType(IntData(5), Unsigned)),
        Value[IntegerType](typ = IntegerType(IntData(5), Unsigned))
      ),
      results = Seq(IntegerType(IntData(25), Unsigned)).map(Result(_)),
      properties = Map(
        ("operandSegmentSizes" -> DenseArrayAttr(
          IntegerType(IntData(32), Signless),
          Seq[IntegerAttr](
            IntegerAttr(IntData(1), IntegerType(IntData(32), Signless)),
            IntegerAttr(IntData(0), IntegerType(IntData(32), Signless)),
            IntegerAttr(IntData(1), IntegerType(IntData(32), Signless))
          )
        ))
      )
    )

    def adtMultiOptionalPropOp = MultiOptionalPropertyOp(
      prop1 = Some(IntegerType(IntData(5), Unsigned)),
      prop2 = Some(IntegerType(IntData(5), Unsigned)),
      prop3 = Some(IntegerType(IntData(5), Unsigned))
    )

    def unstructMultiOptionalPropOp = multiOptPropOpComp.UnstructuredOp(
      properties = Map(
        ("prop1" -> IntegerType(IntData(5), Unsigned)),
        ("prop2" -> IntegerType(IntData(5), Unsigned)),
        ("prop3" -> IntegerType(IntData(5), Unsigned))
      )
    )

    def adtMultiCompOptional = MultiOptionalCompositionOp(
      operand = Some(Value(IntegerType(IntData(5), Unsigned))),
      prop1 = Some(IntegerType(IntData(5), Unsigned)),
      prop2 = IntegerType(IntData(5), Unsigned),
      result = Some(Result(IntegerType(IntData(25), Unsigned)))
    )

    def unstructMultiCompOptional = multiOptCompOp.UnstructuredOp(
      operands = Seq(
        Value[IntegerType](typ = IntegerType(IntData(5), Unsigned))
      ),
      results = Seq(IntegerType(IntData(25), Unsigned)).map(Result(_)),
      properties = Map(
        ("prop1" -> IntegerType(IntData(5), Unsigned)),
        ("prop2" -> IntegerType(IntData(5), Unsigned))
      )
    )

  "Unstructured instantiation" should "Correctly instantiates the UnstructuredOp" in {
    val opT = DerivedOperationCompanion.derived[Mul]

    val unstructMulOp = opT(
      operands = Seq(
        Value[Attribute](typ = IntegerType(IntData(5), Unsigned)),
        Value[IntegerType](typ = IntegerType(IntData(5), Unsigned))
      ),
      results = Seq(IntegerType(IntData(25), Unsigned)).map(Result(_)),
      properties = Map(("randProp" -> StringData("what")))
    )

    unstructMulOp.name should be("cmath.mul")
    unstructMulOp.operands should matchPattern {
      case Seq(
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned))
          ) =>
    }
    unstructMulOp.results should matchPattern {
      case Seq(Result(IntegerType(IntData(25), Unsigned))) =>
    }
    unstructMulOp.properties("randProp") should matchPattern {
      case StringData("what") =>
    }
  }

  "Conversion to Unstructured" should "Correctly translate from ADT operation to Unstructured Operation" in {
    val opT = summon[DerivedOperationCompanion[Mul]]

    val op = TestCases.adtMulOp
    val unstructMulOp = opT.destructure(op)

    unstructMulOp.name should be("cmath.mul")
    unstructMulOp.operands should matchPattern {
      case Seq(
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned))
          ) =>
    }
    unstructMulOp.results should matchPattern {
      case Seq(Result(IntegerType(IntData(25), Unsigned))) =>
    }
    unstructMulOp.properties("randProp") should matchPattern {
      case StringData("what") =>
    }
  }

  "Conversion to ADTOp" should "Correctly translate from Unstructured operation to ADT Operation" in {
    val op = TestCases.unstrucOp
    val adtMulOp = mulComp.structure(op)

    adtMulOp.lhs should matchPattern {
      case Value(IntegerType(IntData(5), Unsigned)) =>
    }
    adtMulOp.rhs should matchPattern {
      case Value(IntegerType(IntData(5), Unsigned)) =>
    }
    adtMulOp.result should matchPattern {
      case Value(IntegerType(IntData(25), Unsigned)) =>
    }
    adtMulOp.randProp should matchPattern { case StringData("what") =>
    }
  }

  "ADTOp test for correctly passing around the same instances" should "test all fields of a ADTOp" in {
    val opT = DerivedOperationCompanion.derived[MulFull]

    val op = TestCases.adtMulOpAllFields

    val unstructMulOp = opT.destructure(op)
    val adtMulOp = opT.structure(unstructMulOp)

    adtMulOp.operand1 `eq` unstructMulOp.operands(0) should be(true)
    adtMulOp.operand2 `eq` unstructMulOp.operands(1) should be(true)
    adtMulOp.result1 `eq` unstructMulOp.results(0) should be(true)
    adtMulOp.result2 `eq` unstructMulOp.results(1) should be(true)
    adtMulOp.randProp1 `eq` unstructMulOp.properties(
      "randProp1"
    ) should be(true)
    adtMulOp.randProp2 `eq` unstructMulOp.properties(
      "randProp2"
    ) should be(true)
    adtMulOp.reg1 `eq` unstructMulOp.regions(0) should be(true)
    adtMulOp.reg2 `eq` unstructMulOp.regions(1) should be(true)
    adtMulOp.succ1 `eq` unstructMulOp.successors(0) should be(true)
    adtMulOp.succ2 `eq` unstructMulOp.successors(1) should be(true)
  }

  "Single Variadic Conversion to ADTOp" should "Correctly translate from Single Variadic Unstructured operation to ADT Operation" in {
    val op = TestCases.unstructMulSinVarOp
    val adtMulSinVarOp = mulSVComp.structure(op)

    adtMulSinVarOp.lhs should matchPattern {
      case Value(IntegerType(IntData(5), Unsigned)) =>
    }
    adtMulSinVarOp.rhs should matchPattern {
      case Seq(
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned))
          ) =>
    }
    adtMulSinVarOp.result should matchPattern {
      case Seq(
            Result(IntegerType(IntData(25), Unsigned)),
            Result(IntegerType(IntData(25), Unsigned))
          ) =>
    }
    adtMulSinVarOp.randProp should matchPattern { case StringData("what") =>
    }
  }

  "Single Variadic Conversion to Unstructured" should "Correctly translate from Single Variadic ADT operation to Unstructured Operation" in {
    val opT = summon[DerivedOperationCompanion[MulSingleVariadic]]

    val op = TestCases.adtMulSinVarOp
    val unstructMulSinVarOp = opT.destructure(op)

    unstructMulSinVarOp.name should be("cmath.mulsinglevariadic")
    unstructMulSinVarOp.operands should matchPattern {
      case Seq(
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned))
          ) =>
    }
    unstructMulSinVarOp.results should matchPattern {
      case Seq(
            Result(IntegerType(IntData(25), Unsigned)),
            Result(IntegerType(IntData(25), Unsigned))
          ) =>
    }
    unstructMulSinVarOp.properties("randProp") should matchPattern {
      case StringData("what") =>
    }
  }

  "Multi Variadic Conversion to ADTOp" should "Correctly translate from Multi Variadic Unstructured operation to ADT Operation" in {
    val op = TestCases.unstructMulMulVarOp
    val adtMulMulVarOp = mulMMVComp.structure(op)

    adtMulMulVarOp.lhs should matchPattern {
      case Value(IntegerType(IntData(5), Unsigned)) =>
    }
    adtMulMulVarOp.rhs should matchPattern {
      case Seq(
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned))
          ) =>
    }
    adtMulMulVarOp.mhs should matchPattern {
      case Seq(
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned))
          ) =>
    }
    adtMulMulVarOp.result should matchPattern {
      case Seq(
            Result(IntegerType(IntData(25), Unsigned)),
            Result(IntegerType(IntData(25), Unsigned)),
            Result(IntegerType(IntData(25), Unsigned))
          ) =>
    }
    adtMulMulVarOp.result2 should matchPattern {
      case Result(IntegerType(IntData(25), Unsigned)) =>
    }
    adtMulMulVarOp.result3 should matchPattern {
      case Seq(
            Result(IntegerType(IntData(25), Unsigned)),
            Result(IntegerType(IntData(25), Unsigned))
          ) =>
    }
  }

  "Multi Variadic Conversion to Unstructured" should "Correctly translate from Multi Variadic ADT operation to Unstructured Operation" in {
    val opT = summon[DerivedOperationCompanion[MulMultiVariadic]]

    val op = TestCases.adtMulMulVarOp
    val unstructMulSinVarOp = opT.destructure(op)

    unstructMulSinVarOp.name should be("cmath.mulmultivariadic")
    unstructMulSinVarOp.operands should matchPattern {
      case Seq(
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned))
          ) =>
    }
    unstructMulSinVarOp.results should matchPattern {
      case Seq(
            Result(IntegerType(IntData(5), Unsigned)),
            Result(IntegerType(IntData(5), Unsigned)),
            Result(IntegerType(IntData(5), Unsigned)),
            Result(IntegerType(IntData(5), Unsigned)),
            Result(IntegerType(IntData(5), Unsigned)),
            Result(IntegerType(IntData(5), Unsigned))
          ) =>
    }
  }

  "Recursive conversion to ADTOp" should "do the thing \\o/" in {
    val comp = summon[DerivedOperationCompanion[RegionOp]]
    val op =
      comp.UnstructuredOp(regions =
        Seq(
          Region(
            Block(comp.UnstructuredOp()),
            Block(comp.UnstructuredOp())
          ),
          Region(
            Block(comp.UnstructuredOp()),
            Block(comp.UnstructuredOp())
          )
        )
      )

    val structured = op.structured
    structured should matchPattern {
      case Right(
            RegionOp(
              Seq(
                Region(
                  Block(_, BlockOperations(RegionOp(_))),
                  Block(_, BlockOperations(RegionOp(_)))
                ),
                Region(_*)
              )
            )
          ) =>
    }
  }

  "Incorrect Conversion to ADTOp" should "fail gracefully on structuring" in {
    def unstrucOp = mulComp.UnstructuredOp()

    val structured = unstrucOp.structured
    structured should matchPattern {
      case Left("java.lang.Exception: Expected 2 operands, got 0.") =>
    }
  }

  "Single Optional Conversion to Unstructured" should "Correctly translate from Single Optional ADT operation to Unstructured Operation" in {
    val op = TestCases.adtMulOptional
    val unstructMulSinVarOp = mulOptComp.destructure(op)

    unstructMulSinVarOp.name should be("cmath.mulopt")
    unstructMulSinVarOp.operands should matchPattern {
      case Seq(
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned))
          ) =>
    }
    unstructMulSinVarOp.results should matchPattern {
      case Seq(Result(IntegerType(IntData(25), Unsigned))) =>
    }
  }

  "Single Optional custom syntax printing" should "Correctly print" in {
    val op = TestCases.adtMulOptional
    val op2 = MulOptional(
      lhs = None,
      rhs = Value(IntegerType(IntData(5), Unsigned)),
      res = Result(IntegerType(IntData(25), Unsigned))
    )

    val out = java.io.StringWriter()
    val printer = new scair.Printer(p = java.io.PrintWriter(out))
    printer.print(op, op2)(using 0)

    out.toString() should be(
      "%0 = cmath.mulopt %1, %2 : (ui5, ui5) -> ui25\n%3 = cmath.mulopt %4 : ( ui5) -> ui25\n"
    )
  }

  "Single Optional Conversion to ADTOp" should "Correctly translate from Single Optional Unstructured operation to ADT Operation" in {
    val op = TestCases.unstructMulOptional
    val adtMulOptional = mulOptComp.structure(op)

    adtMulOptional.lhs should matchPattern {
      case Some(Value(IntegerType(IntData(5), Unsigned))) =>
    }
    adtMulOptional.rhs should matchPattern {
      case Value(IntegerType(IntData(5), Unsigned)) =>
    }
    adtMulOptional.res should matchPattern {
      case Result(IntegerType(IntData(25), Unsigned)) =>
    }
  }

  "Multi Optional Conversion to Unstructured" should "Correctly translate from Multi Optional ADT operation to Unstructured Operation" in {
    val op = TestCases.adtMulMultiOptional
    val unstructMulSinVarOp = mulMultiOptComp.destructure(op)

    unstructMulSinVarOp.name should be("cmath.mulmultiopt")
    unstructMulSinVarOp.operands should matchPattern {
      case Seq(
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned))
          ) =>
    }
    unstructMulSinVarOp.results should matchPattern {
      case Seq(Result(IntegerType(IntData(25), Unsigned))) =>
    }
  }

  "Multi Optional Conversion to ADTOp" should "Correctly translate from Multi Optional Unstructured operation to ADT Operation" in {
    val op = TestCases.unstructMulMultiOptional
    val adtMulOptional = mulMultiOptComp.structure(op)

    adtMulOptional.lhs should matchPattern {
      case Some(Value(IntegerType(IntData(5), Unsigned))) =>
    }
    adtMulOptional.rhs should matchPattern {
      case Some(Value(IntegerType(IntData(5), Unsigned))) =>
    }
    adtMulOptional.additional should matchPattern {
      case Some(Value(IntegerType(IntData(5), Unsigned))) =>
    }
    adtMulOptional.res should matchPattern {
      case Result(IntegerType(IntData(25), Unsigned)) =>
    }
  }

  "Multi Optional Conversion to ADTOp" should "Correctly translate from Multi Optional Unstructured operation to ADT Operation with missing middle Operand" in {
    val op = TestCases.unstructMulMultiMissingOptional
    val adtMulOptional = mulMultiOptComp.structure(op)

    adtMulOptional.lhs should matchPattern {
      case Some(Value(IntegerType(IntData(5), Unsigned))) =>
    }
    adtMulOptional.rhs should matchPattern { case None =>
    }
    adtMulOptional.additional should matchPattern {
      case Some(Value(IntegerType(IntData(5), Unsigned))) =>
    }
    adtMulOptional.res should matchPattern {
      case Result(IntegerType(IntData(25), Unsigned)) =>
    }
  }

  "Multi Optional Property Conversion to ADTOp" should "Correctly translate from Multi Optional Properties Unstructured operation to ADT Operation" in {
    val op = TestCases.unstructMultiOptionalPropOp
    val mulPropOptional = multiOptPropOpComp.structure(op)

    mulPropOptional.prop1 should matchPattern {
      case Some(IntegerType(IntData(5), Unsigned)) =>
    }

    mulPropOptional.prop2 should matchPattern {
      case Some(IntegerType(IntData(5), Unsigned)) =>
    }

    mulPropOptional.prop3 should matchPattern {
      case Some(IntegerType(IntData(5), Unsigned)) =>
    }
  }

  "Multi Optional Property Conversion to Unstructured" should "Correctly translate from Multi Optional Properties ADT operation to Unstructured Operation" in {
    val op = TestCases.adtMultiOptionalPropOp
    val unstructMulPropOptional = multiOptPropOpComp.destructure(op)

    unstructMulPropOptional.properties("prop1") should matchPattern {
      case IntegerType(IntData(5), Unsigned) =>
    }

    unstructMulPropOptional.properties("prop2") should matchPattern {
      case IntegerType(IntData(5), Unsigned) =>
    }

    unstructMulPropOptional.properties("prop3") should matchPattern {
      case IntegerType(IntData(5), Unsigned) =>
    }
  }

  "Multi Optional Composition Conversion to ADTOp" should "Correctly translate from Multi Optional Properties Unstructured operation to ADT Operation" in {
    val op = TestCases.unstructMultiCompOptional
    val adtPropOptionalComp = multiOptCompOp.structure(op)

    adtPropOptionalComp.operand should matchPattern {
      case Some(Value(IntegerType(IntData(5), Unsigned))) =>
    }

    adtPropOptionalComp.prop1 should matchPattern {
      case Some(IntegerType(IntData(5), Unsigned)) =>
    }

    adtPropOptionalComp.prop2 should matchPattern {
      case IntegerType(IntData(5), Unsigned) =>
    }

    adtPropOptionalComp.result should matchPattern {
      case Some(Result(IntegerType(IntData(25), Unsigned))) =>
    }
  }

  "Multi Optional Composition Conversion to Unstructured" should "Correctly translate from Multi Optional Properties ADT operation to Unstructured Operation" in {
    val op = TestCases.adtMultiCompOptional
    val unstructMulPropOptional = multiOptCompOp.destructure(op)

    unstructMulPropOptional.operands(0) should matchPattern {
      case Value(IntegerType(IntData(5), Unsigned)) =>
    }

    unstructMulPropOptional.properties("prop1") should matchPattern {
      case IntegerType(IntData(5), Unsigned) =>
    }

    unstructMulPropOptional.properties("prop2") should matchPattern {
      case IntegerType(IntData(5), Unsigned) =>
    }

    unstructMulPropOptional.results(0) should matchPattern {
      case Result(IntegerType(IntData(25), Unsigned)) =>
    }
  }
