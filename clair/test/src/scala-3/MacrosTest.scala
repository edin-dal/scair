import scair.ir.*
import scair.dialects.builtin.*
import scair.clair.macros.*
import scala.quoted.*
import scair.Printer
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
) extends DerivedOperation["cmath.mul", Mul]
    derives DerivedOperationCompanion

case class MulSingleVariadic(
    lhs: Operand[IntegerType],
    rhs: Seq[Operand[IntegerType]],
    result: Seq[Result[IntegerType]],
    randProp: StringData
) extends DerivedOperation["cmath.mulsinglevariadic", MulSingleVariadic]
    derives DerivedOperationCompanion

case class MulMultiVariadic(
    lhs: Operand[IntegerType],
    rhs: Seq[Operand[IntegerType]],
    mhs: Seq[Operand[IntegerType]],
    result: Seq[Result[IntegerType]],
    result2: Result[IntegerType],
    result3: Seq[Result[IntegerType]],
    operandSegmentSizes: DenseArrayAttr,
    resultSegmentSizes: DenseArrayAttr
) extends DerivedOperation["cmath.mulmultivariadic", MulMultiVariadic]
    derives DerivedOperationCompanion

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
    derives DerivedOperationCompanion

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
) extends DerivedOperation["cmath.multpropop", MultiOptionalPropertyOp]
    derives DerivedOperationCompanion

case class MultiOptionalCompositionOp(
    operand: Option[Operand[IntegerType]],
    prop1: Option[IntegerType],
    prop2: IntegerType,
    result: Option[Result[IntegerType]]
) extends DerivedOperation["cmath.multpropcompop", MultiOptionalCompositionOp]
    derives DerivedOperationCompanion

val mulComp = summon[DerivedOperationCompanion[Mul]]
val mulSVComp = summon[DerivedOperationCompanion[MulSingleVariadic]]
val mulMMVComp = summon[DerivedOperationCompanion[MulMultiVariadic]]
val mulOptComp = summon[DerivedOperationCompanion[MulOptional]]
val mulMultiOptComp = summon[DerivedOperationCompanion[MulMultiOptional]]

val multiOptPropOpComp =
  summon[DerivedOperationCompanion[MultiOptionalPropertyOp]]

val multiOptCompOp =
  summon[DerivedOperationCompanion[MultiOptionalCompositionOp]]

class MacrosTest extends AnyFlatSpec with BeforeAndAfter {

  object TestCases {

    def unverOp = mulComp.UnverifiedOp(
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
      reg1 = Region(Seq()),
      reg2 = Region(Seq()),
      succ1 = scair.ir.Block(),
      succ2 = scair.ir.Block()
    )

    def unverMulSinVarOp = new mulSVComp.UnverifiedOp(
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

    def unverMulMulVarOp = mulMMVComp.UnverifiedOp(
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

    def unverMulOptional = mulOptComp.UnverifiedOp(
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

    def unverMulMultiOptional = mulMultiOptComp.UnverifiedOp(
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

    def unverMulMultiMissingOptional = mulMultiOptComp.UnverifiedOp(
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

    def unverMultiOptionalPropOp = multiOptPropOpComp.UnverifiedOp(
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

    def unverMultiCompOptional = multiOptCompOp.UnverifiedOp(
      operands = Seq(
        Value[IntegerType](typ = IntegerType(IntData(5), Unsigned))
      ),
      results = Seq(IntegerType(IntData(25), Unsigned)).map(Result(_)),
      properties = Map(
        ("prop1" -> IntegerType(IntData(5), Unsigned)),
        ("prop2" -> IntegerType(IntData(5), Unsigned))
      )
    )

  }

  "Unverified instantiation" should "Correctly instantiates the UniverifiedOp" in {
    val opT = DerivedOperationCompanion.derived[Mul]

    val unverMulOp = opT(
      operands = Seq(
        Value[Attribute](typ = IntegerType(IntData(5), Unsigned)),
        Value[IntegerType](typ = IntegerType(IntData(5), Unsigned))
      ),
      results = Seq(IntegerType(IntData(25), Unsigned)).map(Result(_)),
      properties = Map(("randProp" -> StringData("what")))
    )

    unverMulOp.name should be("cmath.mul")
    unverMulOp.operands should matchPattern {
      case Seq(
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned))
          ) =>
    }
    unverMulOp.results should matchPattern {
      case Seq(Result(IntegerType(IntData(25), Unsigned))) =>
    }
    unverMulOp.properties("randProp") should matchPattern {
      case StringData("what") =>
    }
  }

  "Conversion to Unverified" should "Correctly translate from ADT operation to Univerified Operation" in {
    val opT = summon[DerivedOperationCompanion[Mul]]

    val op = TestCases.adtMulOp
    val unverMulOp = opT.unverify(op)

    unverMulOp.name should be("cmath.mul")
    unverMulOp.operands should matchPattern {
      case Seq(
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned))
          ) =>
    }
    unverMulOp.results should matchPattern {
      case Seq(Result(IntegerType(IntData(25), Unsigned))) =>
    }
    unverMulOp.properties("randProp") should matchPattern {
      case StringData("what") =>
    }
  }

  "Conversion to ADTOp" should "Correctly translate from Unverified operation to ADT Operation" in {
    val op = TestCases.unverOp
    val adtMulOp = mulComp.verify(op)

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

    val unverMulOp = opT.unverify(op)
    val adtMulOp = opT.verify(unverMulOp)

    adtMulOp.operand1 `eq` unverMulOp.operands(0) should be(true)
    adtMulOp.operand2 `eq` unverMulOp.operands(1) should be(true)
    adtMulOp.result1 `eq` unverMulOp.results(0) should be(true)
    adtMulOp.result2 `eq` unverMulOp.results(1) should be(true)
    adtMulOp.randProp1 `eq` unverMulOp.properties(
      "randProp1"
    ) should be(true)
    adtMulOp.randProp2 `eq` unverMulOp.properties(
      "randProp2"
    ) should be(true)
    adtMulOp.reg1 `eq` unverMulOp.regions(0) should be(true)
    adtMulOp.reg2 `eq` unverMulOp.regions(1) should be(true)
    adtMulOp.succ1 `eq` unverMulOp.successors(0) should be(true)
    adtMulOp.succ2 `eq` unverMulOp.successors(1) should be(true)
  }

  "Single Variadic Conversion to ADTOp" should "Correctly translate from Single Variadic Unverified operation to ADT Operation" in {
    val op = TestCases.unverMulSinVarOp
    val adtMulSinVarOp = mulSVComp.verify(op)

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

  "Single Variadic Conversion to Unverified" should "Correctly translate from Single Variadic ADT operation to Univerified Operation" in {
    val opT = summon[DerivedOperationCompanion[MulSingleVariadic]]

    val op = TestCases.adtMulSinVarOp
    val unverMulSinVarOp = opT.unverify(op)

    unverMulSinVarOp.name should be("cmath.mulsinglevariadic")
    unverMulSinVarOp.operands should matchPattern {
      case Seq(
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned))
          ) =>
    }
    unverMulSinVarOp.results should matchPattern {
      case Seq(
            Result(IntegerType(IntData(25), Unsigned)),
            Result(IntegerType(IntData(25), Unsigned))
          ) =>
    }
    unverMulSinVarOp.properties("randProp") should matchPattern {
      case StringData("what") =>
    }
  }

  "Multi Variadic Conversion to ADTOp" should "Correctly translate from Multi Variadic Unverified operation to ADT Operation" in {
    val op = TestCases.unverMulMulVarOp
    val adtMulMulVarOp = mulMMVComp.verify(op)

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

  "Multi Variadic Conversion to Unverified" should "Correctly translate from Multi Variadic ADT operation to Univerified Operation" in {
    val opT = summon[DerivedOperationCompanion[MulMultiVariadic]]

    val op = TestCases.adtMulMulVarOp
    val unverMulSinVarOp = opT.unverify(op)

    unverMulSinVarOp.name should be("cmath.mulmultivariadic")
    unverMulSinVarOp.operands should matchPattern {
      case Seq(
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned))
          ) =>
    }
    unverMulSinVarOp.results should matchPattern {
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
      comp.UnverifiedOp(regions =
        Seq(
          Region(Seq(Block(comp.UnverifiedOp()), Block(comp.UnverifiedOp()))),
          Region(Seq(Block(comp.UnverifiedOp()), Block(comp.UnverifiedOp())))
        )
      )

    val verified = op.verify()
    verified should matchPattern {
      case Right(
            RegionOp(
              Seq(
                Region(
                  Seq(
                    Block(_, ListType(RegionOp(_))),
                    Block(_, ListType(RegionOp(_)))
                  )
                ),
                Region(_)
              )
            )
          ) =>
    }
  }

  "Incorrect Conversion to ADTOp" should "fail gracefully on verification" in {
    def unverOp = mulComp.UnverifiedOp()

    val verified = unverOp.verify()
    verified should matchPattern {
      case Left("java.lang.Exception: Expected 2 operands, got 0.") =>
    }
  }

  "Single Optional Conversion to Unverified" should "Correctly translate from Single Optional ADT operation to Unverified Operation" in {
    val op = TestCases.adtMulOptional
    val unverMulSinVarOp = mulOptComp.unverify(op)

    unverMulSinVarOp.name should be("cmath.mulopt")
    unverMulSinVarOp.operands should matchPattern {
      case Seq(
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned))
          ) =>
    }
    unverMulSinVarOp.results should matchPattern {
      case Seq(Result(IntegerType(IntData(25), Unsigned))) =>
    }
  }

  "Single Optional Conversion to ADTOp" should "Correctly translate from Single Optional Unverified operation to ADT Operation" in {
    val op = TestCases.unverMulOptional
    val adtMulOptional = mulOptComp.verify(op)

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

  "Multi Optional Conversion to Unverified" should "Correctly translate from Multi Optional ADT operation to Unverified Operation" in {
    val op = TestCases.adtMulMultiOptional
    val unverMulSinVarOp = mulMultiOptComp.unverify(op)

    unverMulSinVarOp.name should be("cmath.mulmultiopt")
    unverMulSinVarOp.operands should matchPattern {
      case Seq(
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned))
          ) =>
    }
    unverMulSinVarOp.results should matchPattern {
      case Seq(Result(IntegerType(IntData(25), Unsigned))) =>
    }
  }

  "Multi Optional Conversion to ADTOp" should "Correctly translate from Multi Optional Unverified operation to ADT Operation" in {
    val op = TestCases.unverMulMultiOptional
    val adtMulOptional = mulMultiOptComp.verify(op)

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

  "Multi Optional Conversion to ADTOp" should "Correctly translate from Multi Optional Unverified operation to ADT Operation with missing middle Operand" in {
    val op = TestCases.unverMulMultiMissingOptional
    val adtMulOptional = mulMultiOptComp.verify(op)

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

  "Multi Optional Property Conversion to ADTOp" should "Correctly translate from Multi Optional Properties Unverified operation to ADT Operation" in {
    val op = TestCases.unverMultiOptionalPropOp
    val mulPropOptional = multiOptPropOpComp.verify(op)

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

  "Multi Optional Property Conversion to Unverified" should "Correctly translate from Multi Optional Properties ADT operation to Unverified Operation" in {
    val op = TestCases.adtMultiOptionalPropOp
    val unverMulPropOptional = multiOptPropOpComp.unverify(op)

    unverMulPropOptional.properties("prop1") should matchPattern {
      case IntegerType(IntData(5), Unsigned) =>
    }

    unverMulPropOptional.properties("prop2") should matchPattern {
      case IntegerType(IntData(5), Unsigned) =>
    }

    unverMulPropOptional.properties("prop3") should matchPattern {
      case IntegerType(IntData(5), Unsigned) =>
    }
  }

  "Multi Optional Composition Conversion to ADTOp" should "Correctly translate from Multi Optional Properties Unverified operation to ADT Operation" in {
    val op = TestCases.unverMultiCompOptional
    val adtPropOptionalComp = multiOptCompOp.verify(op)

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

  "Multi Optional Composition Conversion to Unverified" should "Correctly translate from Multi Optional Properties ADT operation to Unverified Operation" in {
    val op = TestCases.adtMultiCompOptional
    val unverMulPropOptional = multiOptCompOp.unverify(op)

    unverMulPropOptional.operands(0) should matchPattern {
      case Value(IntegerType(IntData(5), Unsigned)) =>
    }

    unverMulPropOptional.properties("prop1") should matchPattern {
      case IntegerType(IntData(5), Unsigned) =>
    }

    unverMulPropOptional.properties("prop2") should matchPattern {
      case IntegerType(IntData(5), Unsigned) =>
    }

    unverMulPropOptional.results(0) should matchPattern {
      case Result(IntegerType(IntData(25), Unsigned)) =>
    }
  }

}
