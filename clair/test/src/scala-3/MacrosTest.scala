import scair.ir.*
import scair.dialects.builtin.*
import scair.clair.macros.*
import scala.quoted.*
import scair.Printer
import org.scalatest.*
import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*
import scala.collection.mutable.LinkedHashMap

// TODO: create a better constructor for DenseArrayAttr with just integers

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

class MacrosTest extends AnyFlatSpec with BeforeAndAfter {

  object TestCases {

    def unverOp = UnverifiedOp[Mul](
      name = "cmath.mul",
      operands = Seq(
        Value[Attribute](typ = IntegerType(IntData(5), Unsigned)),
        Value[IntegerType](typ = IntegerType(IntData(5), Unsigned))
      ),
      results_types = Seq[Attribute](IntegerType(IntData(25), Unsigned)),
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

    def unverMulSinVarOp = UnverifiedOp[MulSingleVariadic](
      name = "cmath.mulsinglevariadic",
      operands = Seq(
        Value[Attribute](typ = IntegerType(IntData(5), Unsigned)),
        Value[IntegerType](typ = IntegerType(IntData(5), Unsigned)),
        Value[IntegerType](typ = IntegerType(IntData(5), Unsigned))
      ),
      results_types = Seq(
        IntegerType(IntData(25), Unsigned),
        IntegerType(IntData(25), Unsigned)
      ),
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

    def unverMulMulVarOp = UnverifiedOp[MulMultiVariadic](
      name = "cmath.mulmultivariadic",
      operands = Seq(
        Value(typ = IntegerType(IntData(5), Unsigned)),
        Value(typ = IntegerType(IntData(5), Unsigned)),
        Value(typ = IntegerType(IntData(5), Unsigned)),
        Value(typ = IntegerType(IntData(5), Unsigned)),
        Value(typ = IntegerType(IntData(5), Unsigned)),
        Value(typ = IntegerType(IntData(5), Unsigned))
      ),
      results_types = Seq(
        IntegerType(IntData(25), Unsigned),
        IntegerType(IntData(25), Unsigned),
        IntegerType(IntData(25), Unsigned),
        IntegerType(IntData(25), Unsigned),
        IntegerType(IntData(25), Unsigned),
        IntegerType(IntData(25), Unsigned)
      ),
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

  }

  "Unverified instantiation" should "Correctly instantiates the UniverifiedOp" in {
    val opT = DerivedOperationCompanion.derived[Mul]

    val unverMulOp = opT(
      operands = Seq(
        Value[Attribute](typ = IntegerType(IntData(5), Unsigned)),
        Value[IntegerType](typ = IntegerType(IntData(5), Unsigned))
      ),
      results_types = Seq(IntegerType(IntData(25), Unsigned)),
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
    val opT = summon[DerivedOperationCompanion[Mul]]

    val op = TestCases.unverOp
    val adtMulOp = opT.verify(op)

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

    adtMulOp.operand1 `equals` unverMulOp.operands(0) should be(true)
    adtMulOp.operand2 `equals` unverMulOp.operands(1) should be(true)
    adtMulOp.result1 `equals` unverMulOp.results(0) should be(true)
    adtMulOp.result2 `equals` unverMulOp.results(1) should be(true)
    adtMulOp.randProp1 `equals` unverMulOp.properties(
      "randProp1"
    ) should be(true)
    adtMulOp.randProp2 `equals` unverMulOp.properties(
      "randProp2"
    ) should be(true)
    adtMulOp.reg1 `equals` unverMulOp.regions(0) should be(true)
    adtMulOp.reg2 `equals` unverMulOp.regions(1) should be(true)
    adtMulOp.succ1 `equals` unverMulOp.successors(0) should be(true)
    adtMulOp.succ2 `equals` unverMulOp.successors(1) should be(true)
  }

  "Single Variadic Conversion to ADTOp" should "Correctly translate from Single Variadic Unverified operation to ADT Operation" in {
    val opT = summon[DerivedOperationCompanion[MulSingleVariadic]]

    val op = TestCases.unverMulSinVarOp
    val adtMulSinVarOp = opT.verify(op)

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
    val opT = summon[DerivedOperationCompanion[MulMultiVariadic]]

    val op = TestCases.unverMulMulVarOp
    val adtMulMulVarOp = opT.verify(op)

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

}
