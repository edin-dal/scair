package scair

import scair.dialects.irdl.*
import scair.dialects.builtin.*
import scair.core.irdl_printer.IRDLPrinter.printIRDL

import fastparse.*
import org.scalatest.*
import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*
import org.scalatest.prop.TableDrivenPropertyChecks.forAll
import org.scalatest.prop.Tables.Table
import java.io.*

class IRDLPrinterTest extends AnyFlatSpec:

  val ctx = MLContext()
  ctx.registerDialect(BuiltinDialect)
  ctx.registerDialect(IRDL)
  var parser = new Parser(ctx)

  val module = parser.parseThis("""
"builtin.module"() ({
  "irdl.dialect"() <{sym_name = "cmath"}> ({
    "irdl.type"() <{sym_name = "complex"}> ({
        %2 = "irdl.any"() : () -> !irdl.attribute<>
        "irdl.parameters"(%2) <{names = ["elem"]}> : (!irdl.attribute<>) -> ()
    }) : () -> ()
    "irdl.operation"() <{sym_name = "norm"}> ({
      %3 = "irdl.any"() : () -> !irdl.attribute<>
      "irdl.operands"(%3) <{names = ["in"]}> : (!irdl.attribute<>) -> ()
      "irdl.results"(%3) <{names = ["out"]}> : (!irdl.attribute<>) -> ()
    }) : () -> ()
    "irdl.operation"() <{sym_name = "mul"}> ({
      %8 = "irdl.any"() : () -> !irdl.attribute<>
      "irdl.operands"(%8, %8) <{names = ["lhs", "rhs"]}> : (!irdl.attribute<>, !irdl.attribute<>) -> ()
      "irdl.results"(%8) <{names = ["res"]}> : (!irdl.attribute<>) -> ()
    }) : () -> ()
  }) : () -> ()
}) : () -> ()
""".stripMargin) match
    case Parsed.Success(module: ModuleOp, _) => module

  val dialect = module.body.blocks.head.operations.head.asInstanceOf[Dialect]
  val writer = StringWriter()
  printIRDL(dialect)(using PrintWriter(writer))

  writer.toString shouldEqual
    """package scair.dialects.cmath

import scair.dialects.builtin._
import scair.ir._
import scair.clair.macros._

case class Complex(
  elem: Attribute,
) extends DerivedAttribute["cmath.complex", Complex] with TypeAttribute
  derives DerivedAttributeCompanion

case class Norm(
  in: Operand[Attribute],
  out: Result[Attribute],
) extends DerivedOperation["cmath.norm", Norm]
  derives DerivedOperationCompanion

case class Mul(
  lhs: Operand[Attribute],
  rhs: Operand[Attribute],
  res: Result[Attribute],
) extends DerivedOperation["cmath.mul", Mul]
  derives DerivedOperationCompanion

val cmath = summonDialect[Complex *: EmptyTuple, Norm *: Mul *: EmptyTuple]
""".stripMargin
