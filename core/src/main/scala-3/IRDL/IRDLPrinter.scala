package scair.core.irdl_printer

import fastparse.Parsed
import scair.ir.Value
import scair.dialects.irdl._
import scair.Parser
import scair.MLContext
import scair.dialects.irdl.IRDL
import scala.io.Source
import java.io.PrintWriter

extension (operands: Operands)

  def info: Seq[(String, Value[AttributeType])] =
    operands.names.map(_.data) zip operands.args

extension (results: Results)

  def info: Seq[(String, Value[AttributeType])] =
    results.names.map(_.data) zip results.args

extension (parameters: Parameters)

  def info: Seq[(String, Value[AttributeType])] =
    parameters.names.map(_.data) zip parameters.args

extension (operation: Operation)

  def operandDefs = operation.body.blocks.head.operations.collect {
    case o: Operands => o
  }.head

  def resultDefs = operation.body.blocks.head.operations.collect {
    case r: Results => r
  }.head

extension (attribute: Attribute)

  def parameterDefs = attribute.body.blocks.head.operations.collect {
    case p: Parameters => p
  }.head

extension (t: Type)

  def parameterDefs = t.body.blocks.head.operations.collect {
    case p: Parameters => p
  }.head

object IRDLPrinter:

  def IRDLFileToScalaFile(irdl: String, scala: String): Unit =

    val input = Source.fromFile(irdl)

    val ctx = MLContext()
    ctx.registerDialect(IRDL)
    val parser = Parser(ctx)
    val dialect = parser.parseThis(
      input.mkString,
      pattern = parser.OperationPat(using _)
    ) match
      case Parsed.Success(dialect: Dialect, _) => dialect
    val printer = PrintWriter(scala)
    printIRDL(dialect)(using printer)
    printer.flush()
    printer.close()

  def printIRDL(dialect: Dialect)(using p: PrintWriter): Unit =

    given dialectName: String = dialect.sym_name.data
    p.print("package scair.dialects.")
    p.println(dialectName)
    p.println()
    p.println("import scair.dialects.builtin._")
    p.println("import scair.ir._")
    p.println("import scair.clair.macros._")
    p.println()
    dialect.body.blocks.head.operations.foreach({
      case op: Operation   => printOperation(op)
      case attr: Attribute => printAttribute(attr)
      case t: Type         => printType(t)
    })
    p.print("val ")
    p.print(dialect.sym_name.data)
    p.print(" = ")
    p.print("summonDialect[")
    dialect.body.blocks.head.operations.foreach({
      case attr: Attribute =>
        p.print(attr.sym_name.data)
        p.print(" *: ")
      case t: Type =>
        t.sym_name.data
        p.print(t.sym_name.data)
        p.print(" *: ")
      case _ =>
    })
    p.print("EmptyTuple, ")

    dialect.body.blocks.head.operations.foreach({
      case o: Operation =>
        p.print(o.sym_name.data)
        p.print(" *: ")
      case _ =>
    })

    p.println("EmptyTuple]")

  def printOperation(
      op: Operation
  )(using p: PrintWriter, dialectName: String): Unit =

    val className = op.sym_name.data
    val name = s"$dialectName.$className"

    p.print("final case class ")
    p.print(className)
    p.println("(")

    op.operandDefs.info.foreach((name, tpe) =>
      p.print("  ")
      p.print(name)
      p.print(": Operand[")
      printConstraint(tpe)
      p.println("],")
    )

    op.resultDefs.info.foreach((name, tpe) =>
      p.print("  ")
      p.print(name)
      p.print(": Result[")
      printConstraint(tpe)
      p.println("],")
    )

    p.print(") extends DerivedOperation[\"")
    p.print(name)
    p.print("\", ")
    p.print(className)
    p.println("]")
    p.println()

  def printConstraint(tpe: Value[AttributeType])(using p: PrintWriter): Unit =
    tpe.owner.get match
      case _: Any => p.print("Attribute")

  def printType(typ: Type)(using p: PrintWriter, dialectName: String): Unit =
    val className = typ.sym_name.data
    val name = s"$dialectName.$className"

    p.print("final case class ")
    p.print(className)
    p.println("(")

    typ.parameterDefs.info.foreach((name, tpe) =>
      p.print("  ")
      p.print(name)
      p.print(": ")
      printConstraint(tpe)
      p.println(",")
    )

    p.print(") extends DerivedAttribute[\"")
    p.print(name)
    p.print("\", ")
    p.print(className)
    p.println("] with TypeAttribute")
    p.println()

  def printAttribute(
      attr: Attribute
  )(using p: PrintWriter, dialectName: String): Unit =
    val className = attr.sym_name.data
    val name = s"$dialectName.$className"

    p.print("final case class ")
    p.print(className)
    p.println("(")

    attr.parameterDefs.info.foreach((name, tpe) =>
      p.print("  ")
      p.print(name)
      p.print(": ")
      printConstraint(tpe)
      p.println(",")
    )

    p.print(") extends DerivedAttribute[\"")
    p.print(name)
    p.print("\", ")
    p.print(className)
    p.println("]")
    p.println()
