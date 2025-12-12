package scair.core.irdl_printer

import fastparse.Parsed
import scair.MLContext
import scair.dialects.builtin.ArrayAttribute
import scair.dialects.irdl.*
import scair.dialects.irdl.IRDL
import scair.ir.Value
import scair.parse.*

import java.io.PrintWriter
import scala.io.Source

// ██╗ ██████╗░ ██████╗░ ██╗░░░░░
// ██║ ██╔══██╗ ██╔══██╗ ██║░░░░░
// ██║ ██████╔╝ ██║░░██║ ██║░░░░░
// ██║ ██╔══██╗ ██║░░██║ ██║░░░░░
// ██║ ██║░░██║ ██████╔╝ ███████╗
// ╚═╝ ╚═╝░░╚═╝ ╚═════╝░ ╚══════╝
//
// ██████╗░ ██████╗░ ██╗ ███╗░░██╗ ████████╗ ███████╗ ██████╗░
// ██╔══██╗ ██╔══██╗ ██║ ████╗░██║ ╚══██╔══╝ ██╔════╝ ██╔══██╗
// ██████╔╝ ██████╔╝ ██║ ██╔██╗██║ ░░░██║░░░ █████╗░░ ██████╔╝
// ██╔═══╝░ ██╔══██╗ ██║ ██║╚████║ ░░░██║░░░ ██╔══╝░░ ██╔══██╗
// ██║░░░░░ ██║░░██║ ██║ ██║░╚███║ ░░░██║░░░ ███████╗ ██║░░██║
// ╚═╝░░░░░ ╚═╝░░╚═╝ ╚═╝ ╚═╝░░╚══╝ ░░░╚═╝░░░ ╚══════╝ ╚═╝░░╚═╝
//

extension (operands: Operands)

  def info: Seq[(String, Value[AttributeType])] =
    operands.names.map(_.data) zip operands.args

extension (results: Results)

  def info: Seq[(String, Value[AttributeType])] =
    results.names.map(_.data) zip results.args

extension (attributes: Attributes)

  def info: Seq[(String, Value[AttributeType])] =
    attributes.attribute_value_names.map(_.data) zip attributes.args

extension (parameters: Parameters)

  def info: Seq[(String, Value[AttributeType])] =
    parameters.names.map(_.data) zip parameters.args

extension (operation: Operation)

  def operandDefs =
    val x = operation.body.blocks.head.operations.collect { case o: Operands =>
      o
    }
    if x.isEmpty then Operands(Seq(), ArrayAttribute(Seq()))
    else x.head

  def resultDefs =
    val x = operation.body.blocks.head.operations.collect { case r: Results =>
      r
    }
    if x.isEmpty then Results(Seq(), ArrayAttribute(Seq()))
    else x.head

  def attributeDefs =
    val x = operation.body.blocks.head.operations.collect {
      case r: Attributes => r
    }
    if x.isEmpty then Attributes(Seq(), ArrayAttribute(Seq()))
    else x.head

extension (attribute: Attribute)

  def parameterDefs = attribute.body.blocks.head.operations.collect {
    case p: Parameters => p
  }.head

extension (t: Type)

  def parameterDefs = t.body.blocks.head.operations.collect {
    case p: Parameters => p
  }.head

object IRDLPrinter:

  def IRDLFileToScalaFile(irdl: String, dir: String): Unit =

    val input = Source.fromFile(irdl)

    val ctx = MLContext()
    ctx.registerDialect(IRDL)
    val parser = Parser(ctx)
    val dialect = parser.parse(
      input = input.mkString,
      parser = operationP(using _, parser),
    ) match
      case Parsed.Success(dialect: Dialect, _) => dialect
      case Parsed.Success(_, _)                =>
        throw Exception(
          s"Parsed IRDL file did not yield a Dialect operation"
        )

      case Parsed.Failure(str, _, _) =>
        throw Exception(s"Failed to parse IRDL file:\n $str")

    val scalafile = s"$dir/${dialect.sym_name.data}.scala"
    val printer = PrintWriter(scalafile)
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
    dialect.body.blocks.head.operations.foreach {
      case op: Operation   => printOperation(op)
      case attr: Attribute => printAttribute(attr)
      case t: Type         => printType(t)
    }
    p.print("val ")
    p.print(dialect.sym_name.data)
    p.print(" = ")
    p.print("summonDialect[")
    dialect.body.blocks.head.operations.foreach {
      case attr: Attribute =>
        p.print(attr.sym_name.data.capitalize)
        p.print(" *: ")
      case t: Type =>
        p.print(t.sym_name.data.capitalize)
        p.print(" *: ")
      case _ =>
    }
    p.print("EmptyTuple, ")

    dialect.body.blocks.head.operations.foreach {
      case o: Operation =>
        p.print(o.sym_name.data.capitalize)
        p.print(" *: ")
      case _ =>
    }

    p.println("EmptyTuple]")

  def printOperation(
      op: Operation
  )(using p: PrintWriter, dialectName: String): Unit =

    val className = op.sym_name.data
    val name = s"$dialectName.$className"

    p.print("case class ")
    p.print(className.capitalize)
    p.println("(")

    op.operandDefs.info.foreach((name, tpe) =>
      p.print("  ")
      p.print(name)
      p.print(": Operand[")
      printConstraint(tpe)
      p.println("],")
    )

    op.attributeDefs.info.foreach((name, tpe) =>
      p.print("  ")
      p.print(name)
      p.print(": ")
      printConstraint(tpe)
      p.println(",")
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
    p.print(className.capitalize)
    p.println("]")
    p.println("  derives DerivedOperationCompanion")
    p.println()

  def printConstraint(tpe: Value[AttributeType])(using p: PrintWriter): Unit =
    tpe.owner.get match
      case _: Any => p.print("Attribute")
      case owner  =>
        throw Exception(
          s"Unsupported constraint:\n$owner"
        )

  def printType(typ: Type)(using p: PrintWriter, dialectName: String): Unit =
    val className = typ.sym_name.data
    val name = s"$dialectName.$className"

    p.print("case class ")
    p.print(className.capitalize)
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
    p.print(className.capitalize)
    p.println("] with TypeAttribute")
    p.println("  derives DerivedAttributeCompanion")
    p.println()

  def printAttribute(
      attr: Attribute
  )(using p: PrintWriter, dialectName: String): Unit =
    val className = attr.sym_name.data
    val name = s"$dialectName.$className"

    p.print("case class ")
    p.print(className.capitalize)
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
    p.print(className.capitalize)
    p.println("]")
    p.println("  derives DerivedAttributeCompanion")
    p.println()
