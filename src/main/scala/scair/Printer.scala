package scair

import fastparse._, MultiLineWhitespace._
import scala.collection.mutable
import scala.util.{Try, Success, Failure}
import IR._
import AttrParser._

class Printer {

  var indent: String = "  "

  var valueNextID: Int = 0
  var blockNextID: Int = 0

  var valueNameMap: mutable.Map[Value, String] =
    mutable.Map.empty[Value, String]
  var blockNameMap: mutable.Map[Block, String] =
    mutable.Map.empty[Block, String]

  def assignValueName(value: Value): String =
    valueNameMap.contains(value) match {
      case true => valueNameMap(value)
      case false =>
        val name = valueNextID.toString
        valueNextID = valueNextID + 1
        valueNameMap(value) = name
        return name
    }

  def assignBlockName(block: Block): String =
    blockNameMap.contains(block) match {
      case true => blockNameMap(block)
      case false =>
        val name = s"bb${blockNextID.toString}"
        blockNextID = blockNextID + 1
        blockNameMap(block) = name
        return name
    }

  // case class Region(
  //     blocks: Seq[Block],
  //     parent: Option[Operation] = None
  // )
  def printRegion(region: Region, indentLevel: Int = 0): String = {

    val open: String = "{\n"
    val close: String = "\n" + indent * indentLevel + "}"

    val regionBlocks: String =
      (for { block <- region.blocks } yield printBlock(block, indentLevel))
        .mkString("\n")

    return s"${open}${regionBlocks}${close}"
  }

  // case class Block(
  //     operations: Seq[Operation],
  //     arguments: Seq[Value]
  // )
  def printBlock(block: Block, indentLevel: Int = 0): String = {

    val blockArguments: String =
      (for { arg <- block.arguments } yield printBlockArgument(arg))
        .mkString(", ")

    val blockOperations: String =
      (for { op <- block.operations } yield printOperation(op, indentLevel + 1))
        .mkString("\n")

    val blockHead: String =
      indent * indentLevel + s"^${assignBlockName(block)}(${blockArguments}):\n"

    return blockHead + blockOperations
  }

  // case class Attribute(
  //     name: String
  // )
  def printAttribute(attribute: Attribute): String = {
    return attribute.toString
  }

  // case class Value(
  //     typ: Attribute
  // )
  def printValue(value: Value): String = {
    return s"%${assignValueName(value)}"
  }

  def printBlockArgument(value: Value): String = {
    return s"${printValue(value)}: ${printAttribute(value.typ)}"
  }

  // case class Operation(
  //     name: String,
  //     operands: Seq[Value], // TODO rest
  //     results: Seq[Value],
  //     regions: Seq[Region]
  // )
  def printOperation(op: Operation, indentLevel: Int = 0): String = {

    val open: String =
      if (op.regions.length > 0) indent * indentLevel + "(" else ""
    val close: String =
      if (op.regions.length > 0) ")" else ""

    var results: Seq[String] = Seq()
    var resultsTypes: Seq[String] = Seq()
    var operands: Seq[String] = Seq()
    var operandsTypes: Seq[String] = Seq()

    for { res <- op.results } yield (
      results = results :+ printValue(res),
      resultsTypes = resultsTypes :+ printAttribute(res.typ)
    )

    for { oper <- op.operands } yield (
      operands = operands :+ printValue(oper),
      operandsTypes = operandsTypes :+ printAttribute(oper.typ)
    )

    val operationResults: String =
      if (op.results.length > 0) results.mkString(", ") + " = " else ""

    val operationOperands: String =
      if (op.operands.length > 0) operands.mkString(", ") else ""

    val operationRegions: String =
      open + (for { region <- op.regions } yield printRegion(
        region,
        indentLevel + 1
      )).mkString(", ") + close

    val operationSuccessors: String =
      if (op.successors.length > 0)
        "[" + (for { successor <- op.successors } yield blockNameMap(successor))
          .map((x: String) => "^" + x)
          .mkString(", ") + "]"
      else ""

    val functionType: String =
      "(" + operandsTypes.mkString(", ") +
        ") -> (" +
        resultsTypes.mkString(", ") + ")"

    return indent * indentLevel + operationResults + "\"" + op.name + "\"(" + operationOperands + ")" + operationSuccessors + operationRegions + " : " + functionType
  }

  def printProgram(programSequence: Seq[Operation]): String = {

    var programPrint: String = ""

    for { op <- programSequence } op match {
      case x: Operation => programPrint + printOperation(x)
    }

    return programPrint
  }
}
object Printer {
  def main(args: Array[String]): Unit = {
    // println("Printer")

    val parser = new Parser()
    val printer = new Printer()

    val input = """"op1"()({
                  |  ^bb0(%0: f128):
                  |    %1, %2, %3 = "test.op"() : () -> (f32, si64, ui80)
                  |  }) : () -> ()""".stripMargin

    val Parsed.Success(res, x) = parser.parseThis(
      text = input,
      pattern = parser.OperationPat(_)
    )

    println(printer.printOperation(res))

    // println(printOperation(result.asInstanceOf[Operation]))
  }
}
