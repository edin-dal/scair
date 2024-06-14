package scair

import fastparse._, MultiLineWhitespace._
import scala.collection.mutable
import scala.util.{Try, Success, Failure}
import Main._

object Printer {

  var indent: String = "  "

  var valueName: Int = 0
  var blockName: Int = 0

  var valueNameMap: mutable.Map[Value, String] =
    mutable.Map.empty[Value, String]
  var blockNameMap: mutable.Map[Block, String] =
    mutable.Map.empty[Block, String]

  def assignValueName(value: Value): String =
    valueNameMap.contains(value) match {
      case true => valueNameMap(value)
      case false =>
        val name = valueName.toString
        valueName = valueName + 1
        valueNameMap(value) = name
        return name
    }

  def assignBlockName(block: Block): String =
    blockNameMap.contains(block) match {
      case true => blockNameMap(block)
      case false =>
        val name = blockName.toString
        blockName = blockName + 1
        blockNameMap(block) = name
        return s"bb${name}"
    }

  // case class Region(
  //     blocks: Seq[Block],
  //     parent: Option[Operation] = None
  // )
  def printRegion(region: Region, indentLevel: Int = 0): String = {

    val open: String = indent * indentLevel + "{\n"
    val close: String = "\n" + indent * indentLevel + "}"

    val regionBlocks: String =
      (for { block <- region.blocks } yield printBlock(block, indentLevel + 1))
        .mkString("\n")

    return s"${open}${regionBlocks}${close}"
  }

  // case class Block(
  //     operations: Seq[Operation],
  //     arguments: Seq[Value]
  // )
  def printBlock(block: Block, indentLevel: Int = 0): String = {

    val blockArguments: String =
      (for { arg <- block.arguments } yield printValue(arg))
        .map { case (name: String, typ: String) => s"${name}: ${typ}" }
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
    return attribute.name
  }

  // case class Value(
  //     typ: Attribute
  // )
  def printValue(value: Value): (String, String) = {
    return (s"%${assignValueName(value)}", s"${printAttribute(value.typ)}")
  }

  // case class Operation(
  //     name: String,
  //     operands: Seq[Value], // TODO rest
  //     results: Seq[Value],
  //     regions: Seq[Region]
  // )
  def printOperation(op: Operation, indentLevel: Int = 0): String = {

    val open: String =
      if (op.regions.length > 0) "\n" + indent * indentLevel + "(\n" else ""
    val close: String =
      if (op.regions.length > 0) "\n" + indent * indentLevel + ")" else ""

    var results: Seq[String] = Seq()
    var resultsTypes: Seq[String] = Seq()
    var operands: Seq[String] = Seq()
    var operandsTypes: Seq[String] = Seq()

    for { res <- op.results } yield printValue(res) match {
      case (name: String, typ: String) =>
        results = results :+ name
        resultsTypes = resultsTypes :+ typ
    }

    for { oper <- op.operands } yield printValue(oper) match {
      case (name: String, typ: String) =>
        operands = operands :+ name
        operandsTypes = operandsTypes :+ typ
    }

    val operationResults: String =
      if (op.results.length > 0) results.mkString(", ") + " = " else ""

    val operationOperands: String =
      if (op.operands.length > 0) operands.mkString(", ") else ""

    val operationRegions: String =
      open + (for { region <- op.regions } yield printRegion(
        region,
        indentLevel + 1
      ))
        .mkString("\n") + close

    val functionType: String =
      "(" + operandsTypes.mkString(", ") +
        ") -> (" +
        resultsTypes.mkString(", ") + ")"

    return indent * indentLevel + operationResults + "\"" + op.name + "\" (" + operationOperands + ")" + operationRegions + " : " + functionType
  }

  def printProgram(programSequence: Seq[Operation]): String = {

    var programPrint: String = ""

    for { op <- programSequence } op match {
      case x: Operation => programPrint + printOperation(x)
    }

    return programPrint
  }

  def main(args: Array[String]): Unit = {
    println("Printer")

    val parser = new Parser()

    val Parsed.Success(result, _) = parser.testParse(
      text =
        "{^bb0(%5: i32):\n" + "%0, %1, %2 = \"test.op\"() : () -> (i32, i64, i32)\n" +
          "\"test.op\"(%1, %0) : (i64, i32) -> ()" + "^bb1(%4: i32):\n" + "%7, %8, %9 = \"test.op\"() : () -> (i32, i64, i32)\n" +
          "\"test.op\"(%8, %7) : (i64, i32) -> ()" + "}",
      parser = parser.Region(_)
    )

    print(printRegion(result))
  }
}
