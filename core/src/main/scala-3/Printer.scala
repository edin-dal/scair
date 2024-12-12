package scair

import fastparse.*
import scair.ir.*

import scala.collection.mutable

class Printer(val strictly_generic: Boolean) {

  ///////////
  // TOOLS //
  ///////////

  var indent: String = "  "

  var valueNextID: Int = 0
  var blockNextID: Int = 0

  var valueNameMap: mutable.Map[Value[_ <: Attribute], String] =
    mutable.Map.empty[Value[_ <: Attribute], String]
  var blockNameMap: mutable.Map[Block, String] =
    mutable.Map.empty[Block, String]

  def assignValueName(value: Value[_ <: Attribute]): String =
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
      case true =>
        blockNameMap(block)
      case false =>
        val name = s"bb${blockNextID.toString}"
        blockNextID = blockNextID + 1
        blockNameMap(block) = name
        return name
    }

  ///////////////
  // ATTRIBUTE //
  ///////////////

  def printAttribute(attribute: Attribute): String = {
    return attribute.custom_print
  }

  ///////////
  // VALUE //
  ///////////

  def printValue(value: Value[_ <: Attribute]): String = {
    return s"%${assignValueName(value)}"
  }

  ///////////
  // BLOCK //
  ///////////

  def printBlockArgument(value: Value[_ <: Attribute]): String = {
    return s"${printValue(value)}: ${printAttribute(value.typ)}"
  }

  def printBlock(block: Block, indentLevel: Int = 0): String = {

    val blockName = assignBlockName(block)

    val blockArguments: String =
      (for { arg <- block.arguments } yield printBlockArgument(arg))
        .mkString(", ")

    val blockOperations: String =
      printOperations(block.operations.toSeq, indentLevel + 1)

    val blockHead: String =
      indent * indentLevel + s"^${blockName}(${blockArguments}):\n"

    return blockHead + blockOperations
  }

  ////////////
  // REGION //
  ////////////

  def printRegion(region: Region, indentLevel: Int = 0): String = {

    val open: String = "{\n"
    val close: String = "\n" + indent * indentLevel + "}"

    val regionBlocks: String = (region.blocks match {
      case Nil             => Seq("")
      case entry :: blocks =>
        // If the entry block has no arguments, we can avoid printing the header
        // Unless it is empty, which would make the next block read as the entry!
        (if (entry.arguments.nonEmpty || entry.operations.isEmpty) then
           printBlock(entry, indentLevel)
         else printOperations(entry.operations.toSeq, indentLevel + 1))
          :: (for { block <- blocks } yield printBlock(block, indentLevel))

    }).mkString("\n")

    return s"${open}${regionBlocks}${close}"
  }

  ///////////////
  // OPERATION //
  ///////////////

  def printCustomOperation(op: Operation, indentLevel: Int = 0): String = {
    indent * indentLevel + op.print(this)
  }

  def printGenericOperation(op: Operation, indentLevel: Int = 0): String = {
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
      if (op.results.length > 0)
        results.mkString(", ") + " = "
      else ""

    val operationOperands: String =
      if (op.operands.length > 0)
        operands.mkString(", ")
      else ""

    val operationRegions: String =
      if (op.regions.length > 0)
        " (" + (for { region <- op.regions } yield printRegion(
          region,
          indentLevel
        )).mkString(", ") + ")"
      else ""

    val operationSuccessors: String =
      if (op.successors.length > 0)
        "[" + (for { successor <- op.successors } yield assignBlockName(
          successor
        ))
          .map((x: String) => "^" + x)
          .mkString(", ") + "]"
      else ""

    val dictionaryProperties: String =
      if (op.dictionaryProperties.size > 0)
        " <{" + (for {
          (key, value) <- op.dictionaryProperties
        } yield s"$key = ${value.custom_print}")
          .mkString(", ") + "}>"
      else ""

    val dictionaryAttributes: String =
      if (op.dictionaryAttributes.size > 0)
        " {" + (for {
          (key, value) <- op.dictionaryAttributes
        } yield s"$key = ${value.custom_print}")
          .mkString(", ") + "}"
      else ""

    val functionType: String =
      "(" + operandsTypes.mkString(", ") +
        ") -> (" +
        resultsTypes.mkString(", ") + ")"

    return indent * indentLevel + s"$operationResults${"\""}${op.name}${"\""}($operationOperands)$operationSuccessors$dictionaryProperties$operationRegions$dictionaryAttributes : $functionType"
  }

  def printOperation(op: Operation, indentLevel: Int = 0): String = {
    strictly_generic match {
      case true => printGenericOperation(op, indentLevel)
      case false =>
        try {
          printCustomOperation(op, indentLevel)
        } catch {
          case e: Exception =>
            printGenericOperation(op, indentLevel)
        }
    }
  }

  def printOperations(ops: Seq[Operation], indentLevel: Int = 0): String = {
    strictly_generic match {
      case true =>
        (for { op <- ops } yield printGenericOperation(op, indentLevel))
          .mkString("\n")
      case false =>
        (for { op <- ops } yield try {
          printCustomOperation(op, indentLevel)
        } catch {
          case e: Exception =>
            printGenericOperation(op, indentLevel)
        }).mkString("\n")
    }
  }
}

object Printer {

  def main(args: Array[String]): Unit = {
    // println("Printer")

    val parser = new Parser(MLContext())
    val printer = new Printer(true)

    val input = """"test.op"({
                  |  ^bb0(%0: f128):
                  |    %1, %2, %3 = "test.op"() : () -> (f32, si64, ui80)
                  |  }) : () -> ()""".stripMargin

    val text = """"test.op"({
                 |  ^bb3():
                 |    "test.op"()[^bb4] : () -> ()
                 |  ^bb4():
                 |    "test.op"()[^bb3] : () -> ()
                 |  }) : () -> ()""".stripMargin

    val text2 = """"builtin.module"() ({
                  |^bb0():
                  |  %0 = "test.op"() {"quoted" = i3298} : () -> (i32)
                  |  "test.op"() {hello = tensor<f32>} : () -> ()
                  |  "test.op"() {hello = tensor<1xf32>} : () -> ()
                  |  "test.op"() {hello = tensor<?xf32>} : () -> ()
                  |  "test.op"() {hello = tensor<3x?x5xf32>} : () -> ()
                  |  "test.op"() {hello = tensor<?x5x?xf32>} : () -> ()
                  |  "test.op"(%0) : (i32) -> ()
                  |}) : () -> ()""".stripMargin

    val Parsed.Success(res, x) = parser.parseThis(
      text = text2,
      pattern = parser.OperationPat(_)
    )

    println(printer.printOperation(res))

    // println(printOperation(result.asInstanceOf[Operation]))
  }
}
