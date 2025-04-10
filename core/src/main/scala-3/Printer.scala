package scair

import scair.ir.*

import scala.collection.mutable

// ██████╗░ ██████╗░ ██╗ ███╗░░██╗ ████████╗ ███████╗ ██████╗░
// ██╔══██╗ ██╔══██╗ ██║ ████╗░██║ ╚══██╔══╝ ██╔════╝ ██╔══██╗
// ██████╔╝ ██████╔╝ ██║ ██╔██╗██║ ░░░██║░░░ █████╗░░ ██████╔╝
// ██╔═══╝░ ██╔══██╗ ██║ ██║╚████║ ░░░██║░░░ ██╔══╝░░ ██╔══██╗
// ██║░░░░░ ██║░░██║ ██║ ██║░╚███║ ░░░██║░░░ ███████╗ ██║░░██║
// ╚═╝░░░░░ ╚═╝░░╚═╝ ╚═╝ ╚═╝░░╚══╝ ░░░╚═╝░░░ ╚══════╝ ╚═╝░░╚═╝

case class Printer(
    val strictly_generic: Boolean = false,
    val indent: String = "  ",
    var valueNextID: Int = 0,
    var blockNextID: Int = 0,
    val valueNameMap: mutable.Map[Value[? <: Attribute], String] =
      mutable.Map.empty[Value[? <: Attribute], String],
    val blockNameMap: mutable.Map[Block, String] =
      mutable.Map.empty[Block, String]
) {

  /*≡==--==≡≡≡==--=≡≡*\
  ||      TOOLS      ||
  \*≡==---==≡==---==≡*/

  def assignValueName(value: Value[? <: Attribute]): String =
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

  /*≡==--==≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
  ||    ATTRIBUTE PRINTER    ||
  \*≡==---==≡≡≡≡≡≡≡≡≡==---==≡*/

  def printAttribute(attribute: Attribute): String = {
    return attribute.custom_print
  }

  /*≡==--==≡≡≡≡≡≡≡==--=≡≡*\
  ||    VALUE PRINTER    ||
  \*≡==---==≡≡≡≡≡==---==≡*/

  def printValue(value: Value[? <: Attribute]): String = {
    return s"%${assignValueName(value)}"
  }
  /*≡==--==≡≡≡≡≡≡≡==--=≡≡*\
  ||    BLOCK PRINTER    ||
  \*≡==---==≡≡≡≡≡==---==≡*/

  def printBlockArgument(value: Value[? <: Attribute]): String = {
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

  /*≡==--==≡≡≡≡≡≡≡≡==--=≡≡*\
  ||    REGION PRINTER    ||
  \*≡==---==≡≡≡≡≡≡==---==≡*/

  def printRegion(region: Region, indentLevel: Int = 0): String =
    this.copy()._printRegion(region, indentLevel)

  private def _printRegion(region: Region, indentLevel: Int = 0): String = {

    val open: String = "{\n"
    val close: String = "\n" + indent * indentLevel + "}"

    val regionBlocks: String = region.blocks match {
      case Nil => ""
      case entry :: blocks =>
        {
          // If the entry block has no arguments, we can avoid printing the header
          // Unless it is empty, which would make the next block read as the entry!
          (if (entry.arguments.nonEmpty || entry.operations.isEmpty) then
             printBlock(entry, indentLevel)
           else printOperations(entry.operations.toSeq, indentLevel + 1))
            :: (for { block <- blocks } yield printBlock(block, indentLevel))
        }.mkString("\n")

    }

    return s"${open}${regionBlocks}${close}"
  }

  /*≡==--==≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
  ||    OPERATION PRINTER    ||
  \*≡==---==≡≡≡≡≡≡≡≡≡==---==≡*/

  def printGenericMLIROperation(
      op: Operation,
      indentLevel: Int = 0
  ): String = {
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

    val properties: String =
      if (op.properties.size > 0)
        " <{" + (for {
          (key, value) <- op.properties
        } yield s"$key = ${value.custom_print}")
          .mkString(", ") + "}>"
      else ""

    val attributes: String =
      if (op.attributes.size > 0)
        " {" + (for {
          (key, value) <- op.attributes
        } yield s"$key = ${value.custom_print}")
          .mkString(", ") + "}"
      else ""

    val functionType: String =
      "(" + operandsTypes.mkString(", ") +
        ") -> (" +
        resultsTypes.mkString(", ") + ")"

    return s"${"\""}${op.name}${"\""}($operationOperands)$operationSuccessors$properties$operationRegions$attributes : $functionType"
  }

  def printOperation(op: Operation, indentLevel: Int = 0): String = {
    val results =
      op.results.map(printValue(_)).mkString(", ") + (if op.results.nonEmpty
                                                      then " = "
                                                      else "")
    indent * indentLevel + results + (if strictly_generic then
                                        printGenericMLIROperation(
                                          op,
                                          indentLevel
                                        )
                                      else op.custom_print(this))
  }

  def printOperations(
      ops: Seq[Operation],
      indentLevel: Int = 0
  ): String = {
    (for { op <- ops } yield printOperation(op, indentLevel)).mkString("\n")
  }

}
