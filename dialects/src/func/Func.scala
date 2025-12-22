package scair.dialects.func

import fastparse.*
import scair.*
import scair.Printer
import scair.clair.codegen.*
import scair.clair.macros.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.parse.*
import scair.parse.Parser

//
// ███████╗ ██╗░░░██╗ ███╗░░██╗ ░█████╗░
// ██╔════╝ ██║░░░██║ ████╗░██║ ██╔══██╗
// █████╗░░ ██║░░░██║ ██╔██╗██║ ██║░░╚═╝
// ██╔══╝░░ ██║░░░██║ ██║╚████║ ██║░░██╗
// ██║░░░░░ ╚██████╔╝ ██║░╚███║ ╚█████╔╝
// ╚═╝░░░░░ ░╚═════╝░ ╚═╝░░╚══╝ ░╚════╝░
//

case class Call(
    callee: SymbolRefAttr,
    _operands: Seq[Operand[Attribute]],
    _results: Seq[Result[Attribute]],
) extends DerivedOperation["func.call", Call] derives DerivedOperationCompanion

given OperationCustomParser[Func]:

  def parseResultTypes[$: P](using
      Parser
  ): P[Seq[Attribute]] = ("->" ~ (parenTypeListP | typeP.map(Seq(_))))
    .orElse(Seq())

  def parse[$: P](
      resNames: Seq[String]
  )(using Parser): P[Func] =
    ("private".!.? ~ symbolRefAttrP ~
      (("(" ~ valueIdAndTypeP.rep(sep = ",") ~ ")")
        .flatMap((args: Seq[(String, Attribute)]) =>
          Pass(
            args.map(_._2)
          ) ~ parseResultTypes ~ ("attributes" ~ attributeDictionaryP)
            .orElse(Map()) ~ regionP(args)
        ) |
        (
          parenTypeListP ~ parseResultTypes ~
            ("attributes" ~ attributeDictionaryP).orElse(Map()) ~ Pass(
              Region()
            )
        ))).map {
      case (visibility, symbol, (argTypes, resTypes, attributes, body)) =>
        val f = Func(
          sym_name = symbol.rootRef,
          function_type = FunctionType(
            inputs = argTypes,
            outputs = resTypes,
          ),
          sym_visibility = visibility.map(StringData(_)),
          body = body,
        )
        f.attributes.addAll(attributes)
        f
    }

case class Func(
    sym_name: StringData,
    function_type: FunctionType,
    sym_visibility: Option[StringData],
    body: Region,
) extends DerivedOperation["func.func", Func]
    with IsolatedFromAbove derives DerivedOperationCompanion:

  override def customPrint(printer: Printer)(using indentLevel: Int) =
    val lprinter = printer.copy()
    lprinter.print("func.func ")
    sym_visibility match
      case Some(visibility) =>
        lprinter.print(visibility.data); lprinter.print(" ")
      case None => ()
    lprinter.print("@")
    lprinter.print(sym_name.data)
    body.blocks match
      case Seq() =>
        lprinter.print(function_type)
      case entry :: _ =>
        lprinter.printListF(
          entry.arguments,
          lprinter.printArgument,
          "(",
          ", ",
          ")",
        )
        if function_type.outputs.nonEmpty then
          lprinter.print(" -> ")
          function_type.outputs match
            case Seq(single) =>
              lprinter.print(single)
            case outputs =>
              lprinter.printList(outputs, "(", ", ", ")")

    if attributes.nonEmpty then
      lprinter.print(" attributes")
      lprinter.printOptionalAttrDict(attributes.toMap)
    // TODO: Should that simply be a region print?
    body.blocks match
      case Seq()           => ()
      case entry :: others =>
        lprinter.print(" {\n")
        entry.operations.foreach(lprinter.print(_)(using indentLevel + 1))
        others.foreach(lprinter.print)
        lprinter.print(lprinter.indent * indentLevel + "}")

case class Return(
    _operands: Seq[Operand[Attribute]]
) extends DerivedOperation["func.return", Return]
    with AssemblyFormat["attr-dict ($_operands^ `:` type($_operands))?"]
    with NoMemoryEffect
    with IsTerminator derives DerivedOperationCompanion

val FuncDialect = summonDialect[EmptyTuple, (Call, Func, Return)]
