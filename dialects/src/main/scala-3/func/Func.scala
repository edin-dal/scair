package scair.dialects.func

import fastparse.*
import fastparse.ParsingRun
import scair.Parser
import scair.Parser.*
import scair.clair.codegen.*
import scair.clair.macros.*
import scair.dialects.builtin.*
import scair.ir.*

case class Call(
    callee: SymbolRefAttr,
    _operands: Seq[Operand[Attribute]],
    _results: Seq[Result[Attribute]]
) extends DerivedOperation["func.call", Call] derives DerivedOperationCompanion

object Func {

  def parse[$: ParsingRun](parser: Parser): ParsingRun[Operation] =
    ("private".!.? ~ parser.SymbolRefAttrP ~ ((parser.BlockArgList.orElse(Seq())
      ~ ("->" ~ (parser.Type.map(Seq(_)) | parser.ParenTypeList)).orElse(Seq()))
      .flatMap((args, resTypes) => (parser.Region(args).map((_, resTypes))))))
      .map({
        case (visibility, symbol, (body, resTypes)) =>
          Func(
            sym_name = symbol.rootRef,
            function_type = FunctionType(
              inputs = body.blocks.head.arguments.map(_.typ).toSeq,
              outputs = resTypes
            ),
            sym_visibility = visibility.map(StringData(_)),
            body = body
          )
        case _ => throw new Exception("sioadih")
      })

}

case class Func(
    sym_name: StringData,
    function_type: FunctionType,
    sym_visibility: Option[StringData],
    body: Region
) extends DerivedOperation["func.func", Func]
    with IsolatedFromAbove derives DerivedOperationCompanion

case class Return(
    _operands: Seq[Operand[Attribute]]
) extends DerivedOperation["func.return", Return]
    with AssemblyFormat["attr-dict ($_operands^ `:` type($_operands))?"]
    with NoMemoryEffect
    with IsTerminator derives DerivedOperationCompanion

val FuncDialect = summonDialect[EmptyTuple, (Call, Func, Return)](Seq())
