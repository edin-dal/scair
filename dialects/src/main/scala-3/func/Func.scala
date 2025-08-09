package scair.dialects.func

import scair.clair.codegen.*
import scair.clair.macros.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.Parser
import scair.Parser._
import fastparse._
import fastparse.ParsingRun

case class Call(
    callee: SymbolRefAttr,
    _operands: Seq[Operand[Attribute]],
    _results: Seq[Result[Attribute]]
) extends DerivedOperation["func.call", Call] derives DerivedOperationCompanion

object Func {
    def parse[$: ParsingRun](parser: Parser): ParsingRun[Operation] =
        (("private".!).? ~/ parser.SymbolRefAttrP ~/ parser.BlockArgList.orElse(Seq()).map(parser.currentScope.defineValues) ~/ ("->" ~/ (parser.ParenTypeList) | parser.Type.map(Seq(_))).orElse(Seq()) ~/ parser.Region).map {
            case (visibility: Option[String], sym : SymbolRefAttr, args: Seq[Value[Attribute]], resultTypes: Seq[Attribute], body: Region) =>
                Func(
                    sym_name = sym.rootRef,
                    function_type = FunctionType(args.map(_.typ), resultTypes),
                    sym_visibility = visibility.map(StringData(_)),
                    body = body
                )
        }
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
