package scair.dialects.func

import scair.clair.codegen.*
import scair.clair.macros.*
import scair.dialects.builtin.*
import scair.ir.*

/*≡=--==≡≡≡≡==--=≡*\
||   INTERFACES   ||
\*≡=---==≡≡==---=≡*/

trait CallOpInterface
trait MemRefsNormalizable
trait SymbolUserOpInterface
trait DeclareOpInterfaceMethods[T]

/*≡=--==≡≡≡≡==--=≡*\
||   ATTRIBUTES   ||
\*≡=---==≡≡==---=≡*/

type UnitAttr = Attribute

/*≡=--==≡≡≡≡==--=≡*\
||   OPERATIONS   ||
\*≡=---==≡≡==---=≡*/

case class Call(
    _operands: Seq[Operand[Attribute]],
    _results: Seq[Result[Attribute]],
    callee: SymbolRefAttr,
    arg_attrs: Option[ArrayAttribute[DictionaryAttr]],
    res_attrs: Option[ArrayAttribute[DictionaryAttr]],
    no_inline: UnitAttr
) extends DerivedOperation["func.call", Call],
      CallOpInterface,
      MemRefsNormalizable,
      DeclareOpInterfaceMethods[SymbolUserOpInterface]
    derives DerivedOperationCompanion

case class CallIndirect(
    _operands: Seq[Operand[Attribute]],
    _results: Seq[Result[Attribute]],
    callee: FunctionType,
    arg_attrs: Option[ArrayAttribute[DictionaryAttr]],
    res_attrs: Option[ArrayAttribute[DictionaryAttr]],
    no_inline: UnitAttr
) extends DerivedOperation["func.call_indirect", CallIndirect],
      CallOpInterface derives DerivedOperationCompanion

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
