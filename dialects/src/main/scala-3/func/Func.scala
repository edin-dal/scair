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
trait DeclareOpInterfaceMethods[T <: Tuple]
trait TypesMatchWith(tuples: (Seq[Attribute], Seq[Attribute])*)
trait ConstantLike
type Pure = NoMemoryEffect
trait OpAsmOpInterface
trait AffineScope
trait AutomaticAllocationScope
trait FunctionOpInterface
trait IsolatedFromAbove
trait HasParent[P <: Operation]
trait ReturnLike
trait InferTypeOpAdaptor

/*≡=--==≡≡≡≡==--=≡*\
||   ATTRIBUTES   ||
\*≡=---==≡≡==---=≡*/

type UnitAttr = Attribute

/*≡=--==≡≡≡≡==--=≡*\
||   OPERATIONS   ||
\*≡=---==≡≡==---=≡*/

case class Call(
    val _operands: Seq[Operand[Attribute]],
    val _results: Seq[Result[Attribute]],
    val callee: SymbolRefAttr,
    val arg_attrs: Option[ArrayAttribute[DictionaryAttr]],
    val res_attrs: Option[ArrayAttribute[DictionaryAttr]],
    val no_inline: UnitAttr
) extends DerivedOperation["func.call", Call],
      CallOpInterface,
      MemRefsNormalizable,
      DeclareOpInterfaceMethods[Tuple1[SymbolUserOpInterface]]
    derives DerivedOperationCompanion

case class CallIndirect(
    val callee: Operand[FunctionType],
    val callee_operands: Seq[Operand[Attribute]],
    val _results: Seq[Result[Attribute]],
    val arg_attrs: Option[ArrayAttribute[DictionaryAttr]],
    val res_attrs: Option[ArrayAttribute[DictionaryAttr]]
) extends DerivedOperation["func.call_indirect", CallIndirect],
      TypesMatchWith(
        (callee.typ.inputs, callee_operands.map(_.typ)),
        (callee.typ.outputs, _results.map(_.typ))
      ),
      CallOpInterface derives DerivedOperationCompanion

case class ConstantOp(
    val value: SymbolRefAttr,
    val _results: Result[TypeAttribute]
) extends DerivedOperation["func.constant", ConstantOp],
      ConstantLike,
      Pure,
      DeclareOpInterfaceMethods[(SymbolUserOpInterface, OpAsmOpInterface)]
    derives DerivedOperationCompanion

case class Func(
    val sym_name: StringData,
    val function_type: FunctionType,
    val sym_visibility: Option[StringData],
    val arg_attrs: Option[ArrayAttribute[DictionaryAttr]],
    val res_attrs: Option[ArrayAttribute[DictionaryAttr]],
    val body: Region,
    val no_inline: UnitAttr
) extends DerivedOperation["func.func", Func],
      OpAsmOpInterface,
      AffineScope,
      AutomaticAllocationScope,
      FunctionOpInterface,
      IsolatedFromAbove derives DerivedOperationCompanion

case class Return(
    _operands: Seq[Operand[Attribute]]
) extends DerivedOperation["func.return", Return],
      AssemblyFormat["attr-dict ($_operands^ `:` type($_operands))?"],
      Pure,
      HasParent[Func],
      MemRefsNormalizable,
      ReturnLike,
      IsTerminator derives DerivedOperationCompanion

val FuncDialect = summonDialect[EmptyTuple, (Call, Func, Return)](Seq())
