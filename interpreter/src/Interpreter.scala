package scair.interpreter

import scair.dialects.builtin.*
import scair.ir.*

import scala.collection.mutable
import scala.reflect.ClassTag

// global implementation dictionary for interpreter
val impl_dict = mutable
  .Map[Class[? <: Operation], OpImpl[
    ? <: Operation
  ]]()

// custom operations should implement this trait
trait OpImpl[O <: Operation: ClassTag]:

  // get runtime class of operation type
  def opType: Class[O] = summon[ClassTag[O]].runtimeClass.asInstanceOf[Class[O]]

  // compute function to be implemented by each operation implementation
  // compute only needs to return result of operation, no need to worry about storing in context
  // if multiple results, return as Seq[Any]
  def compute(
      op: O,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Seq[Any],
  ): Seq[Any]

  // helper function to get operand values as TypeMap sequence
  // operands must be same type
  def lookup_operands(
      operands: Seq[Value[Attribute]],
      interpreter: Interpreter,
      ctx: RuntimeCtx,
  ): Seq[Any] =
    operands.map(op => interpreter.lookup_op(op, ctx))

  // run function that is automatically defined to store results in context after compute
  final def run(op: O, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
    var args = lookup_operands(op.operands, interpreter, ctx)

    // call compute to get result
    val result = compute(op, interpreter, ctx, args)

    // if operation has results, store them in context
    if op.results.nonEmpty then
      result match
        case Seq()       => () // no result to store
        case Seq(single) =>
          ctx.scopedDict.update(op.results.head, single) // store single result
        case multiple =>
          for (res, value) <- op.results.zip(multiple) do
            ctx.scopedDict.update(res, value) // store multiple results

// interpreter context class stores variables and current result

class RuntimeCtx(
    val scopedDict: ScopedDict,
    var result: Seq[Any] = Seq(),
):

  // creates new runtime ctx with new scope but shared symbol table
  def push_scope(name: String): RuntimeCtx =
    RuntimeCtx(
      ScopedDict(Some(this.scopedDict), mutable.Map(), name),
      Seq(),
    )

//
// ██╗ ███╗░░██╗ ████████╗ ███████╗ ██████╗░ ██████╗░ ██████╗░ ███████╗ ████████╗ ███████╗ ██████╗░
// ██║ ████╗░██║ ╚══██╔══╝ ██╔════╝ ██╔══██╗ ██╔══██╗ ██╔══██╗ ██╔════╝ ╚══██╔══╝ ██╔════╝ ██╔══██╗
// ██║ ██╔██╗██║ ░░░██║░░░ █████╗░░ ██████╔╝ ██████╔╝ ██████╔╝ █████╗░░ ░░░██║░░░ █████╗░░ ██████╔╝
// ██║ ██║╚████║ ░░░██║░░░ ██╔══╝░░ ██╔══██╗ ██╔═══╝░ ██╔══██╗ ██╔══╝░░ ░░░██║░░░ ██╔══╝░░ ██╔══██╗
// ██║ ██║░╚███║ ░░░██║░░░ ███████╗ ██║░░██║ ██║░░░░░ ██║░░██║ ███████╗ ░░░██║░░░ ███████╗ ██║░░██║
// ╚═╝ ╚═╝░░╚══╝ ░░░╚═╝░░░ ╚══════╝ ╚═╝░░╚═╝ ╚═╝░░░░░ ╚═╝░░╚═╝ ╚══════╝ ░░░╚═╝░░░ ╚══════╝ ╚═╝░░╚═╝
//

class Interpreter(
    val module: ModuleOp,
    val dialects: Seq[InterpreterDialect],
):

  val symbolTable: mutable.Map[String, Operation] = mutable.Map()
  val scopes: mutable.ArrayBuffer[ScopedDict] = mutable.ArrayBuffer()

  initialize_interpreter()

  def initialize_interpreter(): Unit =
    register_implementations()
    get_symbols_from_module()

  def get_symbols_from_module(): Unit =
    for op <- module.body.blocks.head.operations do
      op match
        case sym_and_table: (Symbol & SymbolTable) =>
          symbolTable.put(sym_and_table.sym_name.stringLiteral, sym_and_table)
          get_symbols_from_symbol_table(sym_and_table)
        case sym_table: SymbolTable =>
          get_symbols_from_symbol_table(sym_table)
        case sym_op: Symbol =>
          symbolTable.put(sym_op.sym_name.stringLiteral, sym_op)
        case _ => ()

  def get_symbols_from_symbol_table(sym_table: SymbolTable): Unit =
    for op <- sym_table.regions.head.blocks.head.operations do
      op match
        case sym_op: Symbol =>
          symbolTable.put(sym_op.sym_name.stringLiteral, sym_op)
        case _ => ()

  // lookup function for context variables
  // does not work for Bool-like vals due to inability to prove disjoint for TypeMap
  def lookup_op[T <: Value[Attribute]](value: T, ctx: RuntimeCtx): Any =
    ctx.scopedDict.get(value) match
      case Some(v) => v
      case _       =>
        throw new Exception(s"Variable $value not found in context: $ctx")

  def register_implementations(): Unit =
    for dialect <- dialects do
      for impl <- dialect do impl_dict.put(impl.opType, impl)

  def create_scope(name: String): RuntimeCtx =
    RuntimeCtx(ScopedDict(None, mutable.Map(), name), Seq())

  def interpret_op(op: Operation, ctx: RuntimeCtx): Unit =
    val impl = impl_dict.get(op.getClass)
    impl match
      case Some(impl) => impl.asInstanceOf[OpImpl[Operation]].run(op, this, ctx)
      case None       =>
        throw new Exception(
          s"Unsupported operation when interpreting: ${op.getClass}"
        )

  def interpret_region(region: Region, ctx: RuntimeCtx): Unit =
    for operation <- region.blocks.head.operations do
      interpret_op(operation, ctx)

  def interpret_block(block: Block, ctx: RuntimeCtx): Unit =
    for operation <- block.operations do interpret_op(operation, ctx)

  def call_op(op: Operation, ctx: RuntimeCtx): Seq[Any] =
    val impl = impl_dict.get(op.getClass)
    impl match
      case Some(impl) =>
        impl.asInstanceOf[OpImpl[Operation]].compute(
          op,
          this,
          ctx,
          Seq(),
        )
      case None =>
        throw new Exception(
          s"Unsupported operation when interpreting: ${op.getClass}"
        )

  // helper function to print values in interpreter
  // useful if booleans are represented as 0 and 1
  def interpreter_print(value: Any): Unit =
    value match
      case 0 => print("false\n")
      case 1 => print("true\n")
      case _ => print(s"$value\n")
