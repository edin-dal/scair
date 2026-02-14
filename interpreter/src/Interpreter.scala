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
  ): Any

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
        // multiple results
        case s: Seq[Any] =>
          if op.results.length != s.length
          then // must be an list-like expression
            ctx.scopedDict.update(op.results.head, s)
          else // multiple distinct results
            for (res, value) <- op.results.zip(s) do
              ctx.scopedDict.update(res, value)
        case _ =>
          ctx.scopedDict.update(op.results.head, result)

// interpreter context class stores variables, function definitions and the current result
class RuntimeCtx(
    val scopedDict: ScopedDict,
    var result: Option[Any] = None,
):

  // creates new runtime ctx with new scope but shared symbol table
  def push_scope(name: String): RuntimeCtx =
    RuntimeCtx(
      ScopedDict(Some(this.scopedDict), mutable.Map(), name),
      None,
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
    val symbolTable: mutable.Map[String, Operation] = mutable.Map(),
    val scopes: mutable.ArrayBuffer[ScopedDict] = mutable.ArrayBuffer(),
    val dialects: Seq[InterpreterDialect],
):

  val globalRuntimeCtx =
    RuntimeCtx(ScopedDict(None, mutable.Map(), "global"), None)

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

  def lookup_boollike(value: Value[Attribute], ctx: RuntimeCtx): Int =
    ctx.scopedDict.get(value) match
      case Some(v: Int) => v
      case _            =>
        throw new Exception(s"Bool-like $value not found in context: $ctx")

  def register_implementations(): Unit =
    for dialect <- dialects do
      for impl <- dialect do impl_dict.put(impl.opType, impl)

  // keeping buffer function for extensibility
  def interpret(block: Block, ctx: RuntimeCtx): Option[Any] =
    for op <- block.operations do interpret_op(op, ctx)
    ctx.result

  def interpret_op(op: Operation, ctx: RuntimeCtx): Unit =
    val impl = impl_dict.get(op.getClass)
    impl match
      case Some(impl) => impl.asInstanceOf[OpImpl[Operation]].run(op, this, ctx)
      case None       =>
        throw new Exception(
          s"Unsupported operation when interpreting: ${op.getClass}"
        )

  def interpreter_print(value: Any): Unit =
    value match
      case 0 => println("false")
      case 1 => println("true")
      case _ => println(s"Result: $value")
