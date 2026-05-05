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

  // run function that is automatically defined to store results in context after compute
  final def run(op: O, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
    var args = interpreter.get_values(op.operands, ctx)

    // call compute to get result
    val result = compute(op, interpreter, ctx, args)
    val op_results = op.results

    if op_results.size == 0 then ()
    else if op_results.size == 1 then
      ctx.scopedDict.update(op_results.head, result.head)
    else
      var i = 0
      while i < op_results.length do
        ctx.scopedDict.update(op_results(i), result(i))
        i += 1

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
        throw new Exception(
          s"Variable $value not found in context: ${ctx.scopedDict.name}"
        )

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

  def run_ssacfg_region(
      region: Region,
      ctx: RuntimeCtx,
      name: String,
      inputs: Seq[Any],
  ): Seq[Any] =
    var results: Seq[Any] = Seq()

    if region.blocks.isEmpty then return Seq() // no blocks to run
    else
      val new_ctx = ctx.push_scope(name)
      set_values(region.blocks.head.arguments, inputs, new_ctx)
      for operation <- region.blocks.head.operations do
        val inputs = get_values(operation.operands, new_ctx)
        if operation.isInstanceOf[IsTerminator] then
          results = run_op(operation, new_ctx, inputs)
        interpret_op(operation, new_ctx)
      results

  def run_op(op: Operation, ctx: RuntimeCtx, inputs: Seq[Any]): Seq[Any] =
    val impl = impl_dict.get(op.getClass)
    impl match
      case Some(impl) =>
        impl.asInstanceOf[OpImpl[Operation]].compute(
          op,
          this,
          ctx,
          inputs,
        )
      case None =>
        throw new Exception(
          s"Unsupported operation when interpreting: ${op.getClass}"
        )

  def call_op(name: String, ctx: RuntimeCtx, inputs: Seq[Any]): Seq[Any] =
    val callee = symbolTable.get(name)
      .getOrElse(
        throw new Exception(s"Function $name not found")
      )
    set_values(callee.asInstanceOf[Operation].operands, inputs, ctx)
    run_ssacfg_region(callee.regions.head, ctx, name, inputs)

  def get_values(
      operands: Seq[Value[Attribute]],
      ctx: RuntimeCtx,
  ): Seq[Any] =
    operands.map(op => lookup_op(op, ctx))

  def set_values(
      results: Iterable[Value[Attribute]],
      values: Seq[Any],
      ctx: RuntimeCtx,
  ): Unit =
    for (res, value) <- results.zip(values) do ctx.scopedDict.update(res, value)

  // helper function to print values in interpreter
  // useful if booleans are represented as 0 and 1
  def interpreter_print(value: Any): Unit =
    value match
      case 0 => print("false\n")
      case 1 => print("true\n")
      case _ => print(s"$value\n")
