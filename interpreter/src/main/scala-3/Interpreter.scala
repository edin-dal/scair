package scair.interpreter

import scair.dialects.builtin.*
import scair.dialects.func
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
        case r: Seq[Any] =>
          for (res, value) <- op.results.zip(r) do
            ctx.scopedDict.update(res, value)
        case _ =>
          ctx.scopedDict.update(op.results.head, result)

// interpreter context class stores variables, function definitions and the current result
class RuntimeCtx(
    val scopedDict: ScopedDict,
    var result: Option[Any] = None,
):

  // creates new runtime ctx with new scope but shared symbol table
  def push_scope(): RuntimeCtx =
    RuntimeCtx(
      ScopedDict(Some(this.scopedDict), mutable.Map()),
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
    val dialects: Seq[InterpreterDialect],
):

  initialize_interpreter()

  def initialize_interpreter(): Unit =
    register_implementations()
    get_symbols_from_module()

  def get_symbols_from_module(): Unit =
    for op <- module.body.blocks.head.operations do
      op match
        case func_op: func.Func =>
          // add function to symbol table if not main
          symbolTable.put(func_op.sym_name.stringLiteral, func_op)
        case _ => () // ignore other ops, global vars not yet supported prob...

  // lookup function for context variables
  // does not work for Bool-like vals due to inability to prove disjoint for TypeMap
  def lookup_op[T <: Value[Attribute]](value: T, ctx: RuntimeCtx): Any =
    ctx.scopedDict.get(value) match
      case Some(v) => v
      case _ => throw new Exception(s"Variable $value not found in context")

  def lookup_boollike(value: Value[Attribute], ctx: RuntimeCtx): Int =
    ctx.scopedDict.get(value) match
      case Some(v: Int) => v
      case _ => throw new Exception(s"Bool-like $value not found in context")

  def register_implementations(): Unit =
    for dialect <- dialects do
      for impl <- dialect do impl_dict.put(impl.opType, impl)

  // keeping buffer function for extensibility
  def interpret(block: Block, ctx: RuntimeCtx): Option[Any] =
    for op <- block.operations do interpret_op(op, ctx)
    ctx.result

  // note: results are put within implementations, may change later
  def interpret_op(op: Operation, ctx: RuntimeCtx): Unit =
    val impl = impl_dict.get(op.getClass)
    impl match
      case Some(impl) => impl.asInstanceOf[OpImpl[Operation]].run(op, this, ctx)
      case None       =>
        throw new Exception(s"Unsupported operation when interpreting: ${op.getClass}")

  def interpreter_print(value: Any): Unit =
    value match
      case 0 => println("false")
      case 1 => println("true")
      case _ => println(value)
