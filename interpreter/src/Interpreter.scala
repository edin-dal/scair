package scair.interpreter

import scair.dialects.builtin.*
import scair.ir.*

import scala.collection.mutable
import scala.reflect.ClassTag

// global implementation dictionary for interpreter
type RegisteredImpl =
  (Interpreter, RuntimeCtx, Operation, Seq[Any]) => OpImplResult

val impl_dict = mutable.Map[Class[? <: Operation], RegisteredImpl]()

// control-flow step produced by a terminator implementation
enum CFGStep:
  case Return(values: Seq[Any])
  case Jump(dest: Block, args: Seq[Any])

// result of running an operation: produced values plus optional control-flow step
case class OpImplResult(
    values: Seq[Any],
    step: Option[CFGStep] = None,
)

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

// terminator operations implement this trait instead of OpImpl
// computeTerminator returns the control-flow step to take, plus any values the operation produces
trait OpTerminatorImpl[O <: Operation: ClassTag]:

  def opType: Class[O] = summon[ClassTag[O]].runtimeClass.asInstanceOf[Class[O]]

  def computeTerminator(
      op: O,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Seq[Any],
  ): (CFGStep, Seq[Any])

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
      for impl <- dialect do
        impl match
          case opImpl: OpImpl[? <: Operation] =>
            impl_dict.put(
              opImpl.opType,
              (interp, ctx, op, args) =>
                OpImplResult(
                  opImpl
                    .asInstanceOf[OpImpl[Operation]]
                    .compute(op, interp, ctx, args)
                ),
            )
          case termImpl: OpTerminatorImpl[? <: Operation] =>
            impl_dict.put(
              termImpl.opType,
              (interp, ctx, op, args) =>
                val (step, values) =
                  termImpl
                    .asInstanceOf[OpTerminatorImpl[Operation]]
                    .computeTerminator(op, interp, ctx, args)
                OpImplResult(values, Some(step)),
            )

  def create_scope(name: String): RuntimeCtx =
    RuntimeCtx(ScopedDict(None, mutable.Map(), name), Seq())

  def run_ssacfg_region(
      region: Region,
      ctx: RuntimeCtx,
      name: String,
      inputs: Seq[Any],
  ): Seq[Any] =
    if region.blocks.isEmpty then return Seq()

    var current: Option[Block] = Some(region.blocks.head)
    var blockArgs = inputs
    var blockCtx = ctx.push_scope(name)

    while current.isDefined do
      val block = current.get
      current = None
      bind_block_args(block, blockArgs, blockCtx)

      var jump: Option[(Block, Seq[Any])] = None
      val ops = block.operations.toSeq
      var i = 0
      while i < ops.length && jump.isEmpty do
        val op = ops(i)
        val res = run_op(op, blockCtx, get_values(op.operands, blockCtx))
        set_values(op.results, res.values, blockCtx)
        res.step match
          case Some(CFGStep.Return(values)) => return values
          case Some(CFGStep.Jump(dest, args)) =>
            jump = Some((dest, args))
          case None => ()
        i += 1

      jump match
        case Some((dest, args)) =>
          current = Some(dest)
          blockArgs = args
          blockCtx = blockCtx.push_scope(name)
        case None => ()
    Seq()

  private def bind_block_args(
      block: Block,
      args: Seq[Any],
      ctx: RuntimeCtx,
  ): Unit =
    if block.arguments.size != args.size then
      throw new Exception(
        s"Expected ${block.arguments.size} block arguments, got ${args.size}"
      )
    set_values(block.arguments, args, ctx)

  def run_op(op: Operation, ctx: RuntimeCtx, inputs: Seq[Any]): OpImplResult =
    val impl = impl_dict.get(op.getClass)
    impl match
      case Some(impl) =>
        val res = impl(this, ctx, op, inputs)
        if op.results.size != res.values.size then
          throw new Exception(
            s"Operation '${op.name}' produced ${res.values.size} values but declares ${op.results.size} results"
          )
        res
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
