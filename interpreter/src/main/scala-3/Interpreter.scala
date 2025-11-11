package scair.tools

import scair.dialects.arith
import scair.dialects.builtin.*
import scair.dialects.func
import scair.dialects.memref
import scair.ir.*
import scala.collection.mutable
import scala.collection.mutable.ListBuffer

// interpreter context class stores variables, function definitions and the current result
class RuntimeCtx(
    val vars: mutable.Map[Value[Attribute], Any],
    val funcs: ListBuffer[FunctionCtx],
    var result: Option[Any] = None
):

  // creates independent context
  def deep_clone_ctx(): RuntimeCtx =
    RuntimeCtx(
      mutable.Map() ++ this.vars,
      ListBuffer() ++ this.funcs,
      None
    )

  // helper function for adding a function to the context
  def add_func_ctx(function: func.Func): Unit =
    val func_ctx = FunctionCtx(
      name = function.sym_name,
      saved_ctx = this.deep_clone_ctx(),
      body = function.body.blocks.head
    )
    this.funcs.append(func_ctx)

// function context class for saving a context at the function's definition (cannot use values defined after the function itself)
case class FunctionCtx(
    name: String,
    saved_ctx: RuntimeCtx,
    body: Block
)

// custom ShapedArray class used for memory and other multi-dimensional data structures
case class ShapedArray(
    private val data: Array[Any],
    shape: Seq[Int]
):

  lazy val strides: Seq[Int] =
    shape.scanRight(1)(_ * _).tail


  def length: Int =
    shape.product

  private def offset(indices: Seq[Int]): Int =
    indices.zip(strides).map(_ * _).sum

  def apply(indices: Seq[Int]): Any = data(offset(indices))

  def update(indices: Seq[Int], value: Any): Unit =
    data(offset(indices)) = value


// INTERPRETER CLASS
class Interpreter:

  // keeping buffer function for extensibility
  def interpret(block: Block, ctx: RuntimeCtx): Option[Any] =
    for op <- block.operations do interpret_op(op, ctx)
    ctx.result

  // main operation interpretation function
  def interpret_op(op: Operation, ctx: RuntimeCtx): Unit =
    op match
      case func_op: func.Func =>
        interpret_function(func_op, ctx)
        val interpreterCtx = scair.tools.interpreterContext

      // Return Operation
      case return_op: func.Return =>
        //run_return(return_op, ctx)

      // Constant Operation
      case constant: arith.Constant => None
        //run_constant(constant, ctx)

      // Binary Operations
      case addI_op: arith.AddI =>
        //run_addi(addI_op, ctx)
      case subI_op: arith.SubI =>
        //run_subi(subI_op, ctx)
      case mulI_op: arith.MulI =>
        //run_muli(mulI_op, ctx)
      case div_op: arith.DivSI =>
        //run_divsi(div_op, ctx)
      case div_op: arith.DivUI =>
        //run_divui(div_op, ctx)
      case andI_op: arith.AndI =>
        //run_andi(andI_op, ctx)
      case orI_op: arith.OrI =>
        //run_ori(orI_op, ctx)
      case xorI_op: arith.XOrI =>
        //run_xori(xorI_op, ctx)

      // Shift Operations
      case shli_op: arith.ShLI =>
        //run_shli(shli_op, ctx)
      case shrsi_op: arith.ShRSI =>
        //run_shrsi( shrsi_op, ctx)
      case shrui_op: arith.ShRUI =>
        //run_shrui(shrui_op, ctx)

      // Comparison Operation
      case cmpi_op: arith.CmpI =>
        //run_cmpi(cmpi_op, ctx)

      // Select Operation
      case select_op: arith.SelectOp =>
        //run_select(select_op, ctx)

      // Memory operations
      case alloc_op: memref.Alloc =>
        //run_alloc(alloc_op, ctx)
      case load_op: memref.Load =>
        //run_load(load_op, ctx)
      case store_op: memref.Store =>
        //run_store(store_op, ctx)
      case _ => throw new Exception("Unsupported operation when interpreting")

  def interpret_block_or_op(
      value: Operation | Block,
      ctx: RuntimeCtx
  ): Unit =
    value match
      case op: Operation => interpret_op(op, ctx)
      case block: Block  =>
        for op <- block.operations do interpret_op(op, ctx)

  def interpret_function(function: func.Func, ctx: RuntimeCtx): Unit =
    // if main, interpret it immediately by creating a call operation and evaluating it
    if function.sym_name.stringLiteral == "main" then
      val main_ctx = FunctionCtx(
        name = function.sym_name,
        body = function.body.blocks.head,
        saved_ctx = ctx
      )
      ctx.funcs.append(main_ctx)
      val new_call = func.Call(
        callee = SymbolRefAttr(function.sym_name),
        _operands = Seq(),
        _results = function.function_type.outputs.map(res => Result(res))
      )
      interpret_call(new_call, ctx)
    else
      // function definition; no call, add function and current running context FunctionCtx to interpreter context
      ctx.add_func_ctx(function)

  def interpret_call(call_op: func.Call, ctx: RuntimeCtx): Unit =
    for func_ctx <- ctx.funcs do
      if func_ctx.name == call_op.callee.rootRef.stringLiteral then
        interpret_block_or_op(func_ctx.body, func_ctx.saved_ctx)
