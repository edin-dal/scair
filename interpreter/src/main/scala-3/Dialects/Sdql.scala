package scair.interpreter

import scair.dialects.sdql
import scair.dialects.irdl.Attribute
import scair.dialects.builtin.DictionaryType
import scair.dialects.builtin.IntegerType
import scair.ir.TypeAttribute
import scair.ir.IntegerEnumAttr
import scair.ir.ParametrizedAttribute
import scair.ir.DataAttribute
import scair.ir.AliasedAttribute
import scair.dialects.builtin.FloatType
import scair.dialects.sdql.CreateDictionary

object run_empty_dictionary extends OpImpl[sdql.EmptyDictionary]:

  def run(op: sdql.EmptyDictionary, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
    op.result.typ match
        case DictionaryType(_: IntegerType, _: IntegerType) =>
            ctx.vars.put(op.result, Map.empty[Int, Int])
        case _ =>  throw new Exception("Unsupported dict type")

object run_sum extends OpImpl[sdql.Sum]:

  def run(op: sdql.Sum, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
    var acc = op.result.typ match
        // case _: DictionaryType => 
        case _: IntegerType => 0
        // case _: FloatType => 0.0
        case _ => throw new Exception("Unsupported sum result type: " + op.result.typ)
    val arg = interpreter.lookup_op(op.arg, ctx).asInstanceOf[Map[Int, Int]]
    val block = op.region.blocks(0)
    arg.foreach((k, v) => {
        // setup args
        ctx.vars.put(block.arguments(0), k)
        ctx.vars.put(block.arguments(1), v)
        val res = interpreter.interpret(block, ctx)
        acc += res.get.asInstanceOf[Int]
    })
    ctx.vars.put(op.result, acc)

object run_yield extends OpImpl[sdql.Yield]:

    def run(op: sdql.Yield, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
        ctx.result = Some(interpreter.lookup_op(op.arg, ctx))

object run_create_dictionary extends OpImpl[sdql.CreateDictionary]:
    def mapFromFlat(items: Seq[Int]): Map[Int, Int] =
        items.grouped(2).map { case Seq(k, v) => k -> v }.toMap

    def run(op: sdql.CreateDictionary, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
        val pairs = op._operands.map(operand => interpreter.lookup_op(operand, ctx)).asInstanceOf[Seq[Int]]
        val res = mapFromFlat(pairs)
        ctx.vars.put(op.result, res)

val InterpreterSdqlDialect: InterpreterDialect =
    Seq(run_empty_dictionary, run_sum, run_yield, run_create_dictionary)
