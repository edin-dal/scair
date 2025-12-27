package scair.interpreter

import scair.dialects.sdql
import scair.dialects.builtin.DictionaryType
import scair.dialects.builtin.IntegerType
import scair.dialects.sdql.CreateDictionary
import scair.dialects.sdql.CreateRecord
import scair.dialects.sdql.AccessRecord

trait Add {
  type A
  def zero: A
  def plus(a: A, b: A): A
  def cast(x: Any): A
}

object IntAdd extends Add {
  type A = Int

  def zero: Int = 0

  def plus(a: Int, b: Int): Int = a + b

  def cast(a: Any): Int = a.asInstanceOf[Int]
}

object MapAdd extends Add {
  type A = Map[Int, Int]

  def zero: Map[Int, Int] = Map.empty

  def plus(a: Map[Int, Int], b: Map[Int, Int]): Map[Int, Int] =
    (a.keySet ++ b.keySet).map { k =>
      k -> (a.getOrElse(k, 0) + b.getOrElse(k, 0))
    }.toMap

  def cast(a: Any): Map[Int, Int] = a.asInstanceOf[Map[Int, Int]]
}

object run_empty_dictionary extends OpImpl[sdql.EmptyDictionary]:

  def run(op: sdql.EmptyDictionary, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
    op.result.typ match
        case DictionaryType(_: IntegerType, _: IntegerType) =>
            ctx.vars.put(op.result, Map.empty[Int, Int])
        case _ =>  throw new Exception("Unsupported dict type")

object run_sum extends OpImpl[sdql.Sum]:

  def run(op: sdql.Sum, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
    val agg = op.result.typ match
        // TODO handle different width
        case _: IntegerType => IntAdd
        case dt: DictionaryType if dt.keyType.isInstanceOf[IntegerType] && dt.valueType.isInstanceOf[IntegerType] =>
            MapAdd
        case _ => throw new Exception("Unsupported sum result type: " + op.result.typ.asInstanceOf[DictionaryType].valueType.isInstanceOf[IntegerType])
    var acc = agg.zero
    val arg = interpreter.lookup_op(op.arg, ctx).asInstanceOf[Map[Int, Int]]
    val block = op.region.blocks(0)
    arg.foreach((k, v) => {
        // setup args
        ctx.vars.put(block.arguments(0), k)
        ctx.vars.put(block.arguments(1), v)
        val res = interpreter.interpret(block, ctx)
        // do we need to restore ctx in case of shadowing? See OpImpl[func.Call]
        acc = agg.plus(acc, agg.cast(res.get))
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

object run_lookup_dictionary extends OpImpl[sdql.LookupDictionary]:
    def run(op: sdql.LookupDictionary, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
        val dict = interpreter.lookup_op(op.dict, ctx).asInstanceOf[Map[Int, Int]]
        val key = interpreter.lookup_op(op.key, ctx).asInstanceOf[Int]
        ctx.vars.put(op.valueType, dict(key))

object run_create_record extends OpImpl[sdql.CreateRecord]:
    def run(op: sdql.CreateRecord, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
        // for now Seq of (String, Int)
        val fields = op.result.typ.entries.map(_._1).map(_.stringLiteral)
        val computedChildren = op.values.map(operand => interpreter.lookup_op(operand, ctx)).asInstanceOf[Seq[Int]]
        ctx.vars.put(op.result, fields.zip(computedChildren))

object run_access_record extends OpImpl[sdql.AccessRecord]:
    def run(op: AccessRecord, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
        val recordObj = interpreter.lookup_op(op.record, ctx).asInstanceOf[Seq[(String, Int)]]
        val keyObj = op.field.stringLiteral

        val foundEntry = recordObj.find(_._1 == keyObj)
        ctx.vars.put(op.result, foundEntry.get._2)


val InterpreterSdqlDialect: InterpreterDialect =
    Seq(run_empty_dictionary, run_sum, run_yield, run_create_dictionary, run_lookup_dictionary, run_create_record, run_access_record)
