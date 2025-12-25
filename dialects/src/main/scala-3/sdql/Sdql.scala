package scair.dialects.sdql

import fastparse.*
import scair.*
import scair.Printer
import scair.clair.codegen.*
import scair.clair.macros.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.parse.*
import scair.parse.Parser
import scair.utils.OK

/*
sum (x in e1) e2(x),
where e2 is a function of x (x is of type record<key: T1, value: T2>)

becomes

%ev1 = $e1 : dictionary<T1, T2>
%res = sdql.sum %ev1 -> T2 {
^bb0(%x : record<"key": T1, "value": T2>):
    %ev2 = // compute e2 given x
    sdql.sum.yield %ev2
}
 */

case class Sum(
    arg: Operand[Attribute],
    // type of e1
    inType: DictionaryType,
    region: Region,
    result: Result[Attribute],
) extends DerivedOperation["sdql.sum", Sum] derives DerivedOperationCompanion:

  override def customVerify(): OK[Operation] =
    // block has 2 arguments, matching key and value type of inType
    if region.blocks.size != 1 then
      return Left(s"Expected one region in sum, got ${region.blocks.size}")

    val block = region.blocks(0)
    if block.arguments.size != 2 then
      return Left(
        s"Expected sum region to have two arguments, got ${block.arguments.size}"
      )

    val (type0, type1) = (block.arguments(0).typ, block.arguments(1).typ)

    if type0 != inType.keyType || type1 != inType.valueType then
      return Left(s"Expected block args to be (${inType.keyType}, ${inType
          .valueType}), got ($type0, $type1)")

    Right(this)

given OperationCustomParser[Sum]:

  def parse[$: P](resNames: Seq[String])(using p: Parser): P[Sum] =
    (valueIdP ~ (":" ~ typeP) ~ ("->" ~ typeP) ~ regionP()).flatMap {
      case (argName, inType, resType, region) =>
        // declare the resulting val
        (operandP(argName, inType) ~ resultP(resNames.head, resType)).map {
          case (arg, res) =>
            Sum(arg, inType.asInstanceOf[DictionaryType], region, res)
        }
    }

case class Yield(
    arg: Operand[Attribute]
) extends DerivedOperation["sdql.yield", Yield]
    with AssemblyFormat["attr-dict $arg `:` type($arg)"]
    derives DerivedOperationCompanion

final case class EmptyDictionary(
    result: Result[DictionaryType]
) extends DerivedOperation["sdql.empty_dictionary", EmptyDictionary]
    with AssemblyFormat["attr-dict `:` type($result)"]
    derives DerivedOperationCompanion

final case class CreateDictionary(
    _operands: Seq[Operand[Attribute]],
    result: Result[Attribute],
) extends DerivedOperation["sdql.create_dictionary", CreateDictionary]
    with AssemblyFormat[
      "attr-dict $_operands `:` type($_operands) `->` type($result)"
    ] derives DerivedOperationCompanion:

  override def customVerify(): OK[Operation] =
    if !result.typ.isInstanceOf[DictionaryType] then
      return Left("Return type of create_dictionary must be a dictionary")

    // TODO: verify that _operands is exactly [keyTyp, valTyp, keyTyp, ...]
    Right(this)

final case class LookupDictionary(
    dict: Operand[DictionaryType],
    key: Operand[Attribute],
    valueType: Result[Attribute],
) extends DerivedOperation["sdql.lookup_dictionary", LookupDictionary]
    with AssemblyFormat[
      "$dict `[` $key `:` type($key) `]` attr-dict `:` type($dict) `->` type($valueType)"
    ] derives DerivedOperationCompanion

final case class CreateRecord(
    values: Seq[Operand[Attribute]],
    result: Result[RecordType],
) extends DerivedOperation["sdql.create_record", CreateRecord]
    with AssemblyFormat[
      "attr-dict ($values^ `:` type($values))? `->` type($result)"
    ] derives DerivedOperationCompanion

final case class AccessRecord(
    record: Operand[RecordType],
    field: StringData,
    result: Result[Attribute],
) extends DerivedOperation["sdql.access_record", AccessRecord]
    with AssemblyFormat[
      "attr-dict $record $field `:` type($record) `->` type($result)"
    ] derives DerivedOperationCompanion

final case class DictionaryAdd(
    lhs: Operand[DictionaryType],
    rhs: Operand[DictionaryType],
    result: Result[Attribute],
) extends DerivedOperation["sdql.dictionary_add", DictionaryAdd]
    with AssemblyFormat[
      "attr-dict $lhs $rhs `:` type($lhs)`,` type($rhs) `->` type($result)"
    ] derives DerivedOperationCompanion

final case class Concat(
    lhs: Operand[RecordType],
    rhs: Operand[RecordType],
    result: Result[RecordType],
) extends DerivedOperation["sdql.concat", Concat]
    with AssemblyFormat[
      "attr-dict $lhs `,` $rhs `:` type($lhs)`,` type($rhs) `->` type($result)"
    ] derives DerivedOperationCompanion

final case class External(
    _name: StringData,
    args: Seq[Operand[Attribute]],
    result: Result[Attribute],
) extends DerivedOperation["sdql.external", External]
    with AssemblyFormat[
      "attr-dict $_name `,` ($args^ `:` type($args))? `->` type($result)"
    ] derives DerivedOperationCompanion

final case class Cmp(
    lhs: Operand[Attribute],
    rhs: Operand[Attribute],
    result: Result[Attribute],
) extends DerivedOperation["sdql.cmp", Cmp]
    with AssemblyFormat[
      "attr-dict $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `->` type($result)"
    ] derives DerivedOperationCompanion

final case class Load(
    path: StringData,
    typ: Result[Attribute],
) extends DerivedOperation["sdql.load", Load]
    with AssemblyFormat["attr-dict $path `:` type($typ)"]
    derives DerivedOperationCompanion

final case class Unique(
    arg: Operand[Attribute],
    typ: Result[Attribute],
) extends DerivedOperation["sdql.unique", Unique]
    with AssemblyFormat["attr-dict $arg `:` type($arg) `->` type($typ)"]
    derives DerivedOperationCompanion

val SdqlDialect = summonDialect[
  EmptyTuple,
  (
      EmptyDictionary,
      CreateDictionary,
      LookupDictionary,
      CreateRecord,
      AccessRecord,
      Yield,
      Sum,
      DictionaryAdd,
      Concat,
      External,
      Cmp,
      Load,
      Unique,
  ),
]
