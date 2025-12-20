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

/*
sum (x in e1) e2(x),
where e2 is a function of x (x is of type record<key: T1, value: T2>)

becomes

%ev1 = $e1 : dictionary<T1, T2>
%res = sdql.sum %ev1 -> T2 {
^bb0(%x : record<"key": T1, "value": T2>):
    %ev2 = // compute e1 given x
    sdql.sum.yield %ev2
}
 */

case class Sum(
    arg: Operand[Attribute],
    // type of e1
    inType: Attribute,
    region: Region,
    result: Result[Attribute],
) extends DerivedOperation["sdql.sum", Sum]
    derives DerivedOperationCompanion

given OperationCustomParser[Sum]:

  def parse[$: P](
      resNames: Seq[String]
  )(using p: Parser): P[Sum] =
    (valueIdP ~ (":" ~ typeP) ~ ("->" ~ typeP) ~ regionP()).flatMap {
      case (attributeName, inType, resTypes, body) =>
        val attrTyp = body.blocks.head.arguments.apply(0).typ
        operandP(attributeName, inType).map(
            Sum(
            _,
            inType,
            body,
            Result(resTypes),
        )
        )
    }

case class LetIn(
    arg: Operand[Attribute],
    region: Region,
    result: Result[Attribute],
) extends DerivedOperation["sdql.let_in", LetIn]
    // with AssemblyFormat["$arg `->` type($result) `{` $region `}`"]
    derives DerivedOperationCompanion

case class Yield(
    arg: Operand[Attribute]
) extends DerivedOperation["sdql.yield", Yield]
    with AssemblyFormat["attr-dict $arg `:` type($arg)"]
    derives DerivedOperationCompanion

given OperationCustomParser[LetIn]:

  def parse[$: P](
      resNames: Seq[String]
  )(using p: Parser): P[LetIn] =
    (valueIdP ~ ("->" ~ typeP) ~ regionP()).flatMap {
      case (attributeName, resTypes, body) =>
        val attrTyp = body.blocks.head.arguments.apply(0).typ

        // TODO: is this a safe way of obtaining (arg: Operand[Attribute])?
        operandP(attributeName, attrTyp).map(LetIn(
            _,
            body,
            Result(resTypes),
        ))
    }

final case class EmptyDictionary(
    result: Result[DictionaryType]
) extends DerivedOperation["sdql.empty_dictionary", EmptyDictionary]
    with AssemblyFormat["attr-dict `:` type($result)"]
    derives DerivedOperationCompanion

final case class CreateDictionary(
    _operands: Seq[Operand[Attribute]],
    result: Result[DictionaryType],
) extends DerivedOperation["sdql.create_dictionary", CreateDictionary]
    with AssemblyFormat["attr-dict $_operands `:` type($_operands) `->` type($result)"]
    derives DerivedOperationCompanion

final case class LookupDictionary(
    dict: Operand[DictionaryType],
    key: Operand[Attribute],
    valueType: Result[Attribute]
) extends DerivedOperation["sdql.lookup_dictionary", LookupDictionary]
    with AssemblyFormat["$dict `[` $key `:` type($key) `]` attr-dict `:` type($dict) `->` type($valueType)"]
    derives DerivedOperationCompanion
 
final case class CreateRecord(
    values: Seq[Operand[Attribute]],
    result: Result[RecordType],
) extends DerivedOperation["sdql.create_record", CreateRecord]
    with AssemblyFormat["attr-dict ($values^ `:` type($values))? `->` type($result)"]
    derives DerivedOperationCompanion

final case class AccessRecord(
    record: Operand[RecordType],
    field: StringData,
    result: Result[Attribute],
) extends DerivedOperation["sdql.access_record", AccessRecord]
    with AssemblyFormat["attr-dict $record $field `:` type($record) `->` type($result)"]
    derives DerivedOperationCompanion

val SdqlDialect = summonDialect[EmptyTuple, (EmptyDictionary, CreateDictionary, LookupDictionary, CreateRecord, AccessRecord, LetIn, Yield, Sum)]
