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
import scair.dialects.complex.Create
import java.util.Dictionary
import java.lang.invoke.MethodHandles.Lookup

/*
/*
sum (x in d) e
becomes


 */

case class SumOp(
    arg: Operand[Attribute],
    region: Region,
    result: Result[Attribute],
) extends DerivedOperation["sdql.sum", SumOp]
    derives DerivedOperationCompanion


/* 
let x = op1(a, b) in op2(x, c)

becomes

%rhs = op1 %a, %b : !dictionary

%res = sdql.let %rhs -> !dictionary {
^bb0(%x: !dictionary):
  %out = op2 %x, %c : !dictionary
  sdql.yield %out : !dictionary
}
 */
case class LetInOp(
    arg: Operand[Attribute],
    region: Region,
    result: Result[Attribute],
) extends DerivedOperation["sdql.letin", LetInOp]
    derives DerivedOperationCompanion
*/
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

val SdqlDialect = summonDialect[EmptyTuple, (EmptyDictionary, CreateDictionary, LookupDictionary)]
