package scair.dialects.samplecnstr

import scair.clair.codegen.*
import scair.clair.macros.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.core.constraints.{_, given}

type T = Var["T"]
val i32 = IntegerType(IntData(32), Signless)

case class MulIEq(
    lhs: Operand[IntegerType !> EqAttr[i32.type]],
    rhs: Operand[IntegerType !> EqAttr[i32.type]],
    result: Result[IntegerType]
) extends DerivedOperation["samplecnstr.mulieq", MulIEq]

case class MulIVar(
    lhs: Operand[IntegerType !> T],
    rhs: Operand[IntegerType !> T],
    result: Result[IntegerType]
) extends DerivedOperation["samplecnstr.mulivar", MulIVar]

val samplecnstr = summonDialect[EmptyTuple, MulIEq *: MulIVar *: EmptyTuple]()
