package scair.dialects.cf

import scair.clair.*
import scair.dialects.builtin.*
import scair.ir.*

//
//  ░█████╗░ ███████╗
//  ██╔══██╗ ██╔════╝
//  ██║░░╚═╝ █████╗░░
//  ██║░░██╗ ██╔══╝░░
//  ╚█████╔╝ ██║░░░░░
//  ░╚════╝░ ╚═╝░░░░░
//

// TODO: this needs to be constrained specifically to an I1 integer
type I1 = IntegerType

case class Assert(
    arg: Operand[I1],
    msg: StringData,
) extends DerivedOperation["cf.assert"] derives OpDefs

case class Branch(
    destOperands: Seq[Operand[Attribute]] = Seq.empty,
    dest: Successor,
) extends DerivedOperation["cf.br"]
    with Pure
    with IsTerminator derives OpDefs

case class CondBranch(
    condition: Operand[I1],
    trueDestOperands: Seq[Operand[Attribute]] = Seq.empty,
    falseDestOperands: Seq[Operand[Attribute]] = Seq.empty,
    // TODO: DenseI32ArrayAttr upstream
    branch_weights: Option[DenseArrayAttr] = None,
    trueDest: Successor,
    falseDest: Successor,
) extends DerivedOperation["cf.cond_br"]
    with Pure
    with IsTerminator derives OpDefs

// `caseOperands` is a VariadicOfVariadic upstream, flattened here;
// `case_operand_segments` gives the grouping.
case class Switch(
    flag: Operand[IntegerType],
    defaultOperands: Seq[Operand[Attribute]] = Seq.empty,
    caseOperands: Seq[Operand[Attribute]] = Seq.empty,
    case_values: Option[DenseIntElementsAttr] = None,
    // TODO: DenseI32ArrayAttr upstream
    case_operand_segments: DenseArrayAttr,
    defaultDestination: Successor,
    caseDestinations: Seq[Successor] = Seq.empty,
) extends DerivedOperation["cf.switch"]
    with Pure
    with IsTerminator derives OpDefs

val CfDialect = summonDialect[EmptyTuple, (Assert, Branch, CondBranch, Switch)]
