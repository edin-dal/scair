package scair

/** Clair is a Scala library for creating custom IRs. It provides the end-user
  * with a way to define custom IRs packaged in a Dialect (MLIR word for DSL)
  * using Scala syntax and then generate the IR code in a target language
  * (currently limited to Scala).
  *
  * To define a custom Dialect, the user needs to define an IR and package it
  * into a Dialect object as shown in the code below with the Sample dialect
  * example:
  *
  * ```scala
  * import scair.ir.*
  * import scair.clair.macros.*
  * import scair.dialects.builtin.*
  * import scair.dialects.cmath.*
  * import scair.enums.I32Enum
  * import fastparse.*
  * import scair.parse.*
  *
  * /*≡≡=---=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=---=≡≡*\
  * ||   defining a custom I32 enum attribute   ||
  * \*≡==----=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=----==≡*/
  *
  * enum Color(name: String) extends I32Enum(name):
  *   case Red extends Color("red")
  *   case Green extends Color("green")
  *   case Blue extends Color("blue")
  *
  * case class EnumOperation(
  *     val color: Color
  * ) extends DerivedOperation["arith.enum_op"] derives OpDefs
  *
  * /*≡≡=---=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=---=≡≡*\
  * ||   defining a custom data attribute   ||
  * \*≡==----=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=----==≡*/
  *
  * given AttributeCompanion[SampleData]:
  *   override def name: String = "sample"
  *
  *   override def parse[$: P](using Parser) =
  *     "#sample<" ~ CharsWhile(_ != '>').!.map(SampleData.apply) ~ ">"
  *
  * case class SampleData(val d: String)
  *     extends DataAttribute[String]("sample", d)
  *
  * /*≡≡=---=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=---=≡≡*\
  * ||   defining a custom attribute   ||
  * \*≡==----=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=----==≡*/
  *
  * case class SampleAttr(
  *     val value: FloatType
  * ) extends DerivedAttribute["sample.sample_attr"] derives AttrDefs
  *
  * /*≡≡=---=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=---=≡≡*\
  * ||   defining a custom type attribute   ||
  * \*≡==----=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=----==≡*/
  *
  * case class SampleType(
  *     val value: FloatType
  * ) extends DerivedAttribute["sample.sample_type"]
  *     with TypeAttribute derives AttrDefs
  *
  * /*≡≡=---=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=---=≡≡*\
  * ||   defining custom operations   ||
  * \*≡==----=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=----==≡*/
  *
  * case class SampOp1(
  *     e1: Seq[Operand[IntegerAttr]],
  *     e2: Result[Attribute],
  *     e3: Region,
  * ) extends DerivedOperation["sample.sampop1"] derives OpDefs
  *
  * case class SampOp2(
  *     e1: Seq[Operand[Complex]],
  *     e2: Result[Attribute],
  * ) extends DerivedOperation["sample.sampop2"] derives OpDefs
  *
  * /*≡≡=---=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=---=≡≡*\
  * ||   constraints over operation components   ||
  * \*≡==----=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=----==≡*/
  *
  * import scair.constraints.*
  *
  * type T = Var["T"]
  * val i32 = IntegerType(IntData(32), Signless)
  *
  * // `EqAttr` pins an operand to one exact attribute.
  * case class MulIEq(
  *     lhs: Operand[IntegerType !> EqAttr[i32.type]],
  *     rhs: Operand[IntegerType !> EqAttr[i32.type]],
  *     result: Result[IntegerType],
  * ) extends DerivedOperation["samplecnstr.mulieq"] derives OpDefs
  *
  * // A `Var` ties components together: whichever type `lhs` has, `rhs` and
  * // `result` must have the same. This is MLIR's SameOperandsAndResultType,
  * // and it costs nothing at run time -- the generated check is a direct
  * // comparison of the two fields, with no variable to look up anywhere.
  * case class MulIVar(
  *     lhs: Operand[IntegerType !> T],
  *     rhs: Operand[IntegerType !> T],
  *     result: Result[IntegerType !> T],
  * ) extends DerivedOperation["samplecnstr.mulivar"] derives OpDefs
  *
  * // Constraints compose. Name the combinations you use often; a type alias
  * // is all a reusable constraint needs to be.
  * type AnyFloat = Base[Float16Type] || Base[Float32Type] || Base[Float64Type]
  * type SignlessInt = Param[IntegerType, (AnyAttr, EqAttr[Signless.type])]
  *
  * case class AddF(
  *     lhs: Operand[Attribute !> (AnyFloat && T)],
  *     rhs: Operand[Attribute !> T],
  *     result: Result[Attribute !> Msg["result must match the operands", T]],
  * ) extends DerivedOperation["samplecnstr.addf"] derives OpDefs
  *
  * // Because `result`'s type is determined by `T`, an assembly format may
  * // leave it out entirely; it is inferred when parsing.
  * case class NegF(
  *     operand: Operand[Attribute !> (AnyFloat && T)],
  *     result: Result[Attribute !> T],
  * ) extends DerivedOperation["samplecnstr.negf"]
  *     with AssemblyFormat["$operand attr-dict `:` type($operand)"]
  *     derives OpDefs
  *
  * /*≡≡=---=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=---=≡≡*\
  * ||   packaging into a dialect   ||
  * \*≡==----=≡≡≡≡≡≡≡≡≡≡≡≡≡≡=----==≡*/
  *
  * val Sample =
  *   summonDialect[(SampleAttr, SampleType, SampleData), (SampOp1, SampOp1)]
  * ```
  *
  * To include the defined Dialect in ScaIR, the user should put the file into
  * the dialects directory in the scair project, and package it appropriately.
  * @see
  *   [[scair.dialects.cmathgen]]
  */
package object clair {
  // Package-level definitions and utilities can go here
}

package clair {

  /** This package contains the mirrored logic for the front-end of the Clair
    * library.
    */
  package object mirrored {}
}
