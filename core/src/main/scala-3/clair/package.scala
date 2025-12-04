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
  * import scair.enums.enumattr.I32Enum
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
  * ) extends DerivedOperation["arith.enum_op", EnumOperation]
  *     derives DerivedOperationCompanion
  *
  * /*≡≡=---=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=---=≡≡*\
  * ||   defining a custom data attribute   ||
  * \*≡==----=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=----==≡*/
  *
  * object SampleData extends AttributeCompanion:
  *   override def name: String = "sample"
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
  * ) extends DerivedAttribute["sample.sample_attr", SampleAttr]
  *     derives DerivedAttributeCompanion
  *
  * /*≡≡=---=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=---=≡≡*\
  * ||   defining a custom type attribute   ||
  * \*≡==----=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=----==≡*/
  *
  * case class SampleType(
  *     val value: FloatType
  * ) extends DerivedAttribute["sample.sample_type", SampleType]
  *     with TypeAttribute derives DerivedAttributeCompanion
  *
  * /*≡≡=---=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=---=≡≡*\
  * ||   defining custom operations   ||
  * \*≡==----=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=----==≡*/
  *
  * case class SampOp1(
  *     e1: Seq[Operand[IntegerAttr]],
  *     e2: Result[Attribute],
  *     e3: Region
  * ) extends DerivedOperation["sample.sampop1", SampOp1]
  *     derives DerivedOperationCompanion
  *
  * case class SampOp2(
  *     e1: Seq[Operand[Complex]],
  *     e2: Result[Attribute]
  * ) extends DerivedOperation["sample.sampop2", SampOp2]
  *     derives DerivedOperationCompanion
  *
  * /*≡≡=---=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=---=≡≡*\
  * ||   constraints over operation components   ||
  * \*≡==----=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=----==≡*/
  *
  * import scair.core.constraints.{*, given}
  *
  * type T = Var["T"]
  * val i32 = IntegerType(IntData(32), Signless)
  *
  * case class MulIEq(
  *     lhs: Operand[IntegerType !> EqAttr[i32.type]],
  *     rhs: Operand[IntegerType !> EqAttr[i32.type]],
  *     result: Result[IntegerType]
  * ) extends DerivedOperation["samplecnstr.mulieq", MulIEq]
  *     derives DerivedOperationCompanion
  *
  * case class MulIVar(
  *     lhs: Operand[IntegerType !> T],
  *     rhs: Operand[IntegerType !> T],
  *     result: Result[IntegerType]
  * ) extends DerivedOperation["samplecnstr.mulivar", MulIVar]
  *     derives DerivedOperationCompanion
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
