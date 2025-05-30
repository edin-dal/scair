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
  * import scair.ir.{DataAttribute, AttributeObject}
  *
  * /*≡≡=---=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=---=≡≡*\
  * ||   defining a custom data attribute   ||
  * \*≡==----=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=----==≡*/
  *
  * object SampleData extends AttributeObject {
  *   override def name: String = "sample"
  * }
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
  * ) extends ParametrizedAttribute(
  *       name = "sample.sample_attr",
  *       parameters = Seq(value)
  *     )
  *     with MLIRName["sample.sample_attr"]
  *     derives AttributeTrait
  *
  * /*≡≡=---=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=---=≡≡*\
  * ||   defining a custom type attribute   ||
  * \*≡==----=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=----==≡*/
  *
  * case class SampleType(
  *     val value: FloatType
  * ) extends ParametrizedAttribute(
  *       name = "sample.sample_type",
  *       parameters = Seq(value)
  *     )
  *     with TypeAttribute
  *     with MLIRName["sample.sample_type"]
  *     derives AttributeTrait
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
  * /*≡≡=---=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=---=≡≡*\
  * ||   packaging into a dialect   ||
  * \*≡==----=≡≡≡≡≡≡≡≡≡≡≡≡≡≡=----==≡*/
  *
  * val Sample = summonDialect[(SampleAttr, SampleType), (SampOp1, SampOp1)](
  *   Seq(SampleData)
  * )
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
