package scair

/** Clair is a Scala library for creating custom IRs. 
  * It provides the end-user with a way to define custom IRs packaged in a Dialect (MLIR word for DSL)
  * using Scala syntax and then generate the IR code in a target language (currently limited to Scala).
  * 
  * To define a custom Dialect, the user needs to define an IR 
  * and package it into a Dialect object as shown in the code below with the Sample dialect example:
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
  * case class SampleAttribute(
  *     e1: Operand[IntegerAttr]
  * ) extends AttributeFE
  * 
  * /*≡≡=---=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=---=≡≡*\
  * ||   defining a custom type attribute   ||
  * \*≡==----=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=----==≡*/
  * 
  * case class SampleType(
  *     e1: Operand[IntegerAttr]
  * ) extends TypeAttributeFE
  * 
  * /*≡≡=---=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=---=≡≡*\
  * ||   defining custom operations   ||
  * \*≡==----=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=----==≡*/
  * 
  * case class SampOp1(
  *     e1: Variadic[Operand[IntegerAttr]],
  *     e2: Result[AnyAttribute],
  *     e3: Region
  * ) extends OperationFE
  * 
  * case class SampOp2(
  *     e1: Variadic[Operand[Complex]],
  *     e2: Result[AnyAttribute]
  * ) extends OperationFE
  * 
  * /*≡≡=---=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=---=≡≡*\
  * ||   packaging into a dialect   ||
  * \*≡==----=≡≡≡≡≡≡≡≡≡≡≡≡≡≡=----==≡*/
  * 
  * object Sample {
  *   val opHatches = Seq()
  *   val attrHatches = Seq(new AttrEscapeHatch[SampleData])
  *   val generator =
  *     summonDialect[(SampleAttribute, SampOp1, SampOp2)]("Sample", opHatches, attrHatches)
  * }
  * 
  * /*≡≡=---=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=---=≡≡*\
  * ||   print the generated code   ||
  * \*≡==----≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=----==≡*/
  * 
  * def main(args: Array[String]): Unit = {
  *   println(CMath.generator.print(0))
  * }
  * ```
  * 
  * To include the defined Dialect in ScaIR, the user should
  * put the file into the dialects directory in the scair project, and package it appropriately.
  * @see [[scair.dialects.cmathgen]]
  *
  */
package object clair {}
