package scair.clair.test

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import scair.clair.macros.isAttributeOfType
import scair.dialects.builtin.*
import scair.ir.*

/** `isAttributeOfType` believes what the type system already established, and
  * only walks an array's elements when nothing else vouches for them.
  *
  * The two are told apart by handing it an array whose static type lies: a walk
  * would notice, and trusting the static type cannot. That is the intended
  * contract -- a cast is a promise the type system passes on -- so it is also
  * what makes the shortcut observable.
  */
class AttributeTypeTest extends AnyFlatSpec with Matchers:

  val lyingInts: ArrayAttribute[IntData] =
    ArrayAttribute[Attribute](StringData("a"))
      .asInstanceOf[ArrayAttribute[IntData]]

  val lyingAttrs: ArrayAttribute[IntegerAttr] =
    ArrayAttribute[Attribute](StringData("a"))
      .asInstanceOf[ArrayAttribute[IntegerAttr]]

  val erasedInts: Attribute = ArrayAttribute[Attribute](StringData("a"))

  "An element type nothing vouches for" should "be walked" in {
    isAttributeOfType[ArrayAttribute[IntData]](erasedInts) should be(false)
    isAttributeOfType[ArrayAttribute[IntData]](
      ArrayAttribute[Attribute](IntData(1)): Attribute
    ) should be(true)
  }

  "An element type the value's own carries" should "be taken as given" in {
    isAttributeOfType[ArrayAttribute[IntData]](lyingInts) should be(true)
  }

  it should "be taken as given through a subtype too" in {
    // IntegerAttr <: ParametrizedAttribute, so an ArrayAttribute[IntegerAttr]
    // is element-wise a ParametrizedAttribute array without looking.
    isAttributeOfType[ArrayAttribute[ParametrizedAttribute]](
      lyingAttrs
    ) should be(true)
    // Whereas the same contents, erased, have to be walked -- and a StringData
    // is no ParametrizedAttribute.
    isAttributeOfType[ArrayAttribute[ParametrizedAttribute]](
      erasedInts
    ) should be(false)
  }

  "An element type every attribute satisfies" should "not be walked" in {
    isAttributeOfType[ArrayAttribute[Attribute]](erasedInts) should be(true)
    isAttributeOfType[ArrayAttribute[Attribute]](
      IntData(1): Attribute
    ) should be(false)
  }

  "A plain attribute type" should "still be tested" in {
    isAttributeOfType[IntData](IntData(1): Attribute) should be(true)
    isAttributeOfType[IntData](StringData("a"): Attribute) should be(false)
  }

  "A union" should "be tested on each side" in {
    isAttributeOfType[IntData | StringData](StringData("a"): Attribute) should
      be(true)
    isAttributeOfType[IntData | StringData](
      FloatData(1.0): Attribute
    ) should be(false)
  }
