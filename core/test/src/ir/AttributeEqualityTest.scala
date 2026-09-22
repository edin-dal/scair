package scair

import org.scalatest.*
import org.scalatest.flatspec.*
import scair.clair.*
import scair.dialects.builtin.*
import scair.ir.*

// Two parametrized attributes with identical shapes, so that equality has to
// discriminate on the class and not just on the parameters.
case class PairAttr(fst: IntData, snd: IntData)
    extends DerivedAttribute["test.pair"] derives AttrDefs

case class TwinAttr(fst: IntData, snd: IntData)
    extends DerivedAttribute["test.twin"] derives AttrDefs

// A derived attribute with no parameters at all, so `equal` degenerates to the
// class check alone.
case class NilAttr() extends DerivedAttribute["test.nil"] derives AttrDefs

class AttributeEqualityTest extends AnyFlatSpec:

  /*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
  ||   DERIVED PARAMETRIZED ATTRIBUTES    ||
  \*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

  "A derived attribute" should
    "equal a distinct instance with equal parameters" in {
      val a = PairAttr(IntData(1), IntData(2))
      val b = PairAttr(IntData(1), IntData(2))
      assert(!(a eq b))
      assert(a == b)
      assert(b == a)
    }

  it should "equal itself" in {
    val a = PairAttr(IntData(1), IntData(2))
    assert(a == a)
  }

  it should "differ when any one parameter differs" in {
    val a = PairAttr(IntData(1), IntData(2))
    assert(a != PairAttr(IntData(9), IntData(2)))
    assert(a != PairAttr(IntData(1), IntData(9)))
    assert(a != PairAttr(IntData(9), IntData(9)))
  }

  it should "differ from another class with the same parameters" in assert(
    PairAttr(IntData(1), IntData(2)) != TwinAttr(IntData(1), IntData(2))
  )

  it should "differ from a non-attribute" in {
    // Spelled with `equals` rather than `!=`, which strict equality rejects
    // across unrelated types.
    assert(!PairAttr(IntData(1), IntData(2)).equals(42))
    assert(!PairAttr(IntData(1), IntData(2)).equals("test.pair"))
  }

  it should "agree with hashCode, so that hashing containers keep working" in {
    val a = PairAttr(IntData(1), IntData(2))
    val b = PairAttr(IntData(1), IntData(2))
    assert(a.hashCode == b.hashCode)
    assert(Set(a).contains(b))
    assert(Map(a -> "x").get(b).contains("x"))
  }

  "A parameterless derived attribute" should
    "equal any other instance of its class" in {
      assert(NilAttr() == NilAttr())
      assert(NilAttr().hashCode == NilAttr().hashCode)
      assert(NilAttr() != PairAttr(IntData(1), IntData(2)))
    }

  /*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
  ||   BUILTIN INTEGER TYPES     ||
  \*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

  "IntegerType" should "equal a freshly built one of the same width and sign" in {
    assert(
      IntegerType(IntData(32), Signless) == IntegerType(IntData(32), Signless)
    )
    assert(IntegerType(IntData(32), Signless) == I32)
    assert(IntegerType(IntData(1), Signless) == I1)
  }

  it should "differ on width or on signedness" in {
    assert(
      IntegerType(IntData(32), Signless) != IntegerType(IntData(64), Signless)
    )
    assert(
      IntegerType(IntData(32), Signless) != IntegerType(IntData(32), Signed)
    )
    assert(
      IntegerType(IntData(32), Signed) != IntegerType(IntData(32), Unsigned)
    )
  }

  /*≡==--==≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
  ||   DATA ATTRIBUTES      ||
  \*≡==---==≡≡≡≡≡≡≡≡==---==≡*/

  "A data attribute" should "equal a distinct instance holding equal data" in {
    val a = IntData(7)
    val b = IntData(7)
    assert(!(a eq b))
    assert(a == b)
    assert(a.hashCode == b.hashCode)
  }

  it should "differ when the data differs" in {
    assert(IntData(7) != IntData(8))
    assert(StringData("a") != StringData("b"))
    assert(FloatData(1.5) != FloatData(2.5))
  }

  it should "differ from another data attribute holding equal data" in
    // Both hold a BigInt-free `String`, so only the class tells them apart.
    assert(StringData("signless") != (Signless: Attribute))

  it should "still report its name now that it is no longer a stored field" in {
    assert(IntData(7).name == "builtin.int_attr")
    assert(FloatData(1.5).name == "builtin.float_data")
    assert(StringData("x").name == "builtin.string")
    assert(DictionaryAttr(Map.empty).name == "builtin.dict_attr")
    assert(Signless.name == "signless")
  }
