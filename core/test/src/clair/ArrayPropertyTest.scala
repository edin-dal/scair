import fastparse.*
import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*
import scair.clair.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.parse.Parser
import scair.utils.*

/** An operation whose properties are arrays. Erasure hides their element types
  * from a plain type test, so structuring one has to establish them itself.
  */
case class ArrayPropertyOp(
    ints: ArrayAttribute[IntData],
    anything: ArrayAttribute[Attribute],
    eitherOr: ArrayAttribute[IntData | StringData],
) extends DerivedOperation["test.array_property"] derives OpDefs

/** The same question one level down, where a derived attribute's parameters are
  * read back out of a Seq[Attribute] as it is parsed.
  */
case class NamesAttr(
    names: ArrayAttribute[StringData]
) extends DerivedAttribute["test.names"] derives AttrDefs

class ArrayPropertyTest extends AnyFlatSpec:

  val ops = summon[OpDefs[ArrayPropertyOp]]

  /** Structures an operation from properties, defaulting to well-typed ones. */
  def structuring(
      ints: Attribute = ArrayAttribute(IntData(1), IntData(2)),
      anything: Attribute = ArrayAttribute(StringData("a")),
      eitherOr: Attribute = ArrayAttribute(IntData(1), StringData("a")),
  ) =
    ops.UnstructuredOp(
      properties = Map(
        "ints" -> ints,
        "anything" -> anything,
        "eitherOr" -> eitherOr,
      )
    ).structured

  "Well-typed array properties" should "structure" in {
    structuring().isOK should be(true)
  }

  "An array of the wrong element type" should "be rejected" in {
    structuring(ints = ArrayAttribute(StringData("a"))) should matchPattern {
      case Err(msg, _)
          if msg.startsWith("Type mismatch for property \"ints\"") =>
    }
  }

  it should "be rejected however far along the mismatch sits" in {
    structuring(ints = ArrayAttribute(IntData(1), StringData("a"))).isOK should
      be(false)
  }

  "An empty array" should "satisfy any element type" in {
    structuring(ints = ArrayAttribute()).isOK should be(true)
  }

  "An array property declared over Attribute" should "take any elements" in {
    val mixed =
      structuring(anything = ArrayAttribute(IntData(1), StringData("a")))
    mixed.isOK should be(true)
  }

  "A union element type" should "take either side" in {
    structuring(eitherOr = ArrayAttribute(IntData(1))).isOK should be(true)
    structuring(eitherOr = ArrayAttribute(StringData("a"))).isOK should be(true)
  }

  it should "take neither anything else" in {
    structuring(eitherOr = ArrayAttribute(FloatData(1.0))).isOK should be(false)
  }

  "A property that is not an array at all" should "be rejected" in {
    structuring(ints = StringData("a")).isOK should be(false)
  }

  val names = summon[AttrDefs[NamesAttr]]
  val parser = Parser(scair.MLContext(), allowUnregisteredDialect = true)

  def parsingNames(input: String) =
    fastparse.parse(input, (ctx: P[?]) => names.parse(using ctx, parser))

  "An attribute parameter of the right element type" should "parse" in {
    parsingNames("""<["a", "b"]>""").isSuccess should be(true)
  }

  it should "be rejected when the elements are of another" in {
    an[Exception] should be thrownBy parsingNames("<[1, 2]>")
  }
