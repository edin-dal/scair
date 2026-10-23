import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*
import scair.MLContext
import scair.ir.*
import scair.transformations.*

enum Direction:
  case TopDown, BottomUp

final case class OptionsPass(
    count: Int,
    scale: Double = 1.0,
    label: String = "none",
    verbose: Boolean = false,
    limit: Option[Int] = None,
    sizes: Seq[Int] = Seq(),
    names: List[String] = List("a"),
    direction: Direction = Direction.TopDown,
) extends ModulePass derives PassDefs:
  override def transform(op: Operation)(using MLContext) = op

@passName("short")
final case class LongNamedLLVMPass() extends ModulePass derives PassDefs:
  override def transform(op: Operation)(using MLContext) = op

class PassOptionsTest extends AnyFlatSpec:

  val defs = PassDefs[OptionsPass]

  def build(spec: String) =
    PassPipeline.parse(spec).flatMap(specs => defs.fromArgs(specs.head.args))

  "PassPipeline" should "parse names, options and lists" in {
    PassPipeline
      .parse(
        "a, b{},c-d{x=1 y z=1,-2.5e3,foo s=\"hi, {there} \\\"you\\\"\"}"
      ) shouldEqual Right(
      Seq(
        PassSpec("a", Map()),
        PassSpec("b", Map()),
        PassSpec(
          "c-d",
          Map(
            "x" -> Seq("1"),
            "y" -> Seq(),
            "z" -> Seq("1", "-2.5e3", "foo"),
            "s" -> Seq("hi, {there} \"you\""),
          ),
        ),
      )
    )
    PassPipeline.parse("") shouldEqual Right(Seq())
  }

  it should "reject malformed pipelines" in {
    for spec <- Seq("a,", "a{x=}", "a{x", "a{x=1,}", "a b", ",a") do
      PassPipeline.parse(spec).isLeft shouldBe true
  }

  "PassDefs" should "derive names" in {
    defs.name shouldEqual "options-pass"
    PassDefs[LongNamedLLVMPass].name shouldEqual "short"
    kebabCase("ConvertArithToLLVM") shouldEqual "convert-arith-to-llvm"
    kebabCase("LLVMToArith") shouldEqual "llvm-to-arith"
  }

  it should "build passes from options, with defaults" in {
    build("options-pass{count=3}") shouldEqual Right(OptionsPass(3))
    build(
      "options-pass{count=-1 scale=2 label=\"a b\" verbose limit=4 sizes=1,2,3 names=x,y direction=bottom-up}"
    ) shouldEqual Right(
      OptionsPass(
        -1,
        2.0,
        "a b",
        true,
        Some(4),
        Seq(1, 2, 3),
        List("x", "y"),
        Direction.BottomUp,
      )
    )
    build("options-pass{count=1 verbose=false limit sizes}") shouldEqual Right(
      OptionsPass(1, sizes = Seq())
    )
  }

  it should "report bad options" in {
    build("options-pass") shouldEqual Left(
      "options-pass: missing required option 'count'"
    )
    build("options-pass{count=x}") shouldEqual Left(
      "options-pass: option 'count': expected Int, got 'x'"
    )
    build("options-pass{count=1,2}") shouldEqual Left(
      "options-pass: option 'count': expected a single Int"
    )
    build("options-pass{count=1 direction=sideways}") shouldEqual Left(
      "options-pass: option 'direction': expected top-down|bottom-up, got 'sideways'"
    )
    build("options-pass{count=1 cuont=2}") shouldEqual Left(
      "options-pass: unknown option 'cuont', expected one of: count, scale, label, verbose, limit, sizes, names, direction"
    )
  }

  it should "print passes back as pipelines" in {
    defs.show(OptionsPass(3)) shouldEqual "options-pass{count=3}"
    val pass = OptionsPass(
      3,
      label = "a b",
      verbose = true,
      sizes = Seq(1, 2),
      direction = Direction.BottomUp,
    )
    defs.show(pass) shouldEqual
      "options-pass{count=3 label=\"a b\" verbose=true sizes=1,2 direction=bottom-up}"
    build(defs.show(pass)) shouldEqual Right(pass)
    PassDefs[LongNamedLLVMPass].show(LongNamedLLVMPass()) shouldEqual "short"
  }

  it should "describe its signature" in {
    defs.signature shouldEqual
      "options-pass{count: Int (required), scale: Double = 1.0, label: String = none, verbose: Boolean = false, limit: Option[Int], sizes: Seq[Int], names: List[String] = a, direction: top-down|bottom-up = top-down}"
  }
