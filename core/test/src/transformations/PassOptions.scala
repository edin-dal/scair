import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*
import scair.MLContext
import scair.ir.*
import scair.transformations.*

import java.util.Locale

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
) extends ModulePass derives PassCompanion:
  override def transform(op: Operation)(using MLContext) = op

final case class LongNamedLLVMPass() extends ModulePass with PassName["short"]
    derives PassCompanion:
  override def transform(op: Operation)(using MLContext) = op

final case class EnumOptionsPass(
    selected: Option[Direction] = None,
    order: List[Direction] = List(),
) extends ModulePass derives PassCompanion:
  override def transform(op: Operation)(using MLContext) = op

object PassConstruction:
  var defaults = 0
  var instances = 0

  def defaultCount(): Int =
    defaults += 1
    1

final case class CountedPass(count: Int = PassConstruction.defaultCount())
    extends ModulePass derives PassCompanion:
  PassConstruction.instances += 1
  override def transform(op: Operation)(using MLContext) = op

class PassOptionsTest extends AnyFlatSpec:

  val companions =
    summonPasses[(OptionsPass, LongNamedLLVMPass, EnumOptionsPass)]

  val ctx = MLContext()
  companions.foreach(ctx.registerPass)

  def pipeline(spec: String): Either[String, Seq[ModulePass]] =
    PipelineParser(ctx).parse(spec)

  def build(spec: String) = pipeline(spec).map(_.head)

  "PipelineParser" should "construct pipelines with typed options" in {
    pipeline(
      """short, options-pass{count=1 names="a,b",c label="hi, {there} \"you\""},enum-options-pass{}"""
    ) shouldEqual Right(
      Seq(
        LongNamedLLVMPass(),
        OptionsPass(
          1,
          label = "hi, {there} \"you\"",
          names = List("a,b", "c"),
        ),
        EnumOptionsPass(),
      )
    )
    pipeline("") shouldEqual Right(Seq())
  }

  it should "reject malformed pipelines" in {
    for spec <- Seq(
        "short,",
        "short{x=}",
        "short{x",
        "options-pass{count=1,}",
        "short short",
        ",short",
        "options-pass{count=1foo}",
        "options-pass{count=1 names=}",
        "options-pass{count=1 names=x,,y}",
      )
    do pipeline(spec).isLeft shouldBe true
  }

  it should "report failures without repeating pass construction" in {
    val context = MLContext()
    summonPasses[Tuple1[CountedPass]].foreach(context.registerPass)
    for input <- Seq("counted-pass,not-a-pass", "counted-pass,") do
      PassConstruction.defaults = 0
      PassConstruction.instances = 0
      PipelineParser(context).parse(input).isLeft shouldBe true
      PassConstruction.defaults shouldEqual 1
      PassConstruction.instances shouldEqual 1
  }

  it should "report unregistered passes" in {
    pipeline("not-a-pass").left.toOption.get should include(
      "'not-a-pass' does not refer to a registered pass."
    )
    pipeline("not-a-pass").left.toOption.get should include(
      "Currently registered passes are:"
    )
  }

  "PassCompanion" should "derive names" in {
    companions.map(_.name) shouldEqual
      Seq("options-pass", "short", "enum-options-pass")
    kebabCase("ConvertArithToLLVM") shouldEqual "convert-arith-to-llvm"
    kebabCase("LLVMToArith") shouldEqual "llvm-to-arith"
  }

  it should "derive locale-independent names" in {
    val previous = Locale.getDefault
    try
      Locale.setDefault(Locale.forLanguageTag("tr-TR"))
      kebabCase("maxIterations") shouldEqual "max-iterations"
      kebabCase("LLVMToIR") shouldEqual "llvm-to-ir"
    finally Locale.setDefault(previous)
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
    build("options-pass{count=1 verbose}") shouldEqual Right(
      OptionsPass(1, verbose = true)
    )
  }

  it should "support automatic enums inside optional and list options" in {
    val spec = "enum-options-pass{selected=bottom-up order=top-down,bottom-up}"
    val pass = EnumOptionsPass(
      Some(Direction.BottomUp),
      List(Direction.TopDown, Direction.BottomUp),
    )
    build(spec) shouldEqual Right(pass)
    build("enum-options-pass{selected order}").isLeft shouldBe true
    build("enum-options-pass{order=top-down,sideways}").left.toOption.get should
      include(
        "enum-options-pass: option 'order': expected List[top-down|bottom-up]"
      )
  }

  it should "reject duplicate options within each pass" in {
    for (options, key) <- Seq(
        "count=1 count=2" -> "count",
        "count=1 count=1" -> "count",
        "count=1 verbose verbose=false" -> "verbose",
        "count=1 verbose=false verbose" -> "verbose",
        "count=1 limit=2 limit=4" -> "limit",
        "count=1 sizes=1,2 sizes=3" -> "sizes",
        "count=1 names=x names=y" -> "names",
      )
    do
      build(s"options-pass{$options}").left.toOption.get should
        include(s"options-pass: duplicate option '$key'")

    pipeline("options-pass{count=1},options-pass{count=2}") shouldEqual
      Right(Seq(OptionsPass(1), OptionsPass(2)))
  }

  it should "report the input, caret and message without a location header" in {
    pipeline("options-pass{count=x}").left.toOption.get shouldEqual
      """options-pass{count=x}
        |                   ^
        |options-pass: option 'count': expected Int""".stripMargin
  }

  it should "report bad options" in {
    for (spec, message) <- Seq(
        "options-pass" -> "options-pass: missing required option 'count'",
        "options-pass{count=x}" -> "options-pass: option 'count': expected Int",
        "options-pass{count}" -> "options-pass: option 'count': expected Int",
        "options-pass{count=1 sizes}" ->
          "options-pass: option 'sizes': expected Seq[Int]",
        "options-pass{count=1 label}" ->
          "options-pass: option 'label': expected String",
        "options-pass{count=1 direction=sideways}" ->
          "options-pass: option 'direction': expected top-down|bottom-up",
        "options-pass{count=1 cuont=2}" ->
          "options-pass: unknown option 'cuont', expected one of: count, scale, label, verbose, limit, sizes, names, direction",
      )
    do build(spec).left.toOption.get should include(message)
    build("options-pass{count=1,2}").isLeft shouldBe true
  }
