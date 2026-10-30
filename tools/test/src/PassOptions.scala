import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*
import scair.MLContext
import scair.ir.*
import scair.transformations.*
import scair.tools.PipelineParser
import scair.passes.mlir_opt.MLIROpt

enum Direction:
  case TopDown, BottomUp

final case class OptionsPass(
    count: Int,
    scale: Double = 1.0,
    label: String = "none",
    verbose: Boolean = false,
    limit: Option[Int] = None,
    sizes: Seq[Int] = Seq(),
    names: Seq[String] = Seq("a"),
    direction: Direction = Direction.TopDown,
) extends ModulePass derives PassCompanion:
  override def transform(op: Operation)(using MLContext) = op

final case class LongNamedLLVMPass()
    extends ModulePass
    with PassName["ShortPass"] derives PassCompanion:
  override def transform(op: Operation)(using MLContext) = op

final case class EnumOptionsPass(
    selected: Option[Direction] = None,
    order: Seq[Direction] = Seq(),
) extends ModulePass derives PassCompanion:
  override def transform(op: Operation)(using MLContext) = op

final case class CamelCasePass(maxIterations: Int = 10) extends ModulePass
    derives PassCompanion:
  override def transform(op: Operation)(using MLContext) = op

final case class NumericOptionsPass(
    long: Long,
    big: BigInt,
    optional: Option[BigInt],
    longs: Seq[Long],
    bigs: Seq[BigInt],
    flags: Seq[Boolean],
    flag: Option[Boolean],
    scales: Seq[Double],
    text: Option[String],
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
    summonPasses[
      (OptionsPass, LongNamedLLVMPass, EnumOptionsPass, MLIROpt)
    ]

  val ctx = MLContext()
  companions.foreach(ctx.registerPass)

  def pipeline(spec: String): Either[String, Seq[ModulePass]] =
    PipelineParser(ctx).parse(spec)

  def build(spec: String) = pipeline(spec).map(_.head)

  "PipelineParser" should "construct pipelines with typed options" in {
    pipeline(
      """ShortPass, OptionsPass{count=1 names="a,b",c label="hi, {there} \"you\""},EnumOptionsPass{}"""
    ) shouldEqual Right(
      Seq(
        LongNamedLLVMPass(),
        OptionsPass(
          1,
          label = "hi, {there} \"you\"",
          names = Seq("a,b", "c"),
        ),
        EnumOptionsPass(),
      )
    )
    pipeline("") shouldEqual Right(Seq())
  }

  it should "expand mlir-opt pipelines" in {
    def mlirOpt(pipeline: String) = MLIROpt(
      arguments = Seq(
        "--mlir-print-op-generic",
        "--allow-unregistered-dialect",
        "-p",
        s"builtin.module($pipeline)",
      )
    )
    pipeline("mlir-opt[cse,canonicalize{max-iterations=2}]") shouldEqual
      Right(Seq(mlirOpt("cse,canonicalize{max-iterations=2}")))
    pipeline("ShortPass,mlir-opt[],mlir-opt{generic=false}") shouldEqual Right(
      Seq(LongNamedLLVMPass(), mlirOpt(""), MLIROpt(generic = false))
    )
    pipeline("ShortPass,foo[cse]").left.toOption.get should include(
      "`mlir-opt` to mark an MLIR pipeline"
    )
    pipeline("mlir-opt[cse").isLeft shouldBe true
    build("OptionsPass{count=1 label=/tmp/a.mlir}") shouldEqual
      Right(OptionsPass(1, label = "/tmp/a.mlir"))
  }

  it should "reject malformed pipelines" in {
    for spec <- Seq(
        "ShortPass,",
        "ShortPass{x=}",
        "ShortPass{x",
        "OptionsPass{count=1,}",
        "ShortPass ShortPass",
        ",ShortPass",
        "OptionsPass{count=1foo}",
        "OptionsPass{count=1 names=}",
        "OptionsPass{count=1 names=x,,y}",
      )
    do pipeline(spec).isLeft shouldBe true
  }

  it should "report failures without repeating pass construction" in {
    val context = MLContext()
    summonPasses[Tuple1[CountedPass]].foreach(context.registerPass)
    for input <- Seq("CountedPass,not-a-pass", "CountedPass,") do
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

  "PassCompanion" should
    "use Scala class names unless overridden by PassName" in {
      companions.map(_.name) shouldEqual
        Seq("OptionsPass", "ShortPass", "EnumOptionsPass", "mlir-opt")
    }

  it should "use exact Scala field and enum case names" in {
    val context = MLContext()
    summonPasses[Tuple1[CamelCasePass]].foreach(context.registerPass)
    PipelineParser(context).parse("CamelCasePass{maxIterations=3}") shouldEqual
      Right(Seq(CamelCasePass(3)))
    PipelineParser(context).parse("CamelCasePass{max-iterations=3}")
      .isLeft shouldBe true
    pipeline("options-pass{count=1}").isLeft shouldBe true
    build("OptionsPass{count=1 direction=bottom-up}").isLeft shouldBe true
  }

  it should "build passes from options, with defaults" in {
    build("OptionsPass{count=3}") shouldEqual Right(OptionsPass(3))
    build(
      "OptionsPass{count=-1 scale=2 label=\"a b\" verbose limit=4 sizes=1,2,3 names=x,y direction=BottomUp}"
    ) shouldEqual Right(
      OptionsPass(
        -1,
        2.0,
        "a b",
        true,
        Some(4),
        Seq(1, 2, 3),
        Seq("x", "y"),
        Direction.BottomUp,
      )
    )
    build("OptionsPass{count=1 verbose}") shouldEqual Right(
      OptionsPass(1, verbose = true)
    )
  }

  it should "support automatic enums inside Option and Seq options" in {
    val spec = "EnumOptionsPass{selected=BottomUp order=TopDown,BottomUp}"
    val pass = EnumOptionsPass(
      Some(Direction.BottomUp),
      Seq(Direction.TopDown, Direction.BottomUp),
    )
    build(spec) shouldEqual Right(pass)
    build("EnumOptionsPass{selected order}").isLeft shouldBe true
    build("EnumOptionsPass{order=TopDown,sideways}").left.toOption.get should
      include(
        "EnumOptionsPass: option 'order': expected Seq[TopDown|BottomUp]"
      )
  }

  it should "support Long, BigInt and scalar wrappers" in {
    val context = MLContext()
    summonPasses[Tuple1[NumericOptionsPass]].foreach(context.registerPass)
    val large = BigInt("9223372036854775808")
    val input =
      s"NumericOptionsPass{long=9223372036854775807 big=$large optional=$large longs=1,-2 bigs=$large,-$large flags=true,false flag=false scales=1.5,2 text=hello}"
    PipelineParser(context).parse(input) shouldEqual Right(
      Seq(
        NumericOptionsPass(
          Long.MaxValue,
          large,
          Some(large),
          Seq(1L, -2L),
          Seq(large, -large),
          Seq(true, false),
          Some(false),
          Seq(1.5, 2.0),
          Some("hello"),
        )
      )
    )
    for invalid <- Seq(
        input.replace("long=9223372036854775807", "long=9223372036854775808"),
        input.replace(s"big=$large", "big=oops"),
        input.replace("flag=false", "flag"),
        input.replace("flags=true,false", "flags=true,maybe"),
      )
    do PipelineParser(context).parse(invalid).isLeft shouldBe true
  }

  it should "reject unsupported types and nested wrappers at compile time" in {
    assertDoesNotCompile(
      """case class Bad(x: Float) extends scair.transformations.ModulePass derives scair.transformations.PassCompanion { def transform(op: scair.ir.Operation)(using scair.MLContext) = op }"""
    )
    assertDoesNotCompile(
      """case class Bad(x: List[Int]) extends scair.transformations.ModulePass derives scair.transformations.PassCompanion { def transform(op: scair.ir.Operation)(using scair.MLContext) = op }"""
    )
    assertDoesNotCompile(
      """case class Bad(x: Option[Option[Int]]) extends scair.transformations.ModulePass derives scair.transformations.PassCompanion { def transform(op: scair.ir.Operation)(using scair.MLContext) = op }"""
    )
    assertDoesNotCompile(
      """case class Bad(x: Option[Seq[Int]]) extends scair.transformations.ModulePass derives scair.transformations.PassCompanion { def transform(op: scair.ir.Operation)(using scair.MLContext) = op }"""
    )
    assertDoesNotCompile(
      """case class Bad(x: Seq[Option[Int]]) extends scair.transformations.ModulePass derives scair.transformations.PassCompanion { def transform(op: scair.ir.Operation)(using scair.MLContext) = op }"""
    )
    assertDoesNotCompile(
      """case class Bad(x: Seq[Seq[Int]]) extends scair.transformations.ModulePass derives scair.transformations.PassCompanion { def transform(op: scair.ir.Operation)(using scair.MLContext) = op }"""
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
      build(s"OptionsPass{$options}").left.toOption.get should
        include(s"OptionsPass: duplicate option '$key'")

    pipeline("OptionsPass{count=1},OptionsPass{count=2}") shouldEqual
      Right(Seq(OptionsPass(1), OptionsPass(2)))
  }

  it should "report the input, caret and message without a location header" in {
    pipeline("OptionsPass{count=x}").left.toOption.get shouldEqual
      """OptionsPass{count=x}
        |                  ^
        |OptionsPass: option 'count': expected Int""".stripMargin
  }

  it should "report bad options" in {
    for (spec, message) <- Seq(
        "OptionsPass" -> "OptionsPass: missing required option 'count'",
        "OptionsPass{count=x}" -> "OptionsPass: option 'count': expected Int",
        "OptionsPass{count}" -> "OptionsPass: option 'count': expected Int",
        "OptionsPass{count=1 sizes}" ->
          "OptionsPass: option 'sizes': expected Seq[Int]",
        "OptionsPass{count=1 label}" ->
          "OptionsPass: option 'label': expected String",
        "OptionsPass{count=1 direction=sideways}" ->
          "OptionsPass: option 'direction': expected TopDown|BottomUp",
        "OptionsPass{count=1 cuont=2}" ->
          "OptionsPass: unknown option 'cuont', expected one of: count, scale, label, verbose, limit, sizes, names, direction",
      )
    do build(spec).left.toOption.get should include(message)
    build("OptionsPass{count=1,2}").isLeft shouldBe true
  }
