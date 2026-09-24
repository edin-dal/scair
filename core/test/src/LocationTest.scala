package scair

import fastparse.*
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers.*
import scair.clair.OpDefs
import scair.dialects.builtin.*
import scair.dialects.test.TestOp
import scair.ir.*
import scair.parse.Parser
import scair.print.AssemblyPrinter

import java.io.StringWriter

class LocationTest extends AnyFlatSpec:
  private val point = FileLineColLoc("source.scala", 12, 7)
  private val range = FileLineColRange("source.scala", 12, 7, 14, 3)
  private val ctx = MLContext()
  ctx.registerDialect(BuiltinDialect)

  private def parse(
      input: ParserInputSource,
      path: Option[String] = Some("input.mlir"),
      offset: Int = 0,
  ): Operation =
    Parser(
      ctx,
      inputPath = path,
      allowUnregisteredDialect = true,
      inputLineOffset = offset,
    ).parse(input).get.value

  private def children(op: Operation): Seq[Operation] =
    op.regions.flatMap(_.blocks).flatMap(_.operations.toSeq)

  private def print(
      op: Operation,
      locations: Boolean = false,
      generic: Boolean = false,
  ): String =
    val out = StringWriter()
    AssemblyPrinter(
      strictlyGeneric = generic,
      p = out,
      printLocations = locations,
    ).printTopLevel(op)
    out.toString

  "Operation locations" should
    "default to unknown and preserve the precise fluent type" in {
      val op = ModuleOp(Region())
      op.location shouldBe UnknownLoc
      val pointed: op.type = op.at(point)
      val typed: ModuleOp = ModuleOp(Region()).at(point)
      (pointed eq op) shouldBe true
      typed.location shouldBe point
      pointed.location shouldBe point
      val ranged: op.type = op.at(range)
      ranged.location shouldBe range
      val cleared: op.type = op.at(UnknownLoc)
      (cleared eq op) shouldBe true
      cleared.location shouldBe UnknownLoc
    }

  it should "accept locations through all generic factory paths" in {
    UnregisteredOperation("other.op")(location = point).location shouldBe point
    TestOp(location = range).location shouldBe range
    val defs = summon[OpDefs[ModuleOp]]
    val structured = defs(regions = Seq(Region()), location = point)
    structured shouldBe a[ModuleOp]
    structured.location shouldBe point
    // A missing mandatory region forces the generated factory's fallback.
    val fallback = defs(location = range)
    fallback shouldBe a[defs.UnstructuredOp]
    fallback.location shouldBe range
    defs.UnstructuredOp(location = point).location shouldBe point
    val companion: OperationCompanion[ModuleOp] = defs
    companion(regions = Seq(Region()), location = range).location shouldBe range
  }

  "Parsing locations" should
    "capture operation-name columns after results and within nested regions" in {
      val module = parse("""  builtin.module {
                            |    %a, %b = "other.pair"() : () -> (i32, i32)
                            |    builtin.module {
                            |      "other.leaf"() : () -> ()
                            |    }
                            |  }""".stripMargin)
      module.location shouldBe FileLineColLoc("input.mlir", 1, 3)
      val Seq(pair, nested) = children(module): @unchecked
      pair.location shouldBe FileLineColLoc("input.mlir", 2, 14)
      nested.location shouldBe FileLineColLoc("input.mlir", 3, 5)
      children(nested).head.location shouldBe FileLineColLoc("input.mlir", 4, 7)
      val genericModule = parse("  \"builtin.module\"() ({}) : () -> ()")
      genericModule.location shouldBe FileLineColLoc("input.mlir", 1, 3)
    }

  it should
    "use zero coordinates only for a synthetic module and apply input offsets" in {
      val module = parse("\n  \"other.op\"() : () -> ()", offset = 20)
      module.location shouldBe FileLineColLoc("input.mlir", 0, 0)
      children(module).head.location shouldBe
        FileLineColLoc("input.mlir", 22, 3)
      children(parse("\"other.op\"() : () -> ()", path = None)).head
        .location shouldBe FileLineColLoc("-", 1, 1)
      parse("builtin.module {}", offset = 20).location shouldBe
        FileLineColLoc("input.mlir", 21, 1)
    }

  private val forms: Seq[(String, Location)] = Seq(
    "loc(unknown)" -> UnknownLoc,
    "loc(\"file\":12:7)" -> FileLineColLoc("file", 12, 7),
    "loc(\"file\":12:7 to :18)" -> FileLineColRange("file", 12, 7, 12, 18),
    "loc(\"file\":12:7 to 14:3)" -> FileLineColRange("file", 12, 7, 14, 3),
  )

  it should
    "override generated positions and round-trip each supported form in either syntax" in {
      for (suffix, location) <- forms do
        for generic <- Seq(false, true) do
          val input =
            if generic then "\"builtin.module\"() ({}) : () -> ()"
            else "builtin.module {}"
          val module = parse(s"$input $suffix", offset = 20)
          module.location shouldBe location
          val printed = print(module, locations = true, generic = generic)
          printed should endWith(s" $suffix\n")
          parse(printed).location shouldBe location
          print(parse(printed), locations = true, generic = generic) shouldBe
            printed
    }

  it should
    "fall back to unknown on streamed input while honoring explicit suffixes" in {
      val streamed = parse(Iterator("builtin.module {}"))
      streamed.location shouldBe UnknownLoc
      val explicit = parse(Iterator("builtin.module {} loc(\"file\":12:7)"))
      explicit.location shouldBe FileLineColLoc("file", 12, 7)
    }

  it should "reject unsupported location forms" in {
    for suffix <- Seq(
        "loc(?)",
        "loc(#alias)",
        "loc(\"file\":12)",
        "loc(\"name\")",
        "loc(callsite(unknown at unknown))",
        "loc(fused[unknown])",
        "loc(opaque<\"x\">)",
      )
    do Parser(ctx).parse(s"builtin.module {} $suffix").isSuccess shouldBe false
  }

  "Location printing" should
    "escape filenames exactly like string attributes" in {
      val filename = "quote\"backslash\\newline\ntab\t.mlir"
      val escaped = "\"quote\\\"backslash\\\\newline\\ntab\\t.mlir\""
      val out = StringWriter()
      AssemblyPrinter(p = out).print(StringData(filename))
      out.toString shouldBe escaped
      for location <- Seq(
          FileLineColLoc(filename, 12, 7),
          FileLineColRange(filename, 12, 7, 14, 3),
        )
      do
        val module = ModuleOp(Region()).at(location)
        val printed = print(module, locations = true)
        printed should include(s"loc($escaped:12:7")
        parse(printed).location shouldBe location
    }

  it should
    "omit locations by default and print every nested operation when enabled" in {
      val leaf = UnregisteredOperation("other.op")()
      val nested = ModuleOp(Region(leaf)).at(range)
      val module = ModuleOp(Region(nested)).at(point)
      print(module) shouldBe """builtin.module {
                                |  builtin.module {
                                |    "other.op"() : () -> ()
                                |  }
                                |}
                                |""".stripMargin
      print(module, locations = true) shouldBe
        """builtin.module {
                                                   |  builtin.module {
                                                   |    "other.op"() : () -> () loc(unknown)
                                                   |  } loc("source.scala":12:7 to 14:3)
                                                   |} loc("source.scala":12:7)
                                                   |""".stripMargin
      for generic <- Seq(false, true) do
        val printed = print(module, locations = true, generic = generic)
        " loc\\(".r.findAllIn(printed).length shouldBe 3
        val reparsed = parse(printed)
        reparsed.location shouldBe point
        children(reparsed).head.location shouldBe range
        children(children(reparsed).head).head.location shouldBe UnknownLoc
        print(reparsed, locations = true, generic = generic) shouldBe printed
        print(module, generic = generic) should not include " loc("
    }

  it should
    "include parser-generated positions without requiring explicit suffixes" in {
      val module = parse("  \"other.op\"() : () -> ()")
      val printed = print(module, locations = true)
      printed should include("loc(\"input.mlir\":1:3)")
      printed should endWith("loc(\"input.mlir\":0:0)\n")
    }
