package scair.tools.opt

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers.*
import scair.ir.*
import scair.utils.OK

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.PrintStream
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.Files
import scala.io.Source

class LocationTest extends AnyFlatSpec:

  "scair-opt" should "parse the location flags with disabled defaults" in {
    ScairOpt.parseArgs(Array.empty).printLocations shouldBe false
    ScairOpt.parseArgs(Array("--print-locations")).printLocations shouldBe true
    ScairOpt.parseArgs(Array.empty).parseLocations shouldBe false
    ScairOpt.parseArgs(Array("--parse-locations")).parseLocations shouldBe true
  }

  it should "apply original-file offsets to split input chunks" in {
    val input =
      "\n  builtin.module {}\n// -----\n\n    builtin.module {}\n// -----\nbuiltin.module {} loc(\"explicit\":3:4)"
    val source = Source
      .fromInputStream(ByteArrayInputStream(input.getBytes(UTF_8)))
    try
      val modules = ScairOpt
        .parse(
          ScairOptArgs(
            input = Some("split.mlir"),
            splitInputFile = true,
            parseLocations = true,
          )
        )(source)
      val locations = modules.toSeq.map {
        case OK(op) => op.location
        case error  => fail(error.toString)
      }
      locations shouldBe Seq(
        FileLineColLoc("split.mlir", 2, 3),
        FileLineColLoc("split.mlir", 5, 5),
        FileLineColLoc("explicit", 3, 4),
      )
    finally source.close()
  }

  it should "expose locations end to end while preserving default output" in {
    val input = Files.createTempFile("scair-locations-", ".mlir")
    Files.writeString(
      input,
      "builtin.module {\n  %x = \"arith.constant\"() <{value = 1 : i32}> : () -> i32\n}\n",
    )
    def run(flags: String*): String =
      val output = ByteArrayOutputStream()
      val stream = PrintStream(output, true, UTF_8)
      val previous = System.out
      try
        System.setOut(stream)
        Console
          .withOut(stream) {
            ScairOpt.main((Seq(input.toString) ++ flags).toArray)
          }
        output.toString(UTF_8)
      finally
        System.setOut(previous)
        stream.close()
    try
      run() shouldBe
        "builtin.module {\n  %0 = \"arith.constant\"() <{value = 1 : i32}> : () -> i32\n}\n"
      val located = run("--print-locations", "--parse-locations")
      located should include(
        s"\"arith.constant\"() <{value = 1 : i32}> : () -> i32 loc(\"$input\":2:8)"
      )
      located should endWith(s"} loc(\"$input\":1:1)\n")
      val generic = run("--print-locations", "--parse-locations", "--print-generic")
      generic should include(s"loc(\"$input\":2:8)")
      generic should endWith(s"loc(\"$input\":1:1)\n")
    finally Files.deleteIfExists(input)
  }
