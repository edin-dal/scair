import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*
import scair.MLContext
import scair.dialects.builtin.*
import scair.ir.*
import scair.parse.Parser
import scair.print.AssemblyPrinter
import scair.transformations.*

import java.io.StringWriter

class DialectConversionTest extends AnyFlatSpec:

  /** Parses `text` as a module and returns it. */
  def parseModule(text: String): Operation =
    Parser(MLContext(), allowUnregisteredDialect = true).parse(text).get.value

  /** Renders `op` for comparison. */
  def render(op: Operation): String =
    val out = StringWriter()
    AssemblyPrinter(p = out).print(op)
    out.toString()

  val i32ToI64 = typeConversion { case I32 => I64 }

  /** Converts the named operations by rebuilding them with converted operands
    * and result types.
    */
  def convertNamed(names: String*) = conversionPattern {
    case op if names.contains(op.name) =>
      UnregisteredOperation(op.name)(
        operands = adaptor.operands,
        results = op.results.map(r => Result(convertType(r.typ))),
        regions = op.regions,
      )
  }

  def convert(
      module: Operation,
      types: Seq[TypeConversionPattern],
      patterns: Seq[ConversionPattern],
  ): Operation =
    ConversionDriver(TypeConverter(types), patterns).convert(module)
    module

  "A conversion with no patterns" should "leave the IR untouched" in {
    val module = parseModule("""
%0 = "test.produce"() : () -> i32
"test.sink"(%0) : (i32) -> ()
""")
    val before = render(module)
    val block = module.regions.head.blocks.head
    val ops = block.operations.toSeq

    convert(module, Seq.empty, Seq.empty)

    render(module) shouldEqual before
    // Nothing was rebuilt: the very same blocks and operations are in place.
    module.regions.head.blocks.head should be theSameInstanceAs block
    block.operations.toSeq shouldEqual ops
  }

  "A conversion covering every operation" should "emit no cast" in {
    val module = parseModule("""
%0 = "test.produce"() : () -> i32
%1 = "test.consume"(%0) : (i32) -> i32
"test.sink"(%1) : (i32) -> ()
""")

    convert(
      module,
      Seq(i32ToI64),
      Seq(convertNamed("test.produce", "test.consume", "test.sink")),
    )

    render(module) shouldEqual """builtin.module {
  %0 = "test.produce"() : () -> i64
  %1 = "test.consume"(%0) : (i64) -> i64
  "test.sink"(%1) : (i64) -> ()
}
"""
  }

  "An unconverted user of a converted value" should "get a cast back" in {
    val module = parseModule("""
%0 = "test.produce"() : () -> i32
%1 = "test.consume"(%0) : (i32) -> i32
"test.sink"(%1) : (i32) -> ()
""")

    convert(
      module,
      Seq(i32ToI64),
      Seq(convertNamed("test.produce", "test.consume")),
    )

    render(module) shouldEqual """builtin.module {
  %0 = "test.produce"() : () -> i64
  %1 = "test.consume"(%0) : (i64) -> i64
  %2 = "builtin.unrealized_conversion_cast"(%1) : (i64) -> i32
  "test.sink"(%2) : (i32) -> ()
}
"""
  }

  "A converted user of an unconverted value" should
    "get a cast to the converted type" in {
      val module = parseModule("""
%0 = "test.produce"() : () -> i32
%1 = "test.consume"(%0) : (i32) -> i32
"test.sink"(%1) : (i32) -> ()
""")

      convert(module, Seq(i32ToI64), Seq(convertNamed("test.consume")))

      render(module) shouldEqual """builtin.module {
  %0 = "test.produce"() : () -> i32
  %1 = "builtin.unrealized_conversion_cast"(%0) : (i32) -> i64
  %2 = "test.consume"(%1) : (i64) -> i64
  %3 = "builtin.unrealized_conversion_cast"(%2) : (i64) -> i32
  "test.sink"(%3) : (i32) -> ()
}
"""
    }

  "Two unconverted uses of a converted value" should
    "share one cast, placed at the definition" in {
      val module = parseModule("""
%0 = "test.produce"() : () -> i32
"test.sink"(%0) : (i32) -> ()
"test.other"(%0) : (i32) -> ()
""")

      convert(module, Seq(i32ToI64), Seq(convertNamed("test.produce")))

      render(module) shouldEqual """builtin.module {
  %0 = "test.produce"() : () -> i64
  %1 = "builtin.unrealized_conversion_cast"(%0) : (i64) -> i32
  "test.sink"(%1) : (i32) -> ()
  "test.other"(%1) : (i32) -> ()
}
"""
    }

  "The block arguments of a converted operation" should "be converted too" in {
    val module = parseModule("""
"test.region"() ({
^bb0(%arg: i32):
  %0 = "test.consume"(%arg) : (i32) -> i32
  "test.sink"(%0) : (i32) -> ()
}) : () -> ()
""")

    convert(
      module,
      Seq(i32ToI64),
      Seq(convertNamed("test.region", "test.consume", "test.sink")),
    )

    render(module) shouldEqual """builtin.module {
  "test.region"() ({
  ^bb0(%0: i64):
    %1 = "test.consume"(%0) : (i64) -> i64
    "test.sink"(%1) : (i64) -> ()
  }) : () -> ()
}
"""
  }

  "The block arguments of an unconverted operation" should "keep their types" in {
    val module = parseModule("""
"test.region"() ({
^bb0(%arg: i32):
  %0 = "test.consume"(%arg) : (i32) -> i32
  "test.sink"(%0) : (i32) -> ()
}) : () -> ()
""")

    convert(
      module,
      Seq(i32ToI64),
      Seq(convertNamed("test.consume", "test.sink")),
    )

    // `test.region` declares the signature, and is left alone: the entry block
    // keeps it, and the conversion casts across it instead.
    render(module) shouldEqual """builtin.module {
  "test.region"() ({
  ^bb0(%0: i32):
    %1 = "builtin.unrealized_conversion_cast"(%0) : (i32) -> i64
    %2 = "test.consume"(%1) : (i64) -> i64
    "test.sink"(%2) : (i64) -> ()
  }) : () -> ()
}
"""
  }

  "A pattern moving a region onto its replacement" should "not throw" in {
    val module = parseModule("""
"test.region"() ({
  "test.inner"() : () -> ()
}) : () -> ()
""")

    val movesRegion = conversionPattern {
      case op if op.name == "test.region" =>
        UnregisteredOperation("test.lowered")(regions = op.regions)
    }

    convert(module, Seq.empty, Seq(movesRegion))

    render(module) shouldEqual """builtin.module {
  "test.lowered"() ({
    "test.inner"() : () -> ()
  }) : () -> ()
}
"""
  }

  "Erasing an operation whose results are used" should "throw" in {
    val module = parseModule("""
%0 = "test.produce"() : () -> i32
"test.sink"(%0) : (i32) -> ()
""")

    val erases = conversionPattern {
      case op if op.name == "test.produce" =>
        PatternAction.Erase
    }

    val thrown = intercept[Exception](convert(module, Seq.empty, Seq(erases)))
    thrown.getMessage should include("its results are still used")
  }

  "A pattern returning the wrong number of results" should "throw" in {
    val module = parseModule("""
%0 = "test.produce"() : () -> i32
"test.sink"(%0) : (i32) -> ()
""")

    val wrongArity = conversionPattern {
      case op if op.name == "test.produce" =>
        UnregisteredOperation("test.produce")()
    }

    val thrown =
      intercept[Exception](convert(module, Seq.empty, Seq(wrongArity)))
    thrown.getMessage should include("expected 1 new results but got 0")
  }

  "A cast shared across blocks" should
    "be placed at the definition, dominating both uses" in {
      val module = parseModule("""
"test.region"() ({
^bb0:
  %0 = "test.produce"() : () -> i32
  "test.br"()[^bb1] : () -> ()
^bb1:
  "test.sink"(%0) : (i32) -> ()
}) : () -> ()
""")

      convert(module, Seq(i32ToI64), Seq(convertNamed("test.produce")))

      render(module) shouldEqual """builtin.module {
  "test.region"() ({
    %0 = "test.produce"() : () -> i64
    %1 = "builtin.unrealized_conversion_cast"(%0) : (i64) -> i32
    "test.br"()[^bb0] : () -> ()
  ^bb0:
    "test.sink"(%1) : (i32) -> ()
  }) : () -> ()
}
"""
    }

  "An unconverted branch to a converted block signature" should "throw" in {
    val module = parseModule("""
"test.region"() ({
^bb0:
  %0 = "test.produce"() : () -> i32
  "test.br"(%0)[^bb1] : (i32) -> ()
^bb1(%a: i32):
  "test.sink"(%a) : (i32) -> ()
}) : () -> ()
""")

    val thrown = intercept[Exception](
      convert(
        module,
        Seq(i32ToI64),
        Seq(convertNamed("test.region", "test.produce")),
      )
    )
    thrown.getMessage should include(
      "it branches to a block whose signature was converted"
    )
  }
