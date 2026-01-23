package scair

import scair.analysis.DominanceInfo
import scair.dialects.builtin.*
import scair.ir.*

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers.*
import scair.utils.Err

final class DominanceInfoSpec extends AnyFlatSpec:

  object TestIR:
    // Use any stable type attribute; dominance does not depend on types.
    val ty: TypeAttribute = I32

    def res(): Result[Attribute] = Result(ty)

    def defOp(name: String = "test.def"): Operation =
      UnregisteredOperation(name)(
        results = Seq(res())
      )

    def useOp(v: Value[Attribute], name: String = "test.use"): Operation =
      UnregisteredOperation(name)(
        operands = Seq(v)
      )

    /** A control-flow edge op: we encode CFG edges via `successors` on ops. */
    def br(to: Block, name: String = "test.br"): Operation =
      UnregisteredOperation(name)(successors = Seq(to))

    /** Branch with multiple successors (e.g., conditional). */
    def br2(t: Block, f: Block, name: String = "test.cond_br"): Operation =
      UnregisteredOperation(name)(successors = Seq(t, f))

    def moduleWithRegion(r: Region): ModuleOp =
      ModuleOp(r)

    def singleBlockModule(ops: Operation*): ModuleOp =
      ModuleOp(Region(Seq(Block(operations = ops.toSeq))))

    /** Helper to get the single SSA result of a def op. */
    extension (o: Operation)

      def onlyResult: Value[Attribute] =
        o.results.head.asInstanceOf[Value[Attribute]]

  import TestIR.*

  "DominanceInfo (same block)" should "accept def-before-use" in {
    val d = defOp()
    val u = useOp(d.onlyResult)

    val m = singleBlockModule(d, u)
    m.verify() match
      case e: Err => fail(e.msg)
      case _      => succeed

    val dom = DominanceInfo(m)
    dom.valueDominates(d.onlyResult, u) shouldBe true
  }

  it should "reject use-before-def in the same block" in {
    val d = defOp()
    val u = useOp(d.onlyResult)

    // use appears before def
    val m = singleBlockModule(u, d)
    m.verify() match
      case e: Err => fail(e.msg)
      case _      => succeed

    val dom = DominanceInfo(m)
    dom.valueDominates(d.onlyResult, u) shouldBe false
  }

  "DominanceInfo (CFG)" should
    "accept def in entry dominating use in successor block" in {
      val d = defOp()
      val u = useOp(d.onlyResult)

      val b1 = Block(operations = Seq(u))
      val entry = Block(operations = Seq(d, br(b1)))

      val m = moduleWithRegion(Region(Seq(entry, b1)))
      m.verify() match
        case e: Err => fail(e.msg)
        case _      => succeed

      val dom = DominanceInfo(m)
      dom.valueDominates(d.onlyResult, u) shouldBe true
    }

  it should
    "reject def that does not dominate across a join (entry can bypass def block)" in {
      val d = defOp()
      val u = useOp(d.onlyResult)

      val b2 = Block(operations = Seq(u))
      val b1 = Block(operations = Seq(d, br(b2)))
      val entry = Block(operations = Seq(br2(b1, b2)))

      val m = moduleWithRegion(Region(Seq(entry, b1, b2)))
      m.verify() match
        case e: Err => fail(e.msg)
        case _      => succeed

      val dom = DominanceInfo(m)
      dom.valueDominates(d.onlyResult, u) shouldBe false
    }

  "DominanceInfo (hierarchical / nested regions)" should
    "accept a parent-scope def that dominates the parent op containing the nested region" in {
      val d = defOp()

      // inner region uses %d
      val innerUse = useOp(d.onlyResult)
      val innerRegion = Region(Seq(Block(operations = Seq(innerUse))))

      val hasRegion =
        UnregisteredOperation("test.has_region")(regions = Seq(innerRegion))

      val m = singleBlockModule(d, hasRegion)
      m.verify() match
        case e: Err => fail(e.msg)
        case _      => succeed

      val dom = DominanceInfo(m)
      dom.valueDominates(d.onlyResult, innerUse) shouldBe true
    }

  it should
    "reject a parent-scope def that appears after the op containing the nested region" in {
      val d = defOp()

      val innerUse = useOp(d.onlyResult)
      val innerRegion = Region(Seq(Block(operations = Seq(innerUse))))

      val hasRegion =
        UnregisteredOperation("test.has_region")(regions = Seq(innerRegion))

      // def is after the region-containing op
      val m = singleBlockModule(hasRegion, d)
      m.verify() match
        case e: Err => fail(e.msg)
        case _      => succeed

      val dom = DominanceInfo(m)
      dom.valueDominates(d.onlyResult, innerUse) shouldBe false
    }

  "DominanceInfo (block arguments)" should
    "accept uses of a block argument within the same block" in {
      val b = Block(
        argumentsTypes = Seq(ty),
        operationsExpr = args =>
          val a0 = args.head
          Seq(useOp(a0, "test.use_block_arg")),
      )
      val m = moduleWithRegion(Region(Seq(b)))
      m.verify() match
        case e: Err => fail(e.msg)
        case _      => succeed

      val arg0 = b.arguments.head
      val user = b.operations.head

      val dom = DominanceInfo(m)
      dom.valueDominates(arg0, user) shouldBe true
    }

  it should "reject uses of a block argument from a different block" in {
    val b0 = Block(argumentsTypes = Seq(ty), operations = Seq())
    val arg0 = b0.arguments.head

    val b1User = useOp(arg0, "test.illegal_cross_block_arg_use")
    val b1 = Block(operations = Seq(b1User))

    val entry = Block(operations = Seq(br(b1)))
    val r = Region(Seq(entry, b0, b1))
    val m = moduleWithRegion(r)
    m.verify() match
      case e: Err => fail(e.msg)
      case _      => succeed

    val dom = DominanceInfo(m)
    dom.valueDominates(arg0, b1User) shouldBe false
  }
