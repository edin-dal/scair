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

  // --------------------------------------------------------------------------
  // Same-block dominance
  // --------------------------------------------------------------------------

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

  // --------------------------------------------------------------------------
  // CFG dominance
  // --------------------------------------------------------------------------

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

  it should "handle loops: def in loop header dominates uses in loop body" in {
    // Create blocks first (no ops yet)
    val entry = Block()
    val header = Block()
    val body = Block()

    // Define in header
    val d = defOp("test.loop_def")

    // Use in body (uses the value defined by d)
    val u = useOp(d.onlyResult, "test.loop_use")

    // Now that blocks exist, create terminators with successors
    val entryBr = br(header, "test.to_header")
    val headerBr = br(body, "test.to_body")
    val backedge = br(header, "test.backedge")

    // Populate blocks (this attaches ops to the correct block)
    entry.addOps(Seq(entryBr))
    header.addOps(Seq(d, headerBr))
    body.addOps(Seq(u, backedge))

    // Build module/region
    val m = moduleWithRegion(Region(Seq(entry, header, body)))

    m.verify() match
      case e: Err => fail(e.msg)
      case _      => succeed

    val dom = DominanceInfo(m)
    dom.valueDominates(d.onlyResult, u) shouldBe true
  }

  // --------------------------------------------------------------------------
  // Unreachable blocks (as implemented)
  // --------------------------------------------------------------------------

  "DominanceInfo (unreachable blocks)" should
    "not allow a reachable def to dominate a use in an unreachable-ish block" in {
      val d = defOp("test.reachable_def")
      val uDead = useOp(d.onlyResult, "test.use_in_dead")

      val dead = Block(operations = Seq(uDead)) // no preds, not entry
      val entry = Block(operations = Seq(d)) // entry has no edge to dead

      val m = moduleWithRegion(Region(Seq(entry, dead)))
      m.verify() match
        case e: Err => fail(e.msg)
        case _      => succeed

      val dom = DominanceInfo(m)
      dom.valueDominates(d.onlyResult, uDead) shouldBe false
    }

  it should "still respect same-block order inside an unreachable-ish block" in {
    val dDead = defOp("test.dead_def")
    val uDead = useOp(dDead.onlyResult, "test.dead_use")

    val dead = Block(operations = Seq(dDead, uDead)) // def before use
    val entry = Block(operations = Seq()) // entry exists but doesn't reach dead

    val m = moduleWithRegion(Region(Seq(entry, dead)))
    m.verify() match
      case e: Err => fail(e.msg)
      case _      => succeed

    val dom = DominanceInfo(m)
    dom.valueDominates(dDead.onlyResult, uDead) shouldBe true
  }

  // --------------------------------------------------------------------------
  // Hierarchical dominance / nested regions
  // --------------------------------------------------------------------------

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

  it should "work for multi-level nested regions (lift multiple times)" in {
    val d = defOp("test.outer_def")

    val deepUse = useOp(d.onlyResult, "test.deep_use")
    val innerMost = Region(Seq(Block(operations = Seq(deepUse))))

    val innerOp =
      UnregisteredOperation("test.inner_op")(regions = Seq(innerMost))
    val middleRegion = Region(Seq(Block(operations = Seq(innerOp))))

    val middleOp =
      UnregisteredOperation("test.middle_op")(regions = Seq(middleRegion))

    val m = singleBlockModule(d, middleOp)
    m.verify() match
      case e: Err => fail(e.msg)
      case _      => succeed

    val dom = DominanceInfo(m)
    dom.valueDominates(d.onlyResult, deepUse) shouldBe true
  }

  it should
    "compose hierarchical lifting with nested CFG (multi-block nested region)" in {
      val d = defOp("test.parent_def")

      // Nested region CFG: innerEntry -> innerB2
      val innerUse = useOp(d.onlyResult, "test.inner_cfg_use")
      val innerB2 = Block(operations = Seq(innerUse))
      val innerEntry = Block(operations = Seq(br(innerB2, "test.inner_br")))
      val innerRegion = Region(Seq(innerEntry, innerB2))

      val hasRegion =
        UnregisteredOperation("test.has_region_cfg")(regions = Seq(innerRegion))

      // Put def before region op so it dominates the region op, and thus dominates the nested use
      val m = singleBlockModule(d, hasRegion)
      m.verify() match
        case e: Err => fail(e.msg)
        case _      => succeed

      val dom = DominanceInfo(m)
      dom.valueDominates(d.onlyResult, innerUse) shouldBe true
    }

  it should
    "reject hierarchical dominance if parent def is after the region op (even with nested CFG)" in {
      val d = defOp("test.parent_def_late")

      val innerUse = useOp(d.onlyResult, "test.inner_cfg_use_late")
      val innerB2 = Block(operations = Seq(innerUse))
      val innerEntry =
        Block(operations = Seq(br(innerB2, "test.inner_br_late")))
      val innerRegion = Region(Seq(innerEntry, innerB2))

      val hasRegion =
        UnregisteredOperation("test.has_region_cfg")(regions = Seq(innerRegion))

      // def after region op => should not dominate nested use
      val m = singleBlockModule(hasRegion, d)
      m.verify() match
        case e: Err => fail(e.msg)
        case _      => succeed

      val dom = DominanceInfo(m)
      dom.valueDominates(d.onlyResult, innerUse) shouldBe false
    }

  // --------------------------------------------------------------------------
  // Block arguments
  // --------------------------------------------------------------------------

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

  it should
    "accept uses of a block argument inside a nested region under an op in the same block" in {
      val b = Block(
        argumentsTypes = Seq(ty),
        operationsExpr = args =>
          val a0 = args.head
          val innerUse = useOp(a0, "test.use_arg_in_nested")
          val innerRegion = Region(Seq(Block(operations = Seq(innerUse))))
          val hasRegion =
            UnregisteredOperation("test.has_region")(regions = Seq(innerRegion))
          Seq(hasRegion),
      )

      val m = moduleWithRegion(Region(Seq(b)))
      m.verify() match
        case e: Err => fail(e.msg)
        case _      => succeed

      val arg0 = b.arguments.head

      // The "user op" is innerUse (nested), dominance should lift it to hasRegion in same block,
      // and then require same defining block for block arg => true.
      val hasRegion = b.operations.head
      val innerUse =
        hasRegion.regions.head.blocks.head.operations.head

      val dom = DominanceInfo(m)
      dom.valueDominates(arg0, innerUse) shouldBe true
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
