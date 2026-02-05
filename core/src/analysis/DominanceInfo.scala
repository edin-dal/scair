package scair.analysis

import scair.ir.*

import scala.collection.mutable

/** A simple DominanceInfo-style analysis for ScaIR.
  *
  * Scope:
  *   - Computes CFG block dominance per Region.
  *   - Computes op dominance (same-block order, else block dominance).
  *   - Computes hierarchical dominance: uses in nested regions are "lifted" to
  *     the ancestor operation in the definition's region.
  *
  * Notes:
  *   - Entry block is taken as region.blocks.head (single-entry assumption).
  *   - CFG edges are collected from op.last.successors.
  */
final class DominanceInfo(root: Operation):

  // ----------------------------
  // Public API
  // ----------------------------

  /** Returns true if block 'a' dominates block 'b' within the given region. */
  def blockDominates(region: Region, a: Block, b: Block): Boolean =
    val info = regionInfo(region)
    info.doms.get(b).exists(_.contains(a))

  /** Returns true if 'defOp' dominates 'useOp' (hierarchical dominance aware).
    */
  def opDominates(defOp: Operation, useOp: Operation): Boolean =
    (for
      defBlock <- defOp.containerBlock
      defRegion <- defBlock.containerRegion
      liftedUse <- liftUseToRegion(useOp, defRegion)
      useBlock <- liftedUse.containerBlock
    yield
      if defBlock eq useBlock then
        // Same block => order matters
        val idx = blockOpIndex(useBlock)
        idx.get(defOp).exists(d => idx.get(liftedUse).exists(u => d <= u))
      else
        // Different blocks in same region => CFG block dominance
        blockDominates(defRegion, defBlock, useBlock)
    ).getOrElse(false)

  /** Returns true if SSA value 'v' dominates its use at operation 'user'.
    *
    * Rules:
    *   - Block argument: only valid within the defining block (and nested
    *     regions under ops in that block, via hierarchical lifting).
    *   - Op result: defOp must dominate the (lifted) user op.
    */
  def valueDominates(v: Value[Attribute], user: Operation): Boolean =
    v.owner match
      case Some(b: Block) =>
        // Block arguments are only in-scope in their own block.
        // If 'user' is inside nested regions, lift it up to 'b''s region
        // and require it's still in the same defining block.
        (for
          defRegion <- b.containerRegion
          liftedUse <- liftUseToRegion(user, defRegion)
          useBlock <- liftedUse.containerBlock
        yield useBlock eq b).getOrElse(false)

      case Some(defOp: Operation) =>
        opDominates(defOp, user)

      case _ =>
        // No owner => can't prove dominance.
        false

  // ----------------------------
  // Internal representation
  // ----------------------------

  private case class RegionDomInfo(
      region: Region,
      blocks: Vector[Block],
      preds: Map[Block, Vector[Block]],
      doms: Map[Block, Set[Block]],
  )

  private val regionCache: mutable.Map[Region, RegionDomInfo] = mutable.Map
    .empty

  private val blockIndexCache: mutable.Map[Block, Map[Operation, Int]] =
    mutable.Map.empty

  private def regionInfo(r: Region): RegionDomInfo =
    regionCache.getOrElseUpdate(r, computeRegionDom(r))

  private def blockOpIndex(b: Block): Map[Operation, Int] =
    blockIndexCache.getOrElseUpdate(
      b,
      b.operations.toSeq.zipWithIndex.toMap,
    )

  // ----------------------------
  // Region CFG extraction
  // ----------------------------

  private def computeRegionDom(r: Region): RegionDomInfo =
    val blocks = r.blocks.toVector
    val allBlocks = blocks.toSet

    // Collect successors per block from the block terminator.
    val succs: Map[Block, Vector[Block]] =
      blocks.map { b =>
        val s = b.operations.lastOption.toVector.flatMap(_.successors)
        b -> s
      }.toMap

    // Build preds from succs
    val predsBuf: mutable.Map[Block, mutable.ArrayBuffer[Block]] =
      mutable.Map.from(blocks.map(b => b -> mutable.ArrayBuffer.empty[Block]))

    for (b, ss) <- succs do
      ss.foreach { s =>
        predsBuf.getOrElseUpdate(s, mutable.ArrayBuffer.empty) += b
      }

    val preds: Map[Block, Vector[Block]] =
      predsBuf.view.mapValues(_.toVector).toMap

    // Compute dominance sets via classic iterative algorithm
    val entryOpt = blocks.headOption

    val doms0: mutable.Map[Block, Set[Block]] = mutable.Map.empty
    for b <- blocks do
      if entryOpt.exists(_ eq b) then doms0(b) = Set(b)
      else if preds.getOrElse(b, Vector.empty).isEmpty then
        doms0(b) = Set(b) // unreachable-ish
      else doms0(b) = allBlocks

    var changed = true
    while changed do
      changed = false
      for b <- blocks do
        if !entryOpt.exists(_ eq b) && preds.getOrElse(b, Vector.empty).nonEmpty
        then
          val predSets = preds(b).map(p => doms0.getOrElse(p, allBlocks))
          val inter = predSets.reduce(_ intersect _)
          val newDom = inter + b
          if newDom != doms0(b) then
            doms0(b) = newDom
            changed = true

    RegionDomInfo(r, blocks, preds, doms0.toMap)

  // ----------------------------
  // Hierarchical lifting
  // ----------------------------

  /** Lift a use operation up the parent chain until it's in the given region.
    *
    * If 'useOp' is already in that region, returns it. If we can't reach that
    * region (e.g., malformed IR), returns None.
    */
  private def liftUseToRegion(
      useOp: Operation,
      targetRegion: Region,
  ): Option[Operation] =
    def regionOf(op: Operation): Option[Region] =
      op.containerBlock.flatMap(_.containerRegion)

    var cur: Operation = useOp
    var curRegion = regionOf(cur)

    while curRegion.isDefined && (curRegion.get ne targetRegion) do
      // climb via region's containerOperation (not op.parent!)
      curRegion.get.containerOperation match
        case Some(parentOp) =>
          cur = parentOp
          curRegion = regionOf(cur)
        case None =>
          return None

    curRegion match
      case Some(r) if r eq targetRegion => Some(cur)
      case _                            => None
