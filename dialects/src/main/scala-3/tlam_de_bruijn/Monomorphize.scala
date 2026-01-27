package scair.passes

import scair.MLContext
import scair.ir.*
import scair.transformations.*
import scair.dialects.builtin.*
import scair.dialects.tlam_de_bruijn.*
import scala.collection.mutable

object Monomorphize:

  private def inst(t: TypeAttribute, tyArg: TypeAttribute): TypeAttribute =
    DBI.subst(0, tyArg, t)

  private def replaceAllUsesWith(
      from: Value[Attribute],
      to: Value[Attribute],
  ): Unit =
    val usesSnapshot = from.uses.toList

    val byOp: Map[Operation, List[Int]] =
      usesSnapshot.groupMap(_.operation)(_.index)

    byOp.foreach { case (userOp, indices0) =>
      val blk = userOp.containerBlock
        .getOrElse {
          sys.error("monomorphize: use has no container block (unexpected)")
        }

      val indices = indices0.distinct
      val newOperands =
        indices.foldLeft(userOp.operands)((ops, idx) => ops.updated(idx, to))

      val newUserOp =
        userOp.updated(
          operands = newOperands,
          successors = userOp.successors,
          results = userOp.results,
          regions = userOp.detachedRegions,
          properties = userOp.properties,
          attributes = userOp.attributes,
        )

      blk.insertOpBefore(userOp, newUserOp)
      blk.eraseOp(userOp, safeErase = false)
    }

  private def collectTLambdas(
      mod: ModuleOp
  ): Map[Value[tlamForAllType], TLambda] =
    val buf = mutable.Map.empty[Value[tlamForAllType], TLambda]

    def walkRegion(r: Region): Unit =
      r.blocks.foreach { b =>
        b.operations.foreach { op =>
          op match
            case tl: TLambda => buf += (tl.res: Value[tlamForAllType]) -> tl
            case _           => ()
          op.regions.foreach(walkRegion)
        }
      }

    mod.regions.foreach(walkRegion)
    buf.toMap

  private def collectTApplies(mod: ModuleOp): Seq[TApply] =
    val out = mutable.ArrayBuffer.empty[TApply]

    def walkRegion(r: Region): Unit =
      r.blocks.foreach { b =>
        b.operations.foreach { op =>
          op match
            case ta: TApply => out += ta
            case _          => ()
          op.regions.foreach(walkRegion)
        }
      }

    mod.regions.foreach(walkRegion)
    out.toSeq

  /** Clone a region, specializing all TypeAttributes by inst(..., tyArg), while
    * remapping SSA values so operands inside the clone refer to cloned defs.
    */
  private def cloneRegionSpec(
      r: Region,
      tyArg: TypeAttribute,
  )(using
      valueMapper: mutable.Map[Value[Attribute], Value[Attribute]]
  ): Region =
    Region(r.blocks.map(b => cloneBlockSpec(b, tyArg)))

  private def cloneBlockSpec(
      b: Block,
      tyArg: TypeAttribute,
  )(using
      valueMapper: mutable.Map[Value[Attribute], Value[Attribute]]
  ): Block =
    val newArgTypes: Seq[Attribute] =
      b.arguments.iterator.map { a =>
        a.typ match
          case t: TypeAttribute => inst(t, tyArg)
          case other            => other
      }.toSeq

    Block(
      argumentsTypes = newArgTypes,
      (newArgs: Iterable[Value[Attribute]]) =>
        valueMapper.addAll(b.arguments.zip(newArgs))
        b.operations.map(op => cloneOpSpec(op, tyArg)),
    )

  private def mapOperand(
      v: Value[Attribute]
  )(using
      valueMapper: mutable.Map[Value[Attribute], Value[Attribute]]
  ): Value[Attribute] =
    valueMapper.getOrElse(v, v)

  private def cloneOpSpec(
      op: Operation,
      tyArg: TypeAttribute,
  )(using
      valueMapper: mutable.Map[Value[Attribute], Value[Attribute]]
  ): Operation =
    op match
      case v: VLambda =>
        val newFunTyTA = inst(v.res.typ, tyArg)
        val newFunTy = newFunTyTA match
          case f: tlamFunType => f
          case other          =>
            sys
              .error(
                s"monomorphize: expected tlamFunType after inst, got $other"
              )

        val newRes = Result[tlamFunType](newFunTy)
        valueMapper += (v.res: Value[Attribute]) -> (newRes: Value[Attribute])

        val newBody = cloneRegionSpec(v.body, tyArg)
        VLambda(newBody, newRes)

      case vr: VReturn =>
        val newV = mapOperand(vr.value).asInstanceOf[Value[TypeAttribute]]
        VReturn(newV)

      case va: VApply =>
        val newFun = mapOperand(va.fun).asInstanceOf[Value[tlamFunType]]
        val newArg = mapOperand(va.arg).asInstanceOf[Value[TypeAttribute]]

        val newResTy = inst(va.res.typ, tyArg)
        val newRes = Result[TypeAttribute](newResTy)
        valueMapper += (va.res: Value[Attribute]) -> (newRes: Value[Attribute])

        VApply(newFun, newArg, newRes)

      case tl: TLambda =>
        val newForAllTA = inst(tl.res.typ, tyArg)
        val newForAll = newForAllTA match
          case fa: tlamForAllType => fa
          case other              =>
            sys
              .error(
                s"monomorphize: expected tlamForAllType after inst, got $other"
              )

        val newRes = Result[tlamForAllType](newForAll)
        valueMapper += (tl.res: Value[Attribute]) -> (newRes: Value[Attribute])

        val newBody = cloneRegionSpec(tl.body, tyArg)
        TLambda(newBody, newRes)

      case tr: TReturn =>
        val newV = mapOperand(tr.value).asInstanceOf[Value[TypeAttribute]]
        TReturn(newV)

      case ta: TApply =>
        val newFun = mapOperand(ta.fun).asInstanceOf[Value[tlamForAllType]]
        val newTyArg = inst(ta.tyArg, tyArg)

        val newResTy = inst(ta.res.typ, tyArg)
        val newRes = Result[TypeAttribute](newResTy)
        valueMapper += (ta.res: Value[Attribute]) -> (newRes: Value[Attribute])

        TApply(newFun, newTyArg, newRes)

      case other =>
        val newOperands = other.operands.map(mapOperand)
        val newRegions = other.regions.map(r => cloneRegionSpec(r, tyArg))

        val newResults: Seq[Result[Attribute]] =
          other.results.map { r =>
            val newTy: Attribute = r.typ match
              case t: TypeAttribute => inst(t, tyArg)
              case a                => a
            val nr = Result(newTy)
            valueMapper += (r: Value[Attribute]) -> (nr: Value[Attribute])
            nr
          }

        other.updated(
          operands = newOperands,
          results = newResults,
          regions = newRegions,
          successors = other.successors,
          properties = other.properties,
          attributes = other.attributes,
        )

  /** Rewrite of one TApply
    *   - clones the TLambda block ops (unattached) under specialization
    *   - inserts cloned ops (except final TReturn) before the TApply
    *   - replaces the TApply result with the cloned version of the returned
    *     value
    *
    * Returns the value that replaces the TApply (for memoization).
    */
  private def rewriteOneTApply(
      ta: TApply,
      tl: TLambda,
  ): Value[TypeAttribute] =
    val origBlock =
      tl.body.blocks.headOption
        .getOrElse {
          sys.error("monomorphize: tlambda has no blocks")
        }

    val origOps = origBlock.operations
    if origOps.isEmpty then
      sys.error("monomorphize: tlambda body block is empty")

    val retVal: Value[TypeAttribute] =
      origOps.last match
        case TReturn(v) => v
        case other      =>
          sys.error(
            s"monomorphize: tlambda terminator must be treturn, got ${other.name}"
          )

    val useBlock =
      ta.containerBlock
        .getOrElse {
          sys.error("monomorphize: tapply has no container block")
        }

    given valueMapper: mutable.Map[Value[Attribute], Value[Attribute]] =
      mutable.Map.empty

    val clonedOpsUnattached: Seq[Operation] =
      origOps.toSeq.dropRight(1).map(op => cloneOpSpec(op, ta.tyArg))

    clonedOpsUnattached.foreach(op => useBlock.insertOpBefore(ta, op))

    val newRetAny =
      valueMapper.getOrElse(
        retVal.asInstanceOf[Value[Attribute]],
        sys
          .error(
            "monomorphize: return value not found in valueMapper (clone bug)"
          ),
      )

    val newRet = newRetAny.asInstanceOf[Value[TypeAttribute]]

    replaceAllUsesWith(
      ta.res.asInstanceOf[Value[Attribute]],
      newRet.asInstanceOf[Value[Attribute]],
    )

    useBlock.eraseOp(ta)

    newRet

  def run(mod: ModuleOp): ModuleOp =
    val cache =
      mutable.Map
        .empty[(Block, Value[tlamForAllType], TypeAttribute), Value[
          TypeAttribute
        ]]

    var changed = true
    while changed do
      changed = false

      val tlByValue = collectTLambdas(mod)
      val tapplies = collectTApplies(mod)

      tapplies.foreach { ta =>
        val blk =
          ta.containerBlock
            .getOrElse(sys.error("monomorphize: tapply has no container block"))

        cache.get((blk, ta.fun, ta.tyArg)) match
          case Some(existing) =>
            replaceAllUsesWith(
              ta.res.asInstanceOf[Value[Attribute]],
              existing.asInstanceOf[Value[Attribute]],
            )
            blk.eraseOp(ta)
            changed = true

          case None =>
            tlByValue.get(ta.fun) match
              case Some(tl) =>
                val repl = rewriteOneTApply(ta, tl)
                cache += (blk, ta.fun, ta.tyArg) -> repl
                changed = true

                if tl.res.uses.isEmpty then RewriteMethods.eraseOp(tl)

              case None =>
                ()
      }
    mod

final class MonomorphizePass(ctx: MLContext) extends ModulePass(ctx):
  override val name: String = "monomorphize"

  override def transform(op: Operation): Operation =
    op match
      case m: ModuleOp => Monomorphize.run(m)
      case other       => other
