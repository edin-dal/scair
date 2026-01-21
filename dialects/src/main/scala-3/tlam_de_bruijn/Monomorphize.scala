package scair.passes

import scair.ir.*
import scair.transformations.*
import scair.dialects.tlam_de_bruijn.*
import scair.dialects.builtin.*
import scair.MLContext

object Monomorphize:

  /** Substitute the type argument into any TypeAttribute (only our tlam nodes
    * change).
    */
  private def specType(t: TypeAttribute, arg: TypeAttribute): TypeAttribute =
    DBI.subst(0, arg, t)

  /** Specialize a VLambda by substituting the type argument into:
    *   - funAttr (function type)
    *   - the single block argument type
    *   - the VReturn.expected type
    *
    * Assumes prototype shape: exactly one block with one arg and a single
    * VReturn.
    */
  private def specializeVLambda(v: VLambda, arg: TypeAttribute): VLambda =
    // 1) Specialize the function type. We expect it to stay a tlamFunType.
    val newFunTyTA: TypeAttribute = specType(v.res.typ, arg)
    val newFunTy: tlamFunType = newFunTyTA match
      case f: tlamFunType => f
      case other => sys.error(s"expected tlamFunType after spec, got $other")

    // 2) Specialize the single block argument type (if you want to preserve the old block's annotation).
    val oldBlock = v.body.blocks.head
    val oldArgTyTA = oldBlock.arguments.head.typ.asInstanceOf[TypeAttribute]
    val newArgTyTA: TypeAttribute = specType(oldArgTyTA, arg)

    if newArgTyTA != newFunTy.in then
      sys.error(
        s"specializeVLambda: block arg type $newArgTyTA does not match fun input ${newFunTy
            .in}"
      )

    // 3) Rebuild result + body. No expected on VReturn anymore.
    val newRes = Result[tlamFunType](newFunTy)

    val newBody = Region(
      Seq(
        Block(
          newArgTyTA,
          (x: Value[Attribute]) =>
            val xd = x.asInstanceOf[Value[TypeAttribute]]
            Seq(VReturn(xd)), // <-- expected removed
        )
      )
    )

    VLambda(newBody, newRes)

  /** Map from TLambda result values to the defining TLambda op. */
  private def indexTLambdas(
      mod: ModuleOp
  ): Map[Value[TypeAttribute], TLambda] =
    val buf = scala.collection.mutable.Map.empty[Value[TypeAttribute], TLambda]

    def walkRegion(r: Region): Unit =
      r.blocks.foreach { b =>
        b.operations.foreach { op =>
          op match
            case t: TLambda =>
              buf += (t.res: Value[TypeAttribute]) -> t
            case _ => ()
          op.regions.foreach(walkRegion)
        }
      }

    mod.regions.foreach(walkRegion)
    buf.toMap

  /** Replace `tapply` with a specialized vlam (inserted right before it). */
  private def rewriteTApply(
      tapply: TApply,
      tlambda: TLambda,
      argTy: TypeAttribute,
  ): Unit =
    // Grab the vlambda returned by the TLambda (prototype dictates this shape).
    val (vlamReturned: VLambda) =
      val Block(_, ops) = tlambda.body.blocks.head
      ops.collectFirst { case v: VLambda => v }.get

    // Build the specialized vlam
    val specV = specializeVLambda(vlamReturned, argTy)

    // Insert it before the tapply in the same block
    val blk = tapply.containerBlock.get
    blk.insertOpBefore(tapply, specV)

    // Replace the tapply result with the specialized vlam result
    RewriteMethods.replaceOp(tapply, Seq(), Some(Seq(specV.res)))

  /** Optional: try to erase the now-dead TLambda. */
  private def tryEraseDeadTLambda(tlambda: TLambda): Unit =
    try RewriteMethods.eraseOp(tlambda)
    catch case _: Throwable => ()

  /** Run the pass in-place on the module. */
  def run(mod: ModuleOp): ModuleOp =
    val tMap = indexTLambdas(mod)

    // Collect all TApply first
    val applies = scala.collection.mutable.ArrayBuffer.empty[TApply]

    def collectRegion(r: Region): Unit =
      r.blocks.foreach { b =>
        b.operations.foreach { op =>
          op match
            case ta: TApply => applies += ta
            case _          => ()
          op.regions.foreach(collectRegion)
        }
      }

    mod.regions.foreach(collectRegion)

    // Rewrite each TApply we found
    applies.foreach { ta =>
      tMap.get(ta.fun) match
        case Some(tl) =>
          rewriteTApply(
            ta,
            tl,
            ta.tyArg,
          ) // tyArg is now any TypeAttribute (e.g., I32)
          tryEraseDeadTLambda(tl)
        case None =>
          () // out of prototype shape, ignore
    }

    mod

final class MonomorphizePass(ctx: MLContext) extends ModulePass(ctx: MLContext):
  override val name: String = "monomorphize"

  override def transform(op: Operation): Operation =
    op match
      case m: ModuleOp =>
        Monomorphize.run(m)
      case other => other
