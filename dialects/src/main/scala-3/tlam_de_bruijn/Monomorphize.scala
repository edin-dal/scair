package scair.passes

import scair.ir.*
import scair.transformations.*
import scair.dialects.tlam_de_bruijn.*
import scair.dialects.builtin.*
import scair.MLContext

object Monomorphize:

  private def specType(t: TypeAttribute, arg: TypeAttribute): TypeAttribute =
    DBI.subst(0, arg, t)

  private def specializeVLambda(v: VLambda, arg: TypeAttribute): VLambda =
    val newFunTyTA: TypeAttribute = specType(v.res.typ, arg)
    val newFunTy: tlamFunType = newFunTyTA match
      case f: tlamFunType => f
      case other => sys.error(s"expected tlamFunType after spec, got $other")

    val oldBlock = v.body.blocks.head
    val oldArgTyTA = oldBlock.arguments.head.typ.asInstanceOf[TypeAttribute]
    val newArgTyTA: TypeAttribute = specType(oldArgTyTA, arg)

    if newArgTyTA != newFunTy.in then
      sys.error(
        s"specializeVLambda: block arg type $newArgTyTA does not match fun input ${newFunTy
            .in}"
      )

    val newRes = Result[tlamFunType](newFunTy)

    val newBody = Region(
      Seq(
        Block(
          newArgTyTA,
          (x: Value[Attribute]) =>
            val xd = x.asInstanceOf[Value[TypeAttribute]]
            Seq(VReturn(xd)),
        )
      )
    )

    VLambda(newBody, newRes)

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

  private def rewriteTApply(
      tapply: TApply,
      tlambda: TLambda,
      argTy: TypeAttribute,
  ): Unit =
    val (vlamReturned: VLambda) =
      val Block(_, ops) = tlambda.body.blocks.head
      ops.collectFirst { case v: VLambda => v }.get

    val specV = specializeVLambda(vlamReturned, argTy)

    val blk = tapply.containerBlock.get
    blk.insertOpBefore(tapply, specV)

    RewriteMethods.replaceOp(tapply, Seq(), Some(Seq(specV.res)))

  private def tryEraseDeadTLambda(tlambda: TLambda): Unit =
    try RewriteMethods.eraseOp(tlambda)
    catch case _: Throwable => ()

  def run(mod: ModuleOp): ModuleOp =
    val tMap = indexTLambdas(mod)

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

    applies.foreach { ta =>
      tMap.get(ta.fun) match
        case Some(tl) =>
          rewriteTApply(
            ta,
            tl,
            ta.tyArg,
          )
          tryEraseDeadTLambda(tl)
        case None =>
          ()
    }

    mod

final class MonomorphizePass(ctx: MLContext) extends ModulePass(ctx: MLContext):
  override val name: String = "monomorphize"

  override def transform(op: Operation): Operation =
    op match
      case m: ModuleOp =>
        Monomorphize.run(m)
      case other => other
