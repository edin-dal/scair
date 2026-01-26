package scair.dialects.tlam_de_bruijn.verify

import scair.ir.*
import scair.utils.{Err, OK}
import scair.dialects.tlam_de_bruijn.*
import scair.dialects.builtin.*
import scala.util.boundary, boundary.break
import scair.verify.VerifierCheck

/** Dialect-specific well-formedness check for DeBruijn indices.
  *
  * IMPORTANT: This check must be robust when other dialects are present.
  * Therefore we only inspect types that are actually `TypeAttribute` (and we
  * only recurse into TLam types). Everything else is ignored.
  */
object DeBruijnIndicesCheck extends VerifierCheck:
  override val name: String = "debruijn"

  override def run(root: Operation): OK[Unit] =
    root match
      case m: ModuleOp =>
        walkRegion(m.regions.head, depth = 0)
      case other =>
        Err(s"expected ModuleOp, got `${other.name}`")

  private def walkRegion(r: Region, depth: Int): OK[Unit] =
    boundary:
      r.blocks.foreach { b =>

        // block arguments
        b.arguments.foreach { a =>
          checkMaybeType(a.typ, depth) match
            case e: Err => break(e)
            case _      => ()
        }

        // ops
        b.operations.foreach { op =>
          // operands/results
          op.operands.foreach { v =>
            checkMaybeType(v.typ, depth) match
              case e: Err => break(e)
              case _      => ()
          }
          op.results.foreach { rr =>
            checkMaybeType(rr.typ, depth) match
              case e: Err => break(e)
              case _      => ()
          }

          // binder-aware recursion
          op match
            case tl: TLambda =>
              // tl.res.typ is already a TypeAttribute
              checkType(tl.res.typ, depth) match
                case e: Err => break(e)
                case _      => ()

              // TLambda introduces one binder
              walkRegion(tl.body, depth + 1) match
                case e: Err => break(e)
                case _      => ()

            case ta: TApply =>
              checkType(ta.tyArg, depth) match
                case e: Err => break(e)
                case _      => ()

            case vl: VLambda =>
              checkType(vl.res.typ, depth) match
                case e: Err => break(e)
                case _      => ()

            case _ => ()

          // generic recursion (avoid double-walk TLambda body)
          op match
            case _: TLambda => ()
            case _          =>
              op.regions.foreach { rr =>
                walkRegion(rr, depth) match
                  case e: Err => break(e)
                  case _      => ()
              }
        }
      }
      OK(())

  /** Only check attributes that are actually types. Never cast. */
  private def checkMaybeType(a: Attribute, depth: Int): OK[Unit] =
    a match
      case t: tlamType => checkType(t, depth)
      case _           => OK(())

  private def checkType(t: TypeAttribute, depth: Int): OK[Unit] =
    t match
      case tlamBVarType(IntegerAttr(k, _)) =>
        if k.data >= 0 && k.data < depth then OK(())
        else Err(s"bvar<$k> out of scope at depth=$depth")

      case tlamFunType(in, out) =>
        checkType(in, depth) match
          case e: Err => e
          case _      => checkType(out, depth)

      case tlamForAllType(body) =>
        checkType(body, depth + 1)

      case _ =>
        OK(())
