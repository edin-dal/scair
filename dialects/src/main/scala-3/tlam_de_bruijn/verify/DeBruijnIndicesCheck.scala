package scair.dialects.tlam_de_bruijn.verify

import scair.ir.*
import scair.utils.{Err, OK}
import scair.dialects.tlam_de_bruijn.*
import scair.dialects.builtin.*
import scala.util.boundary, boundary.break
import scair.verify.VerifierCheck

/** Dialect-specific well-formedness check for DeBruijn indices.
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
          checkType(a.typ.asInstanceOf[TypeAttribute], depth) match
            case e: Err => break(e)
            case _      => ()
        }

        // ops
        b.operations.foreach { op =>
          // operands/results
          op.operands.foreach { v =>
            checkType(v.typ.asInstanceOf[TypeAttribute], depth) match
              case e: Err => break(e)
              case _      => ()
          }
          op.results.foreach { rr =>
            checkType(rr.typ.asInstanceOf[TypeAttribute], depth) match
              case e: Err => break(e)
              case _      => ()
          }

          // binder-aware recursion
          op match
            case tl: TLambda =>
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
