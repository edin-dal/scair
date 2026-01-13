package scair.passes

import scair.MLContext
import scair.ir.*
import scair.transformations.{InsertPoint, RewriteMethods}
import scair.transformations.*
import scair.transformations.patterns.*
import scair.dialects.func.*
import scair.dialects.builtin.*
import scair.dialects.tlam_de_bruijn.*

final class LowerTLamToFuncPass(ctx: MLContext) extends ModulePass(ctx):
  override val name = "lower-tlam-to-func"

  override def transform(op: Operation): Operation =
    op match
      case m: ModuleOp =>
        lower(m); m
      case other => other

  private def lower(m: ModuleOp): Unit =
    val lifted = scala.collection.mutable.Map
      .empty[Value[TypeAttribute], SymbolRefAttr]
    val liftedVLambdas = scala.collection.mutable.ArrayBuffer.empty[VLambda]
    var counter = 0
    val top = m.regions.head.blocks.head

    // Phase 1: lift lambdas, DO NOT erase or rewrite returns
    def walkRegion(r: Region): Unit =
      r.blocks.foreach { b =>
        b.operations.foreach { op =>
          op match
            case vl: VLambda =>
              counter += 1
              val name = s"lifted_$counter"
              val (inTy, outTy) = extractFun(vl.funAttr)

              val bodyMoved = vl.body.detached

              val fn = Func(
                sym_name = StringData(name),
                function_type = FunctionType(Seq(inTy), Seq(outTy)),
                sym_visibility = None,
                body = bodyMoved,
              )

              RewriteMethods.insertOpsAt(
                InsertPoint.atStartOf(top),
                fn,
              )

              lifted += (vl.res -> SymbolRefAttr(name))
              liftedVLambdas += vl

            case _ => ()
          op.regions.foreach(walkRegion)
        }
      }

    walkRegion(m.regions.head)

    // Phase 2: rewrite VApply + VReturn (SAFE)
    val p = GreedyRewritePatternApplier(
      Seq(
        pattern { case app: VApply =>
          val sym = lifted.getOrElse(
            app.fun,
            throw new Exception("calling non-lifted lambda"),
          )
          Call(
            callee = sym,
            _operands = Seq(app.arg),
            _results = Seq(Result(app.res.typ)),
          )
        },
        pattern { case vr: VReturn =>
          Return(Seq(vr.value))
        },
      )
    )
    PatternRewriteWalker(p).rewrite(m)

    // Phase 3: erase original VLambda ops (no uses remain)
    liftedVLambdas
      .foreach(vl => if vl.res.uses.isEmpty then RewriteMethods.eraseOp(vl))

def extractFun(t: TypeAttribute): (TypeAttribute, TypeAttribute) =
  t match
    case tlamFunType(in, out) => (in, out)
    case _                    => throw new Exception("expected tlam.fun")

// NOTE (Stage 1 / executability):
// -------------------------------
// This test program only defines a value-level lambda (%f = tlam.vlambda ...)
// after monomorphize + erase-tlam. It does not call it (no tlam.vapply),
// and it has no entry point like func.func @main.
//
// In our current runtime target (func dialect), calls are symbol-based:
//   func.call @callee(args...)
// i.e. func does NOT provide a first-class "function value" representation.
// Therefore, lowering can always lift tlam.vlambda bodies into func.func symbols,
// but it cannot always replace an SSA function value (%f) with an equivalent
// runtime value unless the dialect provides something like func.constant/@sym
// (a function-pointer-like value) or we introduce closure conversion.
//
// Consequence:
// - For Stage 1, it is sufficient that:
//   (1) type-level ops are eliminated (tlam.tlambda/tapply/treturn are gone)
//   (2) lifted func.func exists (so any *call sites* can become func.call)
// - A standalone tlam.vlambda value may remain if there are no call sites,
//   because there is no runtime "function value" in func yet to replace it with.
//
// For actual evaluation/execution tests:
// - We should use a separate test that includes a tlam.vapply call site.
//   In that case, lowering rewrites:
//     tlam.vapply %f %x  ->  func.call @lifted_n(%x)
//   and the original tlam.vlambda becomes dead and can be erased.
