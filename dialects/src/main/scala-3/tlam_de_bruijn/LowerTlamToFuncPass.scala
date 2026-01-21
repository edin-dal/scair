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
    var counter = 0
    val top = m.regions.head.blocks.head

    /** Replace all uses of `oldV` with `newV` by rebuilding each user operation
      * and replacing it via RewriteMethods (so parent pointers stay valid).
      */
    def replaceAllUses(oldV: Value[Attribute], newV: Value[Attribute]): Unit =
      val usesSnapshot = oldV.uses.toList
      usesSnapshot.foreach { use =>
        val userOp = use.operation
        val idx = use.index
        val newOperands = userOp.operands.updated(idx, newV)
        val rebuilt = userOp.updated(operands = newOperands)
        RewriteMethods.replaceOp(userOp, rebuilt)
      }

    // ---------------------------
    // Phase 1: lift every VLambda
    //   - create func.func @lifted_n with MOVED body
    //   - create func.constant @lifted_n : builtin.function_type<...>
    //   - replace uses of vl.res with constant result
    //   - erase original VLambda
    // ---------------------------
    def walkRegion(r: Region): Unit =
      r.blocks.foreach { b =>
        // snapshot, because we'll insert/erase while iterating
        val opsSnapshot = b.operations.toList
        opsSnapshot.foreach { op =>
          op match
            case vl: VLambda =>
              counter += 1
              val name = s"lifted_$counter"
              val (inTy, outTy) = extractFun(vl.res.typ)

              // Move body into func.func (safe because we erase vl immediately).
              val bodyMoved: Region = vl.body.detached

              val fn = Func(
                sym_name = StringData(name),
                function_type = FunctionType(Seq(inTy), Seq(outTy)),
                sym_visibility = None,
                body = bodyMoved,
              )

              // Insert the function at module top
              RewriteMethods.insertOpsAt(InsertPoint.atStartOf(top), fn)

              // Materialize a first-class function value
              val fnValTy = FunctionType(Seq(inTy), Seq(outTy))
              val cst = Constant(
                callee = SymbolRefAttr(name),
                res = Result(fnValTy),
              )

              // Insert constant right before the original VLambda
              b.operations.insert(vl, cst)

              // Replace all uses of the lambda value with the constant value
              replaceAllUses(vl.res, cst.res)

              // Erase the original VLambda
              RewriteMethods.eraseOp(vl)

            case _ => ()
          op.regions.foreach(walkRegion)
        }
      }

    walkRegion(m.regions.head)

    // ---------------------------
    // Phase 2: rewrite remaining tlam value-level ops
    //   - vapply -> func.call_indirect
    //   - vreturn -> func.return
    // ---------------------------
    val p = GreedyRewritePatternApplier(
      Seq(
        pattern { case app: VApply =>
          // app.fun is now a Value[tlamFunType] in tlam, but we replaced all
          // VLambda results with func.constant whose result type is FunctionType,
          // which is what call_indirect expects.
          CallIndirect(
            _operands = Seq(app.fun, app.arg),
            _results = Seq(Result(app.res.typ)),
          )
        },
        pattern { case vr: VReturn =>
          Return(Seq(vr.value))
        },
      )
    )

    PatternRewriteWalker(p).rewrite(m)

private def extractFun(t: TypeAttribute): (TypeAttribute, TypeAttribute) =
  t match
    case tlamFunType(in, out) => (in, out)
    case other => throw new Exception(s"expected tlam.fun, got $other")
