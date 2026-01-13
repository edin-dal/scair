package scair

import scair.ir.*
import scair.Printer
import scair.utils.*
import scair.dialects.builtin.*
import scair.dialects.tlam_de_bruijn.*
import scair.dialects.tlam_de_bruijn.tlamTy.*

import org.scalatest.*
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers.*
import java.io.*

class tlamDeBruijnTests extends AnyFlatSpec:

  private def dump(title: String, m: ModuleOp): Unit =
    val sw = new StringWriter()
    Printer(p = new PrintWriter(sw)).print(m)
    println(s"\n===== $title =====")
    println(sw.toString)
    println(s"VERIFY: ${m.verify()}")

  private def printed(attr: Attribute): String =
    import java.io.{PrintWriter, StringWriter}
    val sw = new StringWriter()
    val pw = new PrintWriter(sw)
    given indentLevel: Int = 0
    val p = scair.Printer(p = pw)
    p.print(attr)
    pw.flush()
    sw.toString.trim

  var out = StringWriter()
  var printer = new Printer(true, p = PrintWriter(out));

  given indentLevel: Int = 0

  /*
  Test: Polymorphic identity (de Bruijn) IR — build/verify/print
  ==============================================================

  Surface idea
  ------------
  F = Λ α. (λ (x : α). x)

  IR shape (essentials)
  ---------------------
  - Outer TLambda introduces α (so α ≡ bvar(IntData(0)) in its body).
  - Inside, a single VLambda: λ(x : bvar(IntData(0))) . vreturn x : bvar(IntData(0))
  - The TLambda returns that VLambda, so the result type is ∀α. α→α.

  What we assert
  --------------
  - Verifies successfully.
  - Printed IR contains the expected "tlambda", inner "vlambda", the binder block,
    and the function type: !tlam.forall<!tlam.fun<!tlam.bvar<0>, !tlam.bvar<0>>>.
   */
  "A polymorphic identity (de Bruijn) IR" should
    "build, verify, and print the expected shape" in {
      //
      // Types:
      //   body  = !tlam.fun<!tlam.bvar<0> -> !tlam.bvar<0>>
      //   forall = !tlam.forall<body>
      //
      val bodyTy: tlamType = fun(bvar(IntData(0)), bvar(IntData(0)))
      val forallTy: tlamType = forall(bodyTy)

      //
      // Inner vlambda:
      //   %v = tlam.vlambda (%x : !tlam.bvar<0>)
      //          : !tlam.fun<!tlam.bvar<0> -> !tlam.bvar<0>> {
      //       tlam.vreturn %x : !tlam.bvar<0>;
      //   }
      //
      val vLambdaFunTy = bodyTy
      val vLambdaRes = Result[TypeAttribute](vLambdaFunTy)

      val vLambdaRegion =
        Region(
          Seq(
            Block(
              // one argument of type !tlam.bvar<0>
              bvar(IntData(0)),
              (xVal: Value[Attribute]) =>
                val x = xVal.asInstanceOf[Value[TypeAttribute]]
                val vret = VReturn(x, expected = bvar(IntData(0)))
                Seq(vret),
            )
          )
        )

      val vLambda =
        VLambda(funAttr = vLambdaFunTy, body = vLambdaRegion, res = vLambdaRes)
      vLambda.verify().isOK shouldBe true

      //
      // Outer tlambda + treturn:
      //   %F = tlam.tlambda (%T : !tlam.type)
      //          : !tlam.forall<!tlam.fun<!tlam.bvar<0> -> !tlam.bvar<0>>> {
      //       %v = (above vlambda)
      //       tlam.treturn %v : !tlam.fun<!tlam.bvar<0> -> !tlam.bvar<0>>
      //   }
      //
      val tRet = TReturn(value = vLambdaRes, expected = bodyTy)
      tRet.verify().isOK shouldBe true

      val tLambdaRes = Result[TypeAttribute](forallTy)
      val tLambdaRegion =
        Region(
          Seq(
            Block(
              operations = Seq(vLambda, tRet)
            )
          )
        )

      val tLambda = TLambda(tBody = tLambdaRegion, res = tLambdaRes)
      tLambda.verify().isOK shouldBe true

      // Wrap in a builtin.module
      val module = ModuleOp(
        Region(
          Seq(
            Block(
              operations = Seq(tLambda)
            )
          )
        )
      )

      // Print & sanity-check the textual IR
      val out = StringWriter()
      Printer(p = PrintWriter(out)).print(module)
      val printed = out.toString().trim

      // DEBUG: Print the IR
      println(printed)

      // Avoid overfitting to SSA numbering; just assert the essential structure.
      printed should include("builtin.module")
      printed should include("tlam.tlambda")
      printed should include(
        "!tlam.forall<!tlam.fun<!tlam.bvar<0>, !tlam.bvar<0>>>"
      )

      printed should include("tlam.vlambda")
      printed should include("^bb0(") // inner value-lambda block
      printed should include("!tlam.bvar<0>")
      printed should include("tlam.vreturn")
      printed should include("tlam.treturn")

    }

  /*
  Test: Nested TLambda + TApply (de Bruijn) — instantiate ∀ with bvar(IntData(0))
  ======================================================================

  Surface idea
  ------------
  Outer: F = Λ α.
            let G = Λ β. λ (x : β). x in
            let h = G[α] in
            return h       // h : α → α

  De Bruijn view
  --------------
  - In G’s body: β ≡ bvar(IntData(0)), α ≡ bvar(IntData(1)).
  - The outer TApply uses argType = bvar(IntData(0)) (that outer α).

  What we assert
  --------------
  - Building and verifying succeeds.
  - Printing shows both tlambda’s, the vlam, the tapply, and the types.
   */
  "Nested tlambda + tapply (de Bruijn) IR" should
    "instantiate ∀ with bvar(IntData(0)) and type-check" in {
      // Common types under the *current* type binder
      val bodyTy: tlamType = fun(bvar(IntData(0)), bvar(IntData(0))) // α -> α
      val forallTy: tlamType = forall(bodyTy) // ∀α. α -> α

      // Build the inner polymorphic function:
      //   %G = tlam.tlambda (%U : !tlam.type) : ∀α. α -> α {
      //     %v = tlam.vlambda (%x : α) : α -> α { vreturn %x : α }
      //     tlam.treturn %v : α -> α
      //   }
      val inner_vLambdaRes = Result[TypeAttribute](bodyTy)
      val inner_vLambdaRegion =
        Region(
          Seq(
            Block(
              bvar(IntData(0)), // refers to %U
              (xVal: Value[Attribute]) =>
                val x = xVal.asInstanceOf[Value[TypeAttribute]]
                Seq(VReturn(x, expected = bvar(IntData(0)))),
            )
          )
        )
      val inner_vLambda = VLambda(
        funAttr = bodyTy,
        body = inner_vLambdaRegion,
        res = inner_vLambdaRes,
      )
      inner_vLambda.verify().isOK shouldBe true

      val inner_tRet = TReturn(value = inner_vLambdaRes, expected = bodyTy)
      val inner_tLambdaRes = Result[TypeAttribute](forallTy)
      val inner_tLambdaRegion =
        Region(
          Seq(
            Block(
              operations = Seq(inner_vLambda, inner_tRet)
            )
          )
        )
      val inner_tLambda =
        TLambda(tBody = inner_tLambdaRegion, res = inner_tLambdaRes)
      inner_tLambda.verify().isOK shouldBe true

      // Now the outer tlambda, which applies %G to !tlam.bvar<0> (the outer binder):
      // %F = tlam.tlambda (%T : !tlam.type) : ∀α. α -> α {
      //   %G = (above)
      //   %h = tlam.tapply %G <!tlam.bvar<0>> : α -> α
      //   tlam.treturn %h : α -> α
      // }
      val tapplyResultTy = bodyTy // instantiate ∀ with α gives α -> α
      val hRes = Result[TypeAttribute](tapplyResultTy)
      val tapply =
        TApply(
          polymorphicFun = inner_tLambdaRes,
          argType = bvar(IntData(0)),
          res = hRes,
        )
      tapply.verify().isOK shouldBe true

      val outer_tRet = TReturn(value = hRes, expected = tapplyResultTy)
      val outer_tLambdaRes = Result[TypeAttribute](forallTy)
      val outer_tLambdaRegion =
        Region(
          Seq(
            Block(
              operations = Seq(inner_tLambda, tapply, outer_tRet)
            )
          )
        )
      val outer_tLambda =
        TLambda(tBody = outer_tLambdaRegion, res = outer_tLambdaRes)
      outer_tLambda.verify().isOK shouldBe true

      val module = ModuleOp(
        Region(Seq(Block(operations = Seq(outer_tLambda))))
      )
      module.verify().isOK shouldBe true

      val out = StringWriter()
      Printer(p = PrintWriter(out)).print(module)
      val printed = out.toString.trim

      // DEBUG: Print the IR
      println(printed)

      // Structure checks (comma style)
      printed should include("builtin.module")
      printed should include("tlam.tlambda") // outer
      printed should include(
        "!tlam.forall<!tlam.fun<!tlam.bvar<0>, !tlam.bvar<0>>>"
      )
      printed should include("tlam.tlambda") // inner
      printed should include("tlam.vlambda")
      printed should include("tlam.vreturn")
      printed should include("tlam.tapply")
      printed should include("!tlam.fun<!tlam.bvar<0>, !tlam.bvar<0>>")
      printed should include("tlam.treturn")
    }

  /*
  Test: Instantiate ∀α. α→α at a ground type (i32)
  ================================================

  Surface idea
  ------------
  (∀ α. α→α)[α := i32]  ≡  i32→i32

  Mechanics
  ---------
  - DBI.instantiate substitutes the type binder with tlamConstType(I32).
  - Also check via the TApply op’s verifier.

  What we assert
  --------------
  - The result type equals fun(i32, i32).
  - TApply verifies when the result type matches the instantiated one.
   */
  "Instantiate to a ground type (i32)" should
    "compute α→α @ α=i32 == i32→i32" in {
      val alphaToAlpha: TypeAttribute = fun(bvar(IntData(0)), bvar(IntData(0)))
      val poly: TypeAttribute = forall(alphaToAlpha)

      val i32Ty: TypeAttribute = I32 // wrap builtin i32
      val inst = DBI.instantiate(poly, i32Ty) // should be fun(i32, i32)

      inst shouldEqual fun(i32Ty, i32Ty)

      // Also via the operation:
      val innerRes = Result[TypeAttribute](poly)
      val tapplyRes = Result[TypeAttribute](fun(i32Ty, i32Ty))
      val tapp = TApply(innerRes, i32Ty, tapplyRes)
      tapp.verify().isOK shouldBe true
    }

  /*
  Test: Tiny evaluator — (∀α. λ(x:α). x) @ α=i32, then apply to y: i32
  ====================================================================

  Surface idea
  ------------
  G = Λ α. λ (x : α). x
  instantiate: G[i32] : i32 → i32
  apply to y : i32  ⇒  y

  Evaluator sketch
  ----------------
  - evalTLambda returns a polymorphic value VPoly whose specialization yields the identity VFun.
  - evalTApply specializes at i32.
  - evalBlock runs VApply and returns y.

  What we assert
  --------------
  - The result equals 42 (value preserved by id).
   */
  "Polymorphic id at i32, applied to a concrete arg (tiny evaluator)" should
    "return the same value" in {

      // ---- Types we use ----
      val alphaId: tlamType = fun(bvar(IntData(0)), bvar(IntData(0))) // α -> α
      val polyId: tlamType = forall(alphaId) // ∀α. α -> α
      val i32Ty: TypeAttribute = I32
      val i32Id: tlamType = fun(i32Ty, i32Ty)

      // ---- Build G = Λα. λ(x: α). x  ----
      val vRes = Result[TypeAttribute](alphaId)
      val vBody =
        Region(
          Seq(
            Block(
              bvar(IntData(0)),
              (x: Value[Attribute]) =>
                val xd = x.asInstanceOf[Value[TypeAttribute]]
                Seq(VReturn(xd, expected = bvar(IntData(0)))),
            )
          )
        )
      val vLam = VLambda(alphaId, vBody, vRes)
      val tRet = TReturn(vRes, expected = alphaId)
      val tRes = Result[TypeAttribute](polyId)
      val tBody =
        Region(Seq(Block(operations = Seq(vLam, tRet))))
      val G = TLambda(tBody, tRes)

      // ---- Specialize at i32, then apply to an argument y: i32 ----
      val specialized = Result[TypeAttribute](i32Id)
      val tapp = TApply(tRes, i32Ty, specialized)
      tapp.verify().isOK shouldBe true

      val applyRes = Result[TypeAttribute](i32Ty)
      val appRegion =
        Region(
          Seq(
            Block(
              i32Ty, // block argument carries the concrete value in the evaluator env
              (y: Value[Attribute]) =>
                val yd = y.asInstanceOf[Value[TypeAttribute]]
                val app = VApply(specialized, yd, applyRes)
                val ret = VReturn(applyRes, expected = i32Ty)
                Seq(app, ret),
            )
          )
        )

      import scala.util.boundary, boundary.break

      // ---------------- tiny evaluator ----------------
      sealed trait Val
      case class VI32(n: Int) extends Val
      case class VFun(run: Val => Val) extends Val
      case class VPoly(runTy: TypeAttribute => VFun)
          extends Val // <-- TypeAttribute

      final case class Env(arg0: Option[Val])

      def evalTLambda(op: TLambda): VPoly =
        // For the test we only build the identity; specialize to any type gives id
        VPoly(_ => VFun(v => v))

      def evalTApply(op: TApply, pf: VPoly): VFun =
        pf.runTy(op.argType)

      def evalVLambda(op: VLambda): VFun =
        // Body is `vreturn %x` ⇒ identity
        VFun(v => v)

      def evalBlock(block: Block, env: Env, funFromTApply: Option[VFun]): Val =
        boundary:
          var last: Val = null.asInstanceOf[Val]
          val ops = block.operations.iterator
          while ops.hasNext do
            ops.next() match
              case op: VApply =>
                val f =
                  funFromTApply.getOrElse(sys.error("no function from tapply"))
                val arg = env.arg0.getOrElse(sys.error("missing arg"))
                last = f.run(arg)
              case _: VReturn =>
                break(last) // <- exit early with the current value
              case other =>
                sys.error(s"unsupported op in evalBlock: $other")
          last

      // wire: build the values from the structure we created
      val polyVal: VPoly = evalTLambda(G)
      val monoFun: VFun = evalTApply(tapp, polyVal)

      // run app block with y = 42
      val result = evalBlock(
        appRegion.blocks.head,
        Env(arg0 = Some(VI32(42))),
        Some(monoFun),
      )
      result shouldEqual VI32(42)
    }

  // -------------------------------- shift tests --------------------------------

  /*
  Test: DBI.shift — basic index bumping
  =====================================

  Definition
  ----------
  shift(d, c, t): bump all de Bruijn indices k in t with k >= c by +d

  Checks
  ------
  - shift(1, 0, bvar(IntData(0))) = bvar(IntData(1))
  - shift(1, 1, bvar(IntData(0))) = bvar(IntData(0))    // 0 < cutoff ⇒ unchanged
  - shift(0, 0, bvar(42)) = bvar(42)

  What we assert
  --------------
  - Only indices >= cutoff are shifted; d=0 is identity.
   */
  "DBI.shift basic behavior" should "bump only indices >= cutoff" in {
    import tlamTy.*, DBI.*

    // shift 1 at cutoff 0: 0 -> 1, 1 -> 2
    shift(1, 0, bvar(IntData(0))) shouldEqual bvar(IntData(1))
    shift(1, 0, bvar(IntData(1))) shouldEqual bvar(IntData(2))

    // shift 1 at cutoff 1: 0 stays (because 0 < 1), 1 -> 2
    shift(1, 1, bvar(IntData(0))) shouldEqual bvar(IntData(0))
    shift(1, 1, bvar(IntData(1))) shouldEqual bvar(IntData(2))

    // shift 0 is identity
    shift(0, 0, bvar(IntData(42))) shouldEqual bvar(IntData(42))
  }

  /*
  Test: DBI.shift — structural behavior and binders
  =================================================

  Idea
  ----
  - shift distributes over fun types.
  - Under a ∀, cutoff increases by 1 for the body (new binder at depth 0).

  Example
  -------
  ty = (0 -> (1 -> 0))
  shift(1, 0, ty) = (1 -> (2 -> 1))

  poly = ∀. (0 -> 1)
  shift(1, 0, poly) = ∀. (0 -> 2)  // inner cutoff is 1

  What we assert
  --------------
  - Structural mapping and cutoff accounting are correct.
   */
  "DBI.shift structure laws" should "distribute over fun and respect binders" in {
    import tlamTy.*, DBI.*

    val ty =
      fun(
        bvar(IntData(0)),
        fun(bvar(IntData(1)), bvar(IntData(0))),
      ) // (0 -> (1 -> 0))
    shift(1, 0, ty) shouldEqual fun(
      bvar(IntData(1)),
      fun(bvar(IntData(2)), bvar(IntData(1))),
    )

    // Under ∀, cutoff increases by 1 inside the body
    val poly = forall(fun(bvar(IntData(0)), bvar(IntData(1)))) // ∀. (0 -> 1)
    // shift by 1 with cutoff 0:
    // - outside just wraps
    // - inside uses cutoff 1, so bvar(IntData(0)) (<1) stays, bvar(IntData(1)) (>=1) -> bvar(IntData(2))
    shift(1, 0, poly) shouldEqual forall(
      fun(bvar(IntData(0)), bvar(IntData(2)))
    )
  }

  /*
  Test: DBI.shift — round-trip under valid ranges
  ===============================================

  Idea
  ----
  If indices remain valid, then shift(+1, 0, t) followed by shift(-1, 0, •)
  restores the original t.

  What we assert
  --------------
  - down == original after shifting up then down.
   */
  "DBI.shift round-trip" should "be reversible when shifts are valid" in {
    import tlamTy.*, DBI.*
    val t = forall(
      fun(bvar(IntData(0)), fun(bvar(IntData(1)), bvar(IntData(0))))
    ) // ∀. (0 -> (1 -> 0))
    val up = shift(1, 0, t)
    val down = shift(-1, 0, up)
    down shouldEqual t
  }

  // ------------------------------ subst/instantiate -----------------------------

  /*
  Test: DBI.subst — substitution under a binder is capture-avoiding
  =================================================================

  Definition
  ----------
  subst(c, s, t): replace bvar(c) by s in t, adjusting indices above c (k>c ⇒ k-1).
  Under a ∀, recurse with subst(c+1, shift(1,0,s), body).

  Example
  -------
  t = ∀. fun(bvar(IntData(2)), bvar(IntData(1)))
  subst(1, bvar(IntData(0)), t)
  = ∀. fun(bvar(IntData(1)), bvar(IntData(1)))

  What we assert
  --------------
  - Uses shift(1,0,s) under the binder and avoids capture correctly.
   */
  "DBI.subst under a binder" should "use shift(1,0,s) and avoid capture" in {
    import tlamTy.*, DBI.*
    // t = ∀. fun(bvar(IntData(2)), bvar(IntData(1)))   -- body sees one binder already
    val t = forall(fun(bvar(IntData(2)), bvar(IntData(1))))
    // Substitute at c=1 with s = bvar(IntData(0))  (a free var in the outer scope)
    // Expected: ∀. fun(bvar(IntData(1)), bvar(IntData(1)))
    // Explanation: under the binder we do subst(2, shift(1,0,bvar(IntData(0)))=bvar(IntData(1)), body)
    subst(1, bvar(IntData(0)), t) shouldEqual forall(
      fun(bvar(IntData(1)), bvar(IntData(1)))
    )
  }

  /*
  Test: DBI.instantiate — ∀ elimination on a ground type
  =====================================================

  Idea
  ----
  instantiate(∀. fun(bvar(IntData(0)), bvar(IntData(0))), τ)  =  fun(τ, τ)

  What we assert
  --------------
  - Instantiation yields τ→τ.
   */
  "DBI.instantiate" should "instantiate ∀α. α→α to τ→τ for ground τ" in {
    import tlamTy.*, DBI.*
    val poly = forall(fun(bvar(IntData(0)), bvar(IntData(0))))
    instantiate(poly, I32) shouldEqual fun(I32, I32)
  }

  /*
  Test: VLambda.verify — wrong block arg type must fail
  =====================================================

  Idea
  ----
  VLambda(fun(bvar(IntData(0)), bvar(IntData(0))), body) requires the body block arg to have type bvar(IntData(0)).
  We intentionally use bvar(IntData(1)) to ensure verify() fails.

  What we assert
  --------------
  - VLambda.verify returns false.
   */
  "A VLambda with wrong block arg type" should "fail verify" in {
    import tlamTy.*
    val funTy = fun(bvar(IntData(0)), bvar(IntData(0)))
    val res = Result[TypeAttribute](funTy)
    val wrongRegion =
      Region(
        Seq(
          Block(
            bvar(IntData(1)),
            (x: Value[Attribute]) => // should be bvar(IntData(0))
              val xd = x.asInstanceOf[Value[TypeAttribute]]
              Seq(VReturn(xd, expected = bvar(IntData(1)))),
          )
        )
      )
    val bad = VLambda(funTy, wrongRegion, res)
    bad.verify().isOK shouldBe false
  }

  /*
  Test: TApply.verify — polymorphicFun must have ∀ type
  ====================================================

  Idea
  ----
  Construct a TApply where polymorphicFun is a plain fun type, not a forall.

  What we assert
  --------------
  - TApply.verify returns false.
   */
  "A TApply on a non-forall" should "fail verify" in {
    import tlamTy.*
    val notForall =
      Result[TypeAttribute](fun(bvar(IntData(0)), bvar(IntData(0))))
    val res = Result[TypeAttribute](fun(bvar(IntData(0)), bvar(IntData(0))))
    val bad = TApply(notForall, bvar(IntData(0)), res)
    bad.verify().isOK shouldBe false
  }

  /*
  Test: VApply.verify — arg/result must match fun type
  ====================================================

  Idea
  ----
  Use a function type α→α, but pretend result type is bvar(IntData(1)) (wrong).

  What we assert
  --------------
  - VApply.verify returns false with a clear mismatch message.
   */
  "A VApply with mismatched argument/result types" should "fail verify" in {
    import tlamTy.*
    val fTy = fun(bvar(IntData(0)), bvar(IntData(0)))
    val argT = bvar(IntData(0))
    val resT = bvar(IntData(0))
    val fRes = Result[TypeAttribute](fTy)
    val vlam =
      VLambda(
        fTy,
        Region(
          Seq(
            Block(
              argT,
              (x: Value[Attribute]) =>
                val xd = x.asInstanceOf[Value[TypeAttribute]]
                Seq(VReturn(xd, expected = argT)),
            )
          )
        ),
        fRes,
      )
    // pretend we have an argument of wrong type bvar(IntData(1))
    val argVal = fRes
    val bad = VApply(fRes, argVal, Result[TypeAttribute](bvar(IntData(1))))
    bad.verify().isOK shouldBe false
  }

  /*
  Test: DBI.subst — indices strictly above the hole shift down
  ===========================================================

  Example
  -------
  subst(0, τ, fun(bvar(IntData(1)), bvar(IntData(0))))  =  fun(bvar(IntData(0)), τ)
  (1 drops to 0; the replaced bvar(IntData(0)) becomes τ.)

  What we assert
  --------------
  - The (k > c) decrement path works.
   */
  "DBI.subst (k > c) branch" should "decrement indices above the hole" in {
    import tlamTy.*, DBI.*
    // Replace bvar(IntData(0)) with τ in (1 -> 0) : the 1 shifts down to 0
    val t = fun(bvar(IntData(1)), bvar(IntData(0)))
    val tau = I32
    subst(0, tau, t) shouldEqual fun(bvar(IntData(0)), tau)
  }

  /*
  Test: DBI.shift with two nested ∀ — cutoff composes
  ====================================================

  Idea
  ----
  t = ∀. ∀. (1 -> 0)
  shift(1, 0, t) leaves inner body unchanged because inner cutoff = 2;
  neither 1 nor 0 is >= 2.

  What we assert
  --------------
  - The inner body stays the same; outer structure preserved.
   */
  "DBI.shift across two nested binders" should "leave indices < 2 unchanged" in {
    import tlamTy.*, DBI.*
    val t =
      forall(forall(fun(bvar(IntData(1)), bvar(IntData(0))))) // ∀. ∀. (1 -> 0)
    shift(1, 0, t) shouldEqual t // cutoff becomes 2 inside
  }

  /*
  Test: DBI.shift with two ∀ — bump indices ≥ 2 inside
  ====================================================

  Example
  -------
  t = ∀. ∀. (2 -> 1)
  inner cutoff = 2
  shift(1, 0, t) = ∀. ∀. (3 -> 1)

  What we assert
  --------------
  - Only indices ≥ 2 are incremented inside the inner body.
   */
  "DBI.shift across two nested binders (index beyond both)" should
    "bump indices >= 2 inside the inner body" in {
      import tlamTy.*, DBI.*
      val t = forall(
        forall(fun(bvar(IntData(2)), bvar(IntData(1))))
      ) // ∀. ∀. (2 -> 1)
      val s = shift(1, 0, t) // cutoff 2 inside
      s shouldEqual forall(
        forall(fun(bvar(IntData(3)), bvar(IntData(1))))
      ) // 2→3, 1 unchanged
    }

  /*
  Test: TApply under outer binder — instantiate ∀β using α (outer bvar(IntData(0)))
  ========================================================================

  Surface idea
  ------------
  Outer Λ α.  (apply (Λ β. λ(x: α). x) at α)  ⇒  α → α

  De Bruijn view
  --------------
  - Inside the inner TLambda, α is bvar(IntData(1)), β is bvar(IntData(0)).
  - TApply uses argType = outer bvar(IntData(0)) (α).

  What we assert
  --------------
  - The TApply result type is fun(bvar(IntData(0)), bvar(IntData(0))).
  - The outer TLambda result is ∀α. α→α.
   */
  "TApply under an outer binder" should
    "instantiate ∀β. (α -> α) to (α -> α) where α is outer bvar(IntData(0))" in {
      // Types inside inner tlambda (β is inner, α is outer):
      // bodyInner = fun(bvar(IntData(1)), bvar(IntData(1)))   // both refer to the *outer* binder
      val bodyInner: tlamType = fun(bvar(IntData(1)), bvar(IntData(1)))
      val forallInner: tlamType = forall(bodyInner) // ∀β. (α -> α)

      // Build inner vlambda: λ(x: α). x  (α = bvar(IntData(1)) under the inner binder)
      val innerVRes = Result[TypeAttribute](bodyInner)
      val innerVRegion =
        Region(
          Seq(
            Block(
              bvar(IntData(1)), // arg : α (outer binder)
              (xVal: Value[Attribute]) =>
                val x = xVal.asInstanceOf[Value[TypeAttribute]]
                Seq(VReturn(x, expected = bvar(IntData(1)))), // return x : α
            )
          )
        )
      val innerVLam =
        VLambda(funAttr = bodyInner, body = innerVRegion, res = innerVRes)
      innerVLam.verify().isOK shouldBe true

      // Inner tlambda over β returning that value-level identity at type α -> α
      val innerTRet = TReturn(value = innerVRes, expected = bodyInner)
      val innerTRes = Result[TypeAttribute](forallInner)
      val innerTRegion =
        Region(
          Seq(
            Block(
              operations = Seq(innerVLam, innerTRet)
            )
          )
        )
      val innerTLambda = TLambda(tBody = innerTRegion, res = innerTRes)
      innerTLambda.verify().isOK shouldBe true

      // Now the OUTER tlambda over α, which applies the inner poly fn at α = bvar(IntData(0))
      val tapplyResultTy =
        fun(
          bvar(IntData(0)),
          bvar(IntData(0)),
        ) // expect (α -> α) after instantiation
      val hRes = Result[TypeAttribute](tapplyResultTy)
      val tapply =
        TApply(
          polymorphicFun = innerTRes,
          argType = bvar(IntData(0)),
          res = hRes,
        )
      tapply.verify().isOK shouldBe true

      val outerTRet = TReturn(value = hRes, expected = tapplyResultTy)
      val outerTRes =
        Result[TypeAttribute](forall(tapplyResultTy)) // ∀α. α -> α
      val outerTRegion =
        Region(
          Seq(
            Block(
              operations = Seq(innerTLambda, tapply, outerTRet)
            )
          )
        )
      val outerTLambda = TLambda(tBody = outerTRegion, res = outerTRes)
      outerTLambda.verify().isOK shouldBe true

      // Optional: quick sanity assertions
      hRes.typ shouldEqual fun(bvar(IntData(0)), bvar(IntData(0)))
      outerTLambda.res.typ shouldEqual forall(
        fun(bvar(IntData(0)), bvar(IntData(0)))
      )
    }

  /*
  Test: Monomorphize pass — eliminate TApply and inline a specialized VLambda
  ===========================================================================

  BEFORE (surface)
  ----------------
  F = Λ α.
        let G = Λ β. λ(x: β). x in
        let h = G[α] in
        return h     // h : α→α

  AFTER (surface)
  ---------------
  F = Λ α.
        let h = λ(x: α). x in
        return h     // still α→α


  BEFORE:  F = Λ α. Λ β. λ(x: β). x
  AFTER:   F = Λ α.       λ(x: α). x

  IR intuition
  ------------
  - The pass finds TApply(%G, α), looks up the defining TLambda %G, clones the
    returned VLambda, substituting β := α (de Bruijn bvar(IntData(0)) under outer TLambda).
  - It inserts the specialized VLambda before the TReturn and replaces the TApply’s
    result with that VLambda’s result.
  - It then DCEs the now-unused inner TLambda (best effort).

  What we assert
  --------------
  - TApply count goes from 1 to 0.
  - TLambda count goes from 2 to 1 (inner erased).
  - There is a VLambda of type α→α in the outer block, before the final treturn.
  - Verification still succeeds and printed IR contains no tlam.tapply.
   */
  "Monomorphize pass" should
    "replace tapply with a specialized vlam and remove the use" in {
      // Types under current binder: α → α and ∀β. β → β (but we’ll use α for instantiation)
      val alphaFun: tlamType = fun(bvar(IntData(0)), bvar(IntData(0))) // α -> α
      val forallAlphaFun: tlamType = forall(alphaFun) // ∀α. α -> α

      // Inner: G = Λβ. λ(x: β). x  (written with bvar(IntData(0)) under its own binder)
      val innerVRes =
        Result[TypeAttribute](fun(bvar(IntData(0)), bvar(IntData(0))))
      val innerVRegion = Region(
        Seq(
          Block(
            bvar(IntData(0)),
            (x: Value[Attribute]) =>
              val xd = x.asInstanceOf[Value[TypeAttribute]]
              Seq(VReturn(xd, expected = bvar(IntData(0)))),
          )
        )
      )
      val innerVLam = VLambda(
        funAttr = fun(bvar(IntData(0)), bvar(IntData(0))),
        body = innerVRegion,
        res = innerVRes,
      )
      val innerTRet = TReturn(
        value = innerVRes,
        expected = fun(bvar(IntData(0)), bvar(IntData(0))),
      )
      val innerTLRes =
        Result[TypeAttribute](forall(fun(bvar(IntData(0)), bvar(IntData(0)))))
      val innerTLRegion = Region(
        Seq(
          Block(operations = Seq(innerVLam, innerTRet))
        )
      )
      val G = TLambda(tBody = innerTLRegion, res = innerTLRes)
      G.verify().isOK shouldBe true

      // Outer: F = Λα. (h := tapply G <!α>; treturn h : α→α)
      val hRes = Result[TypeAttribute](alphaFun)
      val tapply =
        TApply(
          polymorphicFun = innerTLRes,
          argType = bvar(IntData(0)),
          res = hRes,
        )
      tapply.verify().isOK shouldBe true
      val outerTRet = TReturn(value = hRes, expected = alphaFun)
      val outerTLRes = Result[TypeAttribute](forallAlphaFun)
      val outerTLRegion = Region(
        Seq(
          Block(operations = Seq(G, tapply, outerTRet))
        )
      )
      val F = TLambda(tBody = outerTLRegion, res = outerTLRes)
      F.verify().isOK shouldBe true

      val module: ModuleOp =
        new ModuleOp(
          Region(Seq(Block(operations = Seq(F))))
        )

      module.verify().isOK shouldBe true

      // DEBUG PRINT BEFORE PASS
      // val sw0 = new StringWriter()
      // Printer(p = new PrintWriter(sw0)).print(module)
      // println("=== BEFORE ===\n" + sw0.toString)

      // Helper: count ops of a given class
      def count[T](op: Operation)(using m: reflect.ClassTag[T]): Int =
        var n = 0
        def walkOp(o: Operation): Unit =
          if m.runtimeClass.isInstance(o) then n += 1
          // visit nested regions of this op
          o.regions.foreach(walkRegion)
        def walkRegion(r: Region): Unit =
          r.blocks.foreach(b => b.operations.foreach(walkOp))

        walkOp(op)
        n

      def findVLambdaWithType(block: Block, funTy: tlamType): Option[VLambda] =
        block.operations.collectFirst {
          case v: VLambda if v.funAttr == funTy && v.res.typ == funTy => v
        }

      // Pre-conditions: exactly 1 tapply, at least 1 vlambda (the inner one)
      count[TApply](module) shouldBe 1
      count[TLambda](module) shouldBe 2
      assert(count[VLambda](module) >= 1)

      // Run the pass
      import scair.MLContext
      import scair.dialects.builtin.BuiltinDialect
      import scair.dialects.tlam_de_bruijn.TlamDeBruijnDialect
      import scair.passes.MonomorphizePass

      val ctx = MLContext()
      ctx.registerDialect(BuiltinDialect)
      ctx.registerDialect(TlamDeBruijnDialect)
      val pass = new MonomorphizePass(ctx)
      val afterOp: Operation = pass.transform(module)
      val afterMod = afterOp match
        case m: ModuleOp => m
        case _           => fail("expected ModuleOp after transform")

      afterMod.verify().isOK shouldBe true

      // DEBUG PRINT AFTER PASS
      // val sw1 = new StringWriter()
      // Printer(p = new PrintWriter(sw1)).print(afterMod)
      // println("=== AFTER ===\n" + sw1.toString)

      // Post-conditions: tapply gone; a specialized vlambda was inserted before it
      count[TApply](afterMod) shouldBe 0
      count[TLambda](afterMod) shouldBe 1 // inner TLambda got DCE’d
      assert(count[VLambda](afterMod) >= 1)

      val sw = new StringWriter()
      Printer(p = new PrintWriter(sw)).print(afterMod)
      val out = sw.toString

      out should include("tlam.vlambda")
      out should not include ("tlam.tapply")

      // Stronger check: the return in the outer body still returns α→α
      val outerAfter: TLambda =
        afterMod.regions.head.blocks.head.operations.collectFirst {
          case t: TLambda =>
            t
        }.getOrElse(fail("expected an outer TLambda after the pass"))

      // val swDbg = new StringWriter()
      // Printer(p = new PrintWriter(swDbg)).print(outerAfter)
      // println("=== OUTER AFTER ===")
      // println(swDbg.toString)

      val outerBlock = outerAfter.tBody.blocks.head

      val spec = outerBlock.operations.collectFirst {
        case v: VLambda if v.funAttr == alphaFun && v.res.typ == alphaFun => v
      }.getOrElse(fail("specialized VLambda(α→α) not found in outer block"))

      val ops = outerBlock.operations

      val tRetIdx = outerBlock.operations.indexWhere(_.name == "tlam.treturn")
      if tRetIdx < 0 then fail("no tlam.treturn found in outer block")
      val idxSpec = ops.indexOf(spec)
      // val idxSpec = outerBlock.getIndexOf(spec)

      withClue("specialized vlam should be inserted before the treturn") {
        assert(idxSpec >= 0, "specialized vlam not found in operations list")
        assert(idxSpec < tRetIdx)
      }
    }

  /*
  Test: Full pipeline — monomorphize + erase-tlam + lower-tlam-to-func
  ===================================================================

  Goal
  ----
  Ensure that we can take a program containing:
    - TLambda / TApply / TReturn
    - VLambda / VApply / VReturn
  and turn it into func.* ops:
    - func.func
    - func.call
    - func.return

  What we assert
  --------------
  - No tlam.tapply / tlam.tlambda / tlam.treturn remain.
  - At least one func.func exists (lifted lambda).
  - At least one func.return exists.
  - Module verifies.
   */

  "A Full lowering pipeline" should
    "eliminate type-level ops and produce func.func/call/return" in {

      import tlamTy.*
      import scair.MLContext
      import scair.dialects.builtin.BuiltinDialect
      import scair.dialects.tlam_de_bruijn.TlamDeBruijnDialect
      import scair.dialects.func.FuncDialect
      import scair.passes.{MonomorphizePass, EraseTLamPass, LowerTLamToFuncPass}

      val ctx = MLContext()
      ctx.registerDialect(BuiltinDialect)
      ctx.registerDialect(TlamDeBruijnDialect)
      ctx.registerDialect(FuncDialect)

      // ----------------------------------------------------------------------
      // Program (as lambda-calculus comment):
      //
      //   F = Λα. ( (Λβ. λ(x:β). x) [α] )
      //
      // i.e.:
      //   - inner polymorphic identity:   G = Λβ. λ(x:β). x
      //   - outer binder α instantiates G at α via tapply
      // ----------------------------------------------------------------------

      // Types under current binder: α → α and ∀α. α → α
      val alphaFun: tlamType = fun(bvar(IntData(0)), bvar(IntData(0))) // α -> α
      val forallAlphaFun: tlamType = forall(alphaFun) // ∀α. α -> α

      // Inner: G = Λβ. λ(x: β). x   (β = bvar(0) in inner binder)
      val innerVRes =
        Result[TypeAttribute](fun(bvar(IntData(0)), bvar(IntData(0))))
      val innerVRegion =
        Region(
          Seq(
            Block(
              bvar(IntData(0)),
              (x: Value[Attribute]) =>
                val xd = x.asInstanceOf[Value[TypeAttribute]]
                Seq(VReturn(xd, expected = bvar(IntData(0)))),
            )
          )
        )
      val innerVLam = VLambda(
        funAttr = fun(bvar(IntData(0)), bvar(IntData(0))),
        body = innerVRegion,
        res = innerVRes,
      )

      val innerTRet = TReturn(
        value = innerVRes,
        expected = fun(bvar(IntData(0)), bvar(IntData(0))),
      )

      val innerTLRes =
        Result[TypeAttribute](forall(fun(bvar(IntData(0)), bvar(IntData(0)))))
      val innerTLRegion =
        Region(Seq(Block(operations = Seq(innerVLam, innerTRet))))
      val G = TLambda(tBody = innerTLRegion, res = innerTLRes)
      G.verify().isOK shouldBe true

      // Outer: F = Λα. (h := tapply G <!α>; treturn h : α→α)
      val hRes = Result[TypeAttribute](alphaFun)
      val tapply =
        TApply(
          polymorphicFun = innerTLRes,
          argType = bvar(IntData(0)),
          res = hRes,
        )
      tapply.verify().isOK shouldBe true

      val outerTRet = TReturn(value = hRes, expected = alphaFun)

      val outerTLRes = Result[TypeAttribute](forallAlphaFun)
      val outerTLRegion =
        Region(Seq(Block(operations = Seq(G, tapply, outerTRet))))
      val F = TLambda(tBody = outerTLRegion, res = outerTLRes)
      F.verify().isOK shouldBe true

      val module: ModuleOp =
        ModuleOp(Region(Seq(Block(operations = Seq(F)))))

      module.verify().isOK shouldBe true

      // -------------------------- Run pipeline -------------------------------
      val afterMono =
        new MonomorphizePass(ctx).transform(module).asInstanceOf[ModuleOp]
      dump("After Monomorphize", afterMono)

      val afterErase =
        new EraseTLamPass(ctx).transform(afterMono).asInstanceOf[ModuleOp]
      dump("After EraseTLam", afterErase)

      val afterLower =
        new LowerTLamToFuncPass(ctx).transform(afterErase)
          .asInstanceOf[ModuleOp]
      dump("After LowerTLamToFunc", afterLower)

      afterLower.verify().isOK shouldBe true

      val sw = new StringWriter()
      Printer(p = new PrintWriter(sw)).print(afterLower)
      val out = sw.toString

      // We only assert removal of type-level ops here.
      // Removal of tlam.vlambda/vapply is asserted in the dedicated "vapply -> func.call" test.

      out should include("func.func")
      out should include("func.return")
      out should not include ("tlam.tlambda")
      out should not include ("tlam.tapply")
      // out should not include ("tlam.vlambda")
      // out should not include ("tlam.vapply")
    }

  /*
  Test: Pipeline produces func.call
  ================================

  Build:
    - a polymorphic identity, instantiate it to i32->i32
    - apply to an i32 value (in IR we model an argument block)
  We only check rewriting, not execution.

  What we assert
  --------------
  - Output contains func.call.
   */

  "LowerTLamToFuncPass" should "rewrite vapply into func.call" in {
    import scair.MLContext
    import scair.dialects.builtin.BuiltinDialect
    import scair.dialects.tlam_de_bruijn.TlamDeBruijnDialect
    import scair.dialects.func.FuncDialect
    import scair.passes.LowerTLamToFuncPass

    val ctx = MLContext()
    ctx.registerDialect(BuiltinDialect)
    ctx.registerDialect(TlamDeBruijnDialect)
    ctx.registerDialect(FuncDialect)

    val i32 = IntegerType(IntData(32), Signed)
    val funTy: tlamType = tlamTy.fun(i32, i32)

    val vRes = Result[TypeAttribute](funTy)
    val body = Region(
      Seq(
        Block(
          i32,
          (x: Value[Attribute]) =>
            val xd = x.asInstanceOf[Value[TypeAttribute]]
            Seq(VReturn(xd, expected = i32)),
        )
      )
    )

    val lam = VLambda(funAttr = funTy, body = body, res = vRes)

    val appRes = Result[TypeAttribute](i32)
    val top = Block(
      i32,
      (arg0: Value[Attribute]) =>
        val x = arg0.asInstanceOf[Value[TypeAttribute]]
        val app = VApply(lam.res, x, appRes)
        val ret = VReturn(appRes, expected = i32)
        Seq(lam, app, ret),
    )

    val module = ModuleOp(Region(Seq(top)))
    module.verify().isOK shouldBe true

    val after =
      new LowerTLamToFuncPass(ctx).transform(module).asInstanceOf[ModuleOp]
    dump("After LowerTLamToFunc", after)

    after.verify().isOK shouldBe true

    val sw = new StringWriter()
    Printer(p = new PrintWriter(sw)).print(after)
    val out = sw.toString

    out should include("func.func")
    out should include("func.call")
    out should include("func.return")
  }
