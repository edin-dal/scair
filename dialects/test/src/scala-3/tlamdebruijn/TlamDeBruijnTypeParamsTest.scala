package scair

import scair.ir.*
import scair.utils.*
import scair.dialects.testutils.IRTestKit.*
import scair.dialects.builtin.*
import scair.dialects.tlam_de_bruijn.*
import scair.dialects.tlam_de_bruijn.tlamTy.*
import scair.testutils.tlamdebruijn.TlamTestIR.*
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers.*

final class TlamDeBruijnTypeParamsTest extends AnyFlatSpec:

  // ------------------------------ Tests ------------------------------

  "A polymorphic identity (de Bruijn) IR" should
    "build/verify and print expected shape" in {
      // Surface:  Λα. (λ (x: α). x)
      val bodyTy = alphaToAlphaAt(0) // α -> α
      val forallTy = forall1(bodyTy) // ∀α. α -> α

      val idV = vlam(bodyTy)(b0)(x => Seq(VReturn(x)))
      idV.shouldVerify()

      val idT = tlam(forallTy)(idV, TReturn(idV.res))
      idT.shouldVerify()

      val m = module(idT)
      m.shouldVerify()

      val printed = printIR(m)
      println(printed)
      assertPrinted(
        printed,
        includes = Seq(
          "builtin.module",
          "tlam.tlambda",
          "tlam.vlambda",
          "tlam.vreturn",
          "tlam.treturn",
          "!tlam.forall<!tlam.fun<!tlam.bvar<0>, !tlam.bvar<0>>>",
        ),
      )
    }

  "Nested tlambda + tapply (de Bruijn) IR" should
    "instantiate ∀ with outer bvar(0)" in {
      // Surface (sketch):
      //   Λα. let G = Λβ. λ(x:β). x in
      //       let h = G[α] in
      //       return h
      val bodyTy = alphaToAlphaAt(0) // α -> α (from *current* binder)
      val forallTy = forall1(bodyTy)

      val innerIdV = vlam(bodyTy)(b0)(x => Seq(VReturn(x)))
      innerIdV.shouldVerify()

      val innerG = tlam(forallTy)(innerIdV, TReturn(innerIdV.res))
      innerG.shouldVerify()

      val hRes = Result[tlamType](bodyTy)
      val tapp = TApply(innerG.res, b0, hRes)
      tapp.verify().shouldBeOK("verify failed for tapply")

      val outerF = tlam(forallTy)(innerG, tapp, TReturn(hRes))
      outerF.shouldVerify()

      val m = module(outerF)
      m.shouldVerify()

      val printed = printIR(m)
      println(printed)
      assertPrinted(
        printed,
        includes = Seq(
          "builtin.module",
          "tlam.tlambda",
          "tlam.tapply",
          "tlam.vlambda",
          "tlam.vreturn",
          "!tlam.forall<!tlam.fun<!tlam.bvar<0>, !tlam.bvar<0>>>",
        ),
      )
    }

  "Instantiate to a ground type (i32)" should
    "compute (∀α. α→α)[i32] == i32→i32" in {
      val poly = forall1(alphaToAlphaAt(0))
      val inst = DBI.instantiate(poly, I32)

      inst shouldEqual fun(I32, I32)

      val polyDef = polyIdDef()
      polyDef.shouldVerify()

      val appRes = Result[TypeAttribute](fun(I32, I32))
      TApply(polyDef.res, I32, appRes).verify()
        .shouldBeOK("verify failed for tapply")
    }

  "DBI.shift" should
    "bump only indices >= cutoff, distribute over fun, and respect binders" in {
      import DBI.*

      // basic
      shift(1, 0, b0) shouldEqual b1
      shift(1, 1, b0) shouldEqual b0
      shift(0, 0, b(42)) shouldEqual b(42)

      // structure
      val ty = fun(b0, fun(b1, b0)) // (0 -> (1 -> 0))
      shift(1, 0, ty) shouldEqual fun(b1, fun(b2, b1))

      // binder cutoff increase
      val poly = forall1(fun(b0, b1)) // ∀. (0 -> 1)
      shift(1, 0, poly) shouldEqual forall1(fun(b0, b2))
    }

  "DBI.subst" should
    "be capture-avoiding under forall and decrement indices above the hole" in {
      import DBI.*

      val t = forall1(fun(b2, b1))
      subst(1, b0, t) shouldEqual forall1(fun(b1, b1))

      val t2 = fun(b1, b0)
      subst(0, I32, t2) shouldEqual fun(b0, I32)
    }

  "VLambda.verify" should
    "fail when the block arg type doesn't match the function input" in {
      val funTy = alphaToAlphaAt(0)
      val res = Result[tlamFunType](funTy)

      val wrongRegion =
        Region(
          Seq(
            Block(
              b1, // should be b0
              (x: Value[Attribute]) =>
                Seq(VReturn(x.asInstanceOf[Value[TypeAttribute]])),
            )
          )
        )

      VLambda(wrongRegion, res).verify().isError shouldBe true
    }

  "TApply.verify" should "fail when res.typ != instantiated type" in {
    val polyDef = polyIdDef()
    polyDef.shouldVerify()

    val bad = TApply(polyDef.res, I32, Result[TypeAttribute](fun(I32, I64)))
    bad.verify().isError shouldBe true
  }

  "Monomorphize pass" should
    "replace tapply with a specialized vlam and remove the inner tlambda" in {
      import scair.MLContext
      import scair.dialects.builtin.BuiltinDialect
      import scair.dialects.tlam_de_bruijn.TlamDeBruijnDialect
      import scair.passes.MonomorphizePass

      val alphaFun = alphaToAlphaAt(0) // α -> α
      val forallAlphaFun = forall1(alphaFun) // ∀α. α -> α

      // Inner G = Λβ. (λ(x:β). x)
      val innerBody = alphaToAlphaAt(0) // β -> β inside G
      val innerForall = forall1(innerBody)

      val innerIdV = vlam(innerBody)(b0)(x => Seq(VReturn(x)))
      val G = tlam(innerForall)(innerIdV, TReturn(innerIdV.res))
      G.shouldVerify()

      // Outer F = Λα. (tapply G α)
      val hRes = Result[tlamType](alphaFun)
      val tapp = TApply(G.res, b0, hRes)
      tapp.verify().shouldBeOK("verify failed for tapply")

      val F = tlam(forallAlphaFun)(G, tapp, TReturn(hRes))
      F.shouldVerify()

      val before = module(F)
      before.shouldVerify()

      countOps[TApply](before) shouldBe 1
      countOps[TLambda](before) shouldBe 2

      val ctx = MLContext()
      ctx.registerDialect(BuiltinDialect)
      ctx.registerDialect(TlamDeBruijnDialect)

      val after = new MonomorphizePass(ctx).transform(before)
        .asInstanceOf[ModuleOp]
      after.shouldVerify()

      countOps[TApply](after) shouldBe 0
      countOps[TLambda](after) shouldBe 1

      val out = printIR(after)
      println(out)
      out should include("tlam.vlambda")
      out should not include ("tlam.tapply")
    }

  "A full lowering pipeline" should
    "eliminate type-level ops and produce func.func/call/return" in {
      import scair.MLContext
      import scair.dialects.builtin.BuiltinDialect
      import scair.dialects.tlam_de_bruijn.TlamDeBruijnDialect
      import scair.dialects.func.FuncDialect
      import scair.passes.{MonomorphizePass, EraseTLamPass, LowerTLamToFuncPass}

      val ctx = MLContext()
      ctx.registerDialect(BuiltinDialect)
      ctx.registerDialect(TlamDeBruijnDialect)
      ctx.registerDialect(FuncDialect)

      val alphaFun = alphaToAlphaAt(0)
      val forallAlphaFun = forall1(alphaFun)

      val innerBody = alphaToAlphaAt(0)
      val innerForall = forall1(innerBody)

      val innerIdV = vlam(innerBody)(b0)(x => Seq(VReturn(x)))
      val G = tlam(innerForall)(innerIdV, TReturn(innerIdV.res))

      val hRes = Result[tlamType](alphaFun)
      val tapp = TApply(G.res, b0, hRes)

      val F = tlam(forallAlphaFun)(G, tapp, TReturn(hRes))

      val prog = module(F)
      prog.shouldVerify()

      val afterMono = new MonomorphizePass(ctx).transform(prog)
        .asInstanceOf[ModuleOp]
      val afterErase = new EraseTLamPass(ctx).transform(afterMono)
        .asInstanceOf[ModuleOp]
      val afterLower = new LowerTLamToFuncPass(ctx).transform(afterErase)
        .asInstanceOf[ModuleOp]
      afterLower.shouldVerify()

      val out = printIR(afterLower)
      println(out)
      assertPrinted(
        out,
        includes = Seq("func.func", "func.return"),
        excludes = Seq("tlam.tlambda", "tlam.tapply", "tlam.treturn"),
      )
    }

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

    val i32 = IntegerType(i(32), Signed)
    val funTy = tlamTy.fun(i32, i32)

    val lam = vlam(funTy)(i32)(x => Seq(VReturn(x)))

    val appRes = Result[TypeAttribute](i32)
    val top = Block(
      i32,
      (arg0: Value[Attribute]) =>
        val x = arg0.asInstanceOf[Value[TypeAttribute]]
        val app = VApply(lam.res, x, appRes)
        val ret = VReturn(appRes)
        Seq(lam, app, ret),
    )

    val m = ModuleOp(Region(Seq(top)))
    m.shouldVerify()

    val after = new LowerTLamToFuncPass(ctx).transform(m).asInstanceOf[ModuleOp]
    after.shouldVerify()

    val out = printIR(after)
    println(out)
    assertPrinted(out, includes = Seq("func.func", "func.call", "func.return"))
  }
