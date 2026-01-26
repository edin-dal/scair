package scair

import scair.MLContext
import scair.ir.*
import scair.dialects.testutils.IRTestKit.*
import scair.dialects.builtin.*
import scair.dialects.builtin.BuiltinDialect
import scair.dialects.tlam_de_bruijn.*
import scair.dialects.tlam_de_bruijn.TlamDeBruijnDialect
import scair.dialects.tlam_de_bruijn.tlamTy.*
import scair.testutils.tlamdebruijn.TlamTestIR.*
import scair.verify.Verifier
import scair.utils.Err

import org.scalatest.flatspec.AnyFlatSpec

final class DeBruijnIndicesCheckTest extends AnyFlatSpec:

  // ------------------------------ shared runner ------------------------------

  private lazy val ctx: MLContext =
    val c = MLContext()
    c.registerDialect(BuiltinDialect)
    c.registerDialect(TlamDeBruijnDialect)
    c

  private def runDeBruijnVerify(m: ModuleOp): Unit =
    m.shouldVerify()
    Verifier.verify(m, ctx) match
      case e: Err => throw new Exception(e.msg)
      case _      => ()

  // ------------------------------ Tests ------------------------------

  // Lambda notation (de Bruijn):
  //   Λ.  λ (x : #1). x
  // Inside the TLambda body we have depth = 1, so only #0 is valid.
  "DeBruijn verifier" should "reject bvar<1> under a single TLambda binder" in {
    val badFunTy = fun(b1, b1) // invalid at depth=1

    val vLam = vlam(badFunTy)(b1)(x => Seq(VReturn(x))) // also invalid
    vLam.shouldVerify()

    val tl = tlam(forall1(badFunTy))(vLam, TReturn(vLam.res))
    tl.shouldVerify()

    val m = module(tl)
    intercept[Exception](runDeBruijnVerify(m))
  }

  // Lambda notation (de Bruijn):
  //   Λ. Λ. λ (x : #1). x
  // Inside the inner TLambda body we have depth = 2:
  //   #0 = inner binder, #1 = outer binder.
  "DeBruijn verifier" should
    "accept bvar<1> when two TLambda binders are in scope" in {
      val funTy =
        fun(b1, b1)

      val vLam = vlam(funTy)(b1)(x => Seq(VReturn(x)))
      vLam.shouldVerify()

      val innerTL = tlam(forall1(funTy))(vLam, TReturn(vLam.res))
      innerTL.shouldVerify()

      val outerTL =
        tlam(forall1(innerTL.res.typ))(innerTL, TReturn(innerTL.res))
      outerTL.shouldVerify()

      val m = module(outerTL)
      runDeBruijnVerify(m)
    }

  // Type/lambda notation (de Bruijn):
  //   (top-level contains)  (∀. (#0 -> #0)) [ ∀. (#1 -> #0) ]
  // At top-level there is no outer binder. Inside the tyArg's forall body the
  // depth is 1 => only #0 is valid, so #1 is illegal.
  "DeBruijn verifier" should
    "reject bvar<1> inside a forall body at top-level" in {
      // bad forall: inside its body depth=1 => only b0 allowed; b1 invalid
      val badPoly: tlamForAllType = forall1(fun(b1, b0))

      // a well-formed polymorphic function value: ∀. (#0 -> #0)
      // val poly: tlamForAllType = forall1(fun(b0, b0))
      // val polyVal = Result[tlamForAllType](poly)
      /*
      val polyFunTy = forall1(fun(b0, b0))
      val polyProducer =
        tlam(polyFunTy)( /* body that returns a value of type fun(b0,b0) */ )
      val polyVal = polyProducer.res.asInstanceOf[Value[tlamForAllType]]
      
       */
      val polyDef = polyIdDef()
      // Instantiate: (∀. #0->#0)[badPoly] == (badPoly -> badPoly)
      // (Structurally well-formed; rejected by the de Bruijn scoping pass.)

      val tapp = TApply(
        fun = polyDef.res,
        tyArg = badPoly,
        res = Result[TypeAttribute](fun(badPoly, badPoly)),
      )
      tapp.verify().shouldBeOK("verify failed for tapply")

      val m = module(tapp)
      intercept[Exception](runDeBruijnVerify(m))
    }

  // Lambda notation (de Bruijn):
  //   Λ.  (∀. (#0 -> #0)) [ ∀. (#1 -> #0) ]
  // This TApply happens under an outer TLambda (depth=1).
  // The tyArg is ∀. (#1 -> #0); inside that forall body depth becomes 2 => #1 is valid.
  "DeBruijn verifier" should
    "accept bvar<1> inside forall when checked under an outer TLambda binder" in {

      val goodPoly: tlamForAllType = forall1(fun(b1, b0))
      val polyDef = polyIdDef()

      val tapp = TApply(
        fun = polyDef.res,
        tyArg = goodPoly,
        res = Result[TypeAttribute](fun(goodPoly, goodPoly)),
      )
      tapp.verify().shouldBeOK("verify failed for tapply")

      // Ensure polyDef dominates tapp by placing it before in the same block.
      val tl = tlam(forall1(tapp.res.typ))(
        polyDef,
        tapp,
        TReturn(tapp.res),
      )
      tl.shouldVerify()

      val m = module(tl)
      runDeBruijnVerify(m)
    }

  // Lambda notation (de Bruijn):
  //   (∀. (#0 -> #0)) [ #0 ]
  // The application is at module top-level where depth = 0 (no binders in scope).
  // Any bvar<#k> is out of scope at depth 0, so #0 is illegal.
  "DeBruijn verifier" should
    "reject TApply tyArg bvar<0> at top-level (depth=0)" in {

      val polyDef = polyIdDef()

      val bad = TApply(
        fun = polyDef.res,
        tyArg = b0,
        res = Result[TypeAttribute](fun(b0, b0)),
      )
      bad.verify().shouldBeOK("verify failed for tapply")

      val m = module(polyDef, bad)
      intercept[Exception](runDeBruijnVerify(m))
    }
