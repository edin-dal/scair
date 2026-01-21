package scair.testutils.tlamdebruijn

import scair.ir.*
import scair.dialects.tlam_de_bruijn.*
import scair.dialects.tlam_de_bruijn.tlamTy.*
import scair.dialects.builtin.*

object TlamTestIR:

  // --- tiny constructors / aliases ---
  inline def i(n: Int): IntData = IntData(n)
  inline def b(n: Int): tlamBVarType = bvar(i(n))
  inline def b0: tlamBVarType = b(0)
  inline def b1: tlamBVarType = b(1)
  inline def b2: tlamBVarType = b(2)

  inline def alphaToAlphaAt(idx: Int): tlamFunType = fun(b(idx), b(idx))
  inline def forall1(body: TypeAttribute): tlamForAllType = forall(body)

  // --- common IR building patterns ---
  def module(ops: Operation*): ModuleOp =
    ModuleOp(Region(Seq(Block(operations = ops.toSeq))))

  def vlam(funTy: tlamFunType)(argTy: TypeAttribute)(
      bodyOps: Value[TypeAttribute] => Seq[Operation]
  ): VLambda =
    val res = Result[tlamFunType](funTy)
    val region =
      Region(
        Seq(
          Block(
            argTy,
            (x: Value[Attribute]) =>
              bodyOps(x.asInstanceOf[Value[TypeAttribute]]),
          )
        )
      )
    VLambda(body = region, res = res)

  def tlam(resTy: tlamForAllType)(ops: Operation*): TLambda =
    val res = Result[tlamForAllType](resTy)
    val region = Region(Seq(Block(operations = ops.toSeq)))
    TLambda(body = region, res = res)
