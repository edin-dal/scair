package scair.dialects.llvm

import fastparse.*
import scair.clair.*
import scair.dialects.builtin.*
import scair.enums.*
import scair.ir.*
import scair.parse.*
import scair.print.Printer
import scair.utils.*

case class Ptr(addressSpace: IntData = IntData(0))
    extends DerivedAttribute["llvm.ptr"]
    with TypeAttribute derives AttrDefs:

  override def printParameters(p: Printer): Unit =
    if addressSpace.data != 0 then p.print("<", addressSpace, ">")

given AttributeCustomParser[Ptr]:

  override def parse[$: P](using Parser): P[Ptr] =
    P(("<" ~ decimalLiteralP ~ ">").?)
      .map(space => Ptr(IntData(space.getOrElse(BigInt(0)))))

enum ICmpPredicate(name: String) extends I64Enum(name):
  case eq extends ICmpPredicate("eq")
  case ne extends ICmpPredicate("ne")
  case slt extends ICmpPredicate("slt")
  case sle extends ICmpPredicate("sle")
  case sgt extends ICmpPredicate("sgt")
  case sge extends ICmpPredicate("sge")
  case ult extends ICmpPredicate("ult")
  case ule extends ICmpPredicate("ule")
  case ugt extends ICmpPredicate("ugt")
  case uge extends ICmpPredicate("uge")

enum FCmpPredicate(name: String) extends I64Enum(name):
  case AlwaysFalse extends FCmpPredicate("false")
  case OEQ extends FCmpPredicate("oeq")
  case OGT extends FCmpPredicate("ogt")
  case OGE extends FCmpPredicate("oge")
  case OLT extends FCmpPredicate("olt")
  case OLE extends FCmpPredicate("ole")
  case ONE extends FCmpPredicate("one")
  case ORD extends FCmpPredicate("ord")
  case UEQ extends FCmpPredicate("ueq")
  case UGT extends FCmpPredicate("ugt")
  case UGE extends FCmpPredicate("uge")
  case ULT extends FCmpPredicate("ult")
  case ULE extends FCmpPredicate("ule")
  case UNE extends FCmpPredicate("une")
  case UNO extends FCmpPredicate("uno")
  case AlwaysTrue extends FCmpPredicate("true")

final case class StructType(
    elems: ArrayAttribute[TypeAttribute]
) extends DerivedAttribute["llvm.struct"]
    with TypeAttribute derives AttrDefs:

  override def printParameters(p: Printer): Unit =
    given indentLevel: Int = 0
    p.print("<(")
    p.printListF(elems, p.print, sep = ", ")
    p.print(")>")

given AttributeCustomParser[StructType]:

  override def parse[$: P](using Parser): P[StructType] =
    P("<" ~ "(" ~ llvmTypeP.rep(sep = ",") ~ ")" ~ ">")
      .map(elems => StructType(elems.map(_.asInstanceOf[TypeAttribute])))

final case class ArrayType(
    size: IntData,
    elem: TypeAttribute,
) extends DerivedAttribute["llvm.array"]
    with TypeAttribute derives AttrDefs:

  override def printParameters(p: Printer): Unit =
    given indentLevel: Int = 0
    p.print("<", size, " x ", elem, ">")

given AttributeCustomParser[ArrayType]:

  override def parse[$: P](using Parser): P[ArrayType] =
    P("<" ~ decimalLiteralP ~ "x" ~ llvmTypeP ~ ">").map((size, elem) =>
      ArrayType(IntData(size), elem.asInstanceOf[TypeAttribute])
    )

case class Constant(
    value: Attribute,
    res: Result[Attribute],
) extends DerivedOperation["llvm.mlir.constant"]
    with Pure derives OpDefs

case class Zero(
    res: Result[Attribute]
) extends DerivedOperation["llvm.mlir.zero"]
    with Pure derives OpDefs

case class Poison(
    res: Result[Attribute]
) extends DerivedOperation["llvm.mlir.poison"]
    with AssemblyFormat["attr-dict `:` type($res)"]
    with Pure derives OpDefs

case class Add(
    lhs: Operand[IntegerType | VectorType],
    rhs: Operand[IntegerType | VectorType],
    res: Result[IntegerType | VectorType],
    overflowFlags: IntegerAttr = IntegerAttr(IntData(0), I32),
) extends DerivedOperation["llvm.add"]
    with Commutative
    with Pure derives OpDefs

case class Mul(
    lhs: Operand[IntegerType | VectorType],
    rhs: Operand[IntegerType | VectorType],
    res: Result[IntegerType | VectorType],
    overflowFlags: IntegerAttr = IntegerAttr(IntData(0), I32),
) extends DerivedOperation["llvm.mul"]
    with Commutative
    with Pure derives OpDefs

case class FAdd(
    lhs: Operand[FloatType | VectorType],
    rhs: Operand[FloatType | VectorType],
    res: Result[FloatType | VectorType],
    fastmathFlags: FastmathFlagsAttr = FastmathFlagsAttr(),
) extends DerivedOperation["llvm.fadd"]
    with Pure derives OpDefs

case class FMul(
    lhs: Operand[FloatType | VectorType],
    rhs: Operand[FloatType | VectorType],
    res: Result[FloatType | VectorType],
    fastmathFlags: FastmathFlagsAttr = FastmathFlagsAttr(),
) extends DerivedOperation["llvm.fmul"]
    with Pure derives OpDefs

case class ICmp(
    lhs: Operand[IntegerType | Ptr | VectorType],
    rhs: Operand[IntegerType | Ptr | VectorType],
    res: Result[IntegerType | VectorType],
    predicate: ICmpPredicate,
) extends DerivedOperation["llvm.icmp"]
    with Pure derives OpDefs

case class Load(
    addr: Operand[Ptr],
    res: Result[Attribute],
    volatile_ : Option[UnitAttr] = None,
    nontemporal: Option[UnitAttr] = None,
    invariant: Option[UnitAttr] = None,
    invariantGroup: Option[UnitAttr] = None,
    alignment: Option[IntegerAttr] = None,
    ordering: IntegerAttr = IntegerAttr(IntData(0), I64),
    syncscope: Option[StringData] = None,
    dereferenceable: Option[DereferenceableAttr] = None,
    access_groups: Option[ArrayAttribute[Attribute]] = None,
    alias_scopes: Option[ArrayAttribute[Attribute]] = None,
    noalias_scopes: Option[ArrayAttribute[Attribute]] = None,
    tbaa: Option[ArrayAttribute[Attribute]] = None,
) extends DerivedOperation["llvm.load"] derives OpDefs

case class Store(
    value: Operand[Attribute],
    addr: Operand[Ptr],
    volatile_ : Option[UnitAttr] = None,
    nontemporal: Option[UnitAttr] = None,
    invariantGroup: Option[UnitAttr] = None,
    alignment: Option[IntegerAttr] = None,
    ordering: IntegerAttr = IntegerAttr(IntData(0), I64),
    syncscope: Option[StringData] = None,
    access_groups: Option[ArrayAttribute[Attribute]] = None,
    alias_scopes: Option[ArrayAttribute[Attribute]] = None,
    noalias_scopes: Option[ArrayAttribute[Attribute]] = None,
    tbaa: Option[ArrayAttribute[Attribute]] = None,
) extends DerivedOperation["llvm.store"] derives OpDefs

// Mirrors MLIR's LLVM GEPOp encoding. MLIR stores one entry per GEP index in
// rawConstantIndices; constant indices are stored directly, while dynamic/SSA
// indices are marked with LLVM::GEPOp::kDynamicIndex (the minimum int32 value)
// and their actual operands live in dynamicIndices. The verifier below follows
// MLIR by checking that the number of sentinels matches dynamicIndices.size.
private val gepDynamicIndexSentinel: BigInt = BigInt(Int.MinValue)

private def isDynamicGEPIndex(attr: IntegerAttr): Boolean =
  attr.value.value == gepDynamicIndexSentinel

case class GetElementPtr(
    base: Operand[Ptr | VectorType],
    dynamicIndices: Seq[Operand[IntegerType | VectorType]],
    res: Result[Ptr | VectorType],
    rawConstantIndices: DenseArrayAttr,
    elem_type: TypeAttribute,
    noWrapFlags: IntegerAttr = IntegerAttr(IntData(0), I32),
) extends DerivedOperation["llvm.getelementptr"]
    with Pure derives OpDefs:

  override def customVerify(): OK[Operation] =
    val rawIndices = rawConstantIndices.data.data.collect {
      case i: IntegerAttr =>
        i
    }
    val numDynamicMarkers = rawIndices.count(isDynamicGEPIndex)
    if numDynamicMarkers != dynamicIndices.size then
      Err(
        s"llvm.getelementptr: rawConstantIndices contain $numDynamicMarkers dynamic markers but op has ${dynamicIndices
            .size} dynamic indices"
      )
    else OK(this)

case class ExtractValue(
    container: Operand[StructType | ArrayType],
    position: DenseArrayAttr,
    res: Result[Attribute],
) extends DerivedOperation["llvm.extractvalue"]
    with Pure derives OpDefs

case class InsertValue(
    container: Operand[StructType | ArrayType],
    value: Operand[Attribute],
    position: DenseArrayAttr,
    res: Result[StructType | ArrayType],
) extends DerivedOperation["llvm.insertvalue"]
    with Pure derives OpDefs

case class PtrToInt(
    in: Operand[Ptr | VectorType],
    out: Result[IntegerType | VectorType],
) extends DerivedOperation["llvm.ptrtoint"]
    with Pure derives OpDefs

case class IntToPtr(
    in: Operand[IntegerType | VectorType],
    out: Result[Ptr | VectorType],
    dereferenceable: Option[DereferenceableAttr] = None,
) extends DerivedOperation["llvm.inttoptr"]
    with Pure derives OpDefs

case class Call(
    callee: Option[SymbolRefAttr],
    callee_operands: Seq[Operand[Attribute]],
    result: Seq[Result[Attribute]],
    convergent: Option[UnitAttr] = None,
    no_unwind: Option[UnitAttr] = None,
    will_return: Option[UnitAttr] = None,
    no_inline: Option[UnitAttr] = None,
    always_inline: Option[UnitAttr] = None,
    inline_hint: Option[UnitAttr] = None,
    var_callee_type: Option[LLVMFunctionType] = None,
    fastmathFlags: FastmathFlagsAttr = FastmathFlagsAttr(),
    CConv: CConv = scair.dialects.llvm.CConv.C,
    TailCallKind: TailCallKind = scair.dialects.llvm.TailCallKind.None,
    memory_effects: Option[MemoryEffectsAttr] = None,
    op_bundle_operands: Seq[Operand[Attribute]] = Seq.empty,
    op_bundle_sizes: DenseArrayAttr =
      DenseArrayAttr(I32, Seq.empty[IntegerAttr]),
    op_bundle_tags: Option[ArrayAttribute[StringData]] = None,
    arg_attrs: Option[ArrayAttribute[DictionaryAttr]] = None,
    res_attrs: Option[ArrayAttribute[DictionaryAttr]] = None,
    access_groups: Option[ArrayAttribute[Attribute]] = None,
    alias_scopes: Option[ArrayAttribute[Attribute]] = None,
    noalias_scopes: Option[ArrayAttribute[Attribute]] = None,
    tbaa: Option[ArrayAttribute[Attribute]] = None,
) extends DerivedOperation["llvm.call"] derives OpDefs

case class Br(
    args: Seq[Operand[Attribute]],
    dest: Block,
    loop_annotation: Option[Attribute] = None,
) extends DerivedOperation["llvm.br"]
    with Pure
    with IsTerminator derives OpDefs

case class CondBr(
    condition: Operand[IntegerType],
    trueArgs: Seq[Operand[Attribute]],
    falseArgs: Seq[Operand[Attribute]],
    trueDest: Block,
    falseDest: Block,
    branch_weights: Option[DenseArrayAttr] = None,
    loop_annotation: Option[Attribute] = None,
) extends DerivedOperation["llvm.cond_br"]
    with Pure
    with IsTerminator derives OpDefs

case class Return(
    args: Seq[Operand[Attribute]]
) extends DerivedOperation["llvm.return"]
    with Pure
    with IsTerminator derives OpDefs

case class Func(
    sym_name: StringData,
    function_type: LLVMFunctionType,
    sym_visibility: Option[StringData],
    body: Region,
    dso_local: Option[UnitAttr] = None,
    convergent: Option[UnitAttr] = None,
    no_inline: Option[UnitAttr] = None,
    always_inline: Option[UnitAttr] = None,
    inline_hint: Option[UnitAttr] = None,
    no_unwind: Option[UnitAttr] = None,
    will_return: Option[UnitAttr] = None,
    optimize_none: Option[UnitAttr] = None,
    arm_streaming: Option[UnitAttr] = None,
    arm_locally_streaming: Option[UnitAttr] = None,
    arm_streaming_compatible: Option[UnitAttr] = None,
    arm_new_za: Option[UnitAttr] = None,
    arm_in_za: Option[UnitAttr] = None,
    arm_out_za: Option[UnitAttr] = None,
    arm_inout_za: Option[UnitAttr] = None,
    arm_preserves_za: Option[UnitAttr] = None,
    linkage: Linkage = Linkage.External,
    CConv: CConv = scair.dialects.llvm.CConv.C,
    comdat: Option[SymbolRefAttr] = None,
    personality: Option[SymbolRefAttr] = None,
    garbageCollector: Option[StringData] = None,
    passthrough: Option[ArrayAttribute[Attribute]] = None,
    arg_attrs: Option[ArrayAttribute[DictionaryAttr]] = None,
    res_attrs: Option[ArrayAttribute[DictionaryAttr]] = None,
    function_entry_count: Option[IntegerAttr] = None,
    memory_effects: Option[MemoryEffectsAttr] = None,
    visibility_ : IntegerAttr = IntegerAttr(IntData(0), I64),
    unnamed_addr: Option[IntegerAttr] = None,
    section: Option[StringData] = None,
    alignment: Option[IntegerAttr] = None,
    vscale_range: Option[VScaleRangeAttr] = None,
    frame_pointer: Option[FramePointerKind] = None,
    target_cpu: Option[StringData] = None,
    tune_cpu: Option[StringData] = None,
    reciprocal_estimates: Option[StringData] = None,
    prefer_vector_width: Option[StringData] = None,
    target_features: Option[TargetFeaturesAttr] = None,
    no_infs_fp_math: Option[IntegerAttr] = None,
    no_nans_fp_math: Option[IntegerAttr] = None,
    no_signed_zeros_fp_math: Option[IntegerAttr] = None,
    denormal_fp_math: Option[StringData] = None,
    denormal_fp_math_f32: Option[StringData] = None,
    fp_contract: Option[StringData] = None,
    instrument_function_entry: Option[StringData] = None,
    instrument_function_exit: Option[StringData] = None,
    vec_type_hint: Option[VecTypeHintAttr] = None,
    work_group_size_hint: Option[DenseArrayAttr] = None,
    reqd_work_group_size: Option[DenseArrayAttr] = None,
    intel_reqd_sub_group_size: Option[IntegerAttr] = None,
    uwtable_kind: Option[UWTableKind] = None,
) extends DerivedOperation["llvm.func"]
    with IsolatedFromAbove
    with Symbol derives OpDefs

// Definitions and traits follow LLVMOps.td at llvmorg-22.1.1.
case class Sub(
    lhs: Operand[IntegerType | VectorType],
    rhs: Operand[IntegerType | VectorType],
    res: Result[IntegerType | VectorType],
    overflowFlags: IntegerAttr = IntegerAttr(IntData(0), I32),
) extends DerivedOperation["llvm.sub"]
    with Pure derives OpDefs

case class SDiv(
    lhs: Operand[IntegerType | VectorType],
    rhs: Operand[IntegerType | VectorType],
    res: Result[IntegerType | VectorType],
    isExact: Option[UnitAttr] = None,
) extends DerivedOperation["llvm.sdiv"]
    with NoMemoryEffect derives OpDefs

case class SRem(
    lhs: Operand[IntegerType | VectorType],
    rhs: Operand[IntegerType | VectorType],
    res: Result[IntegerType | VectorType],
) extends DerivedOperation["llvm.srem"]
    with Pure derives OpDefs

case class FSub(
    lhs: Operand[FloatType | VectorType],
    rhs: Operand[FloatType | VectorType],
    res: Result[FloatType | VectorType],
    fastmathFlags: FastmathFlagsAttr = FastmathFlagsAttr(),
) extends DerivedOperation["llvm.fsub"]
    with Pure derives OpDefs

case class FDiv(
    lhs: Operand[FloatType | VectorType],
    rhs: Operand[FloatType | VectorType],
    res: Result[FloatType | VectorType],
    fastmathFlags: FastmathFlagsAttr = FastmathFlagsAttr(),
) extends DerivedOperation["llvm.fdiv"]
    with Pure derives OpDefs

case class FRem(
    lhs: Operand[FloatType | VectorType],
    rhs: Operand[FloatType | VectorType],
    res: Result[FloatType | VectorType],
    fastmathFlags: FastmathFlagsAttr = FastmathFlagsAttr(),
) extends DerivedOperation["llvm.frem"]
    with Pure derives OpDefs

case class FNeg(
    operand: Operand[FloatType | VectorType],
    res: Result[FloatType | VectorType],
    fastmathFlags: FastmathFlagsAttr = FastmathFlagsAttr(),
) extends DerivedOperation["llvm.fneg"]
    with Pure derives OpDefs

case class FCmp(
    lhs: Operand[FloatType | VectorType],
    rhs: Operand[FloatType | VectorType],
    res: Result[IntegerType | VectorType],
    predicate: FCmpPredicate,
    fastmathFlags: FastmathFlagsAttr = FastmathFlagsAttr(),
) extends DerivedOperation["llvm.fcmp"]
    with Pure derives OpDefs

case class And(
    lhs: Operand[IntegerType | VectorType],
    rhs: Operand[IntegerType | VectorType],
    res: Result[IntegerType | VectorType],
) extends DerivedOperation["llvm.and"]
    with Pure derives OpDefs

case class Or(
    lhs: Operand[IntegerType | VectorType],
    rhs: Operand[IntegerType | VectorType],
    res: Result[IntegerType | VectorType],
    isDisjoint: Option[UnitAttr] = None,
) extends DerivedOperation["llvm.or"]
    with Pure derives OpDefs

case class XOr(
    lhs: Operand[IntegerType | VectorType],
    rhs: Operand[IntegerType | VectorType],
    res: Result[IntegerType | VectorType],
) extends DerivedOperation["llvm.xor"]
    with Pure derives OpDefs

case class Unreachable()
    extends DerivedOperation["llvm.unreachable"]
    with IsTerminator derives OpDefs

case class Alloca(
    arraySize: Operand[IntegerType],
    res: Result[Ptr],
    elem_type: TypeAttribute,
    alignment: Option[IntegerAttr] = None,
    inalloca: Option[UnitAttr] = None,
) extends DerivedOperation["llvm.alloca"] derives OpDefs

case class Trunc(
    in: Operand[IntegerType | VectorType],
    out: Result[IntegerType | VectorType],
    overflowFlags: IntegerAttr = IntegerAttr(IntData(0), I32),
) extends DerivedOperation["llvm.trunc"]
    with Pure derives OpDefs

case class ZExt(
    in: Operand[IntegerType | VectorType],
    out: Result[IntegerType | VectorType],
    nonNeg: Option[UnitAttr] = None,
) extends DerivedOperation["llvm.zext"]
    with Pure derives OpDefs

case class SExt(
    in: Operand[IntegerType | VectorType],
    out: Result[IntegerType | VectorType],
) extends DerivedOperation["llvm.sext"]
    with Pure derives OpDefs

case class SIToFP(
    in: Operand[IntegerType | VectorType],
    out: Result[FloatType | VectorType],
) extends DerivedOperation["llvm.sitofp"]
    with Pure derives OpDefs

case class FPToSI(
    in: Operand[FloatType | VectorType],
    out: Result[IntegerType | VectorType],
) extends DerivedOperation["llvm.fptosi"]
    with Pure derives OpDefs

case class FPTrunc(
    in: Operand[FloatType | VectorType],
    out: Result[FloatType | VectorType],
) extends DerivedOperation["llvm.fptrunc"]
    with Pure derives OpDefs

case class FPExt(
    in: Operand[FloatType | VectorType],
    out: Result[FloatType | VectorType],
) extends DerivedOperation["llvm.fpext"]
    with Pure derives OpDefs

case class Shl(
    lhs: Operand[IntegerType | VectorType],
    rhs: Operand[IntegerType | VectorType],
    res: Result[IntegerType | VectorType],
    overflowFlags: IntegerAttr = IntegerAttr(IntData(0), I32),
) extends DerivedOperation["llvm.shl"]
    with Pure derives OpDefs

case class LShr(
    lhs: Operand[IntegerType | VectorType],
    rhs: Operand[IntegerType | VectorType],
    res: Result[IntegerType | VectorType],
    isExact: Option[UnitAttr] = None,
) extends DerivedOperation["llvm.lshr"]
    with Pure derives OpDefs

case class AShr(
    lhs: Operand[IntegerType | VectorType],
    rhs: Operand[IntegerType | VectorType],
    res: Result[IntegerType | VectorType],
    isExact: Option[UnitAttr] = None,
) extends DerivedOperation["llvm.ashr"]
    with Pure derives OpDefs

case class Select(
    condition: Operand[IntegerType | VectorType],
    trueValue: Operand[Attribute],
    falseValue: Operand[Attribute],
    res: Result[Attribute],
    fastmathFlags: FastmathFlagsAttr = FastmathFlagsAttr(),
) extends DerivedOperation["llvm.select"]
    with Pure derives OpDefs

val LLVMDialect = summonDialect[
  (
      Ptr,
      StructType,
      ArrayType,
      VoidType,
      LLVMFunctionType,
      FastmathFlagsAttr,
      DereferenceableAttr,
      CConv,
      Linkage,
      TailCallKind,
      FramePointerKind,
      UWTableKind,
      VScaleRangeAttr,
      TargetFeaturesAttr,
      VecTypeHintAttr,
      MemoryEffectsAttr,
  ),
  (
      Func,
      Constant,
      Zero,
      Poison,
      Add,
      Sub,
      Mul,
      SDiv,
      SRem,
      FAdd,
      FSub,
      FMul,
      FDiv,
      FRem,
      FNeg,
      ICmp,
      FCmp,
      And,
      Or,
      XOr,
      Shl,
      LShr,
      AShr,
      Load,
      Store,
      GetElementPtr,
      ExtractValue,
      InsertValue,
      Trunc,
      ZExt,
      SExt,
      SIToFP,
      FPToSI,
      FPTrunc,
      FPExt,
      Select,
      PtrToInt,
      IntToPtr,
      Call,
      Alloca,
      Br,
      CondBr,
      Return,
      Unreachable,
  ),
]
