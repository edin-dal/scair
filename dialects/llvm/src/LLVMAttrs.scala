package scair.dialects.llvm

import fastparse.*
import scair.clair.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.parse.*
import scair.print.Printer

// MLIR spells these flags as keywords; deriving a Set parameter would not
// produce its required #llvm.fastmath<...> representation.
case class FastmathFlagsAttr(flags: Set[String] = Set.empty)
    extends DataAttribute[Set[String]] derives TransparentData:
  override def name = "llvm.fastmath"
  override def data = flags

  override def printParameters(p: Printer): Unit =
    p.print("<")
    if flags.isEmpty then p.print("none")
    else p.printListF(flags.toSeq.sorted, p.print, sep = ", ")
    p.print(">")

given AttributeCompanion[FastmathFlagsAttr]:
  override def name = "llvm.fastmath"

  override def parse[$: P](using Parser): P[FastmathFlagsAttr] =
    P("<" ~ bareIdP.rep(1, sep = ",") ~ ">").map { flags =>
      FastmathFlagsAttr(flags.flatMap {
        case "none" => Seq.empty
        case "fast" =>
          Seq("nnan", "ninf", "nsz", "arcp", "contract", "afn", "reassoc")
        case flag => Seq(flag)
      }.toSet)
    }

case class DereferenceableAttr(bytes: IntData, mayBeNull: IntData = IntData(0))
    extends DerivedAttribute["llvm.dereferenceable"] derives AttrDefs:

  override def printParameters(p: Printer): Unit =
    p.print("<bytes = ", bytes)
    if mayBeNull.data != 0 then p.print(", mayBeNull = true")
    p.print(">")

given AttributeCustomParser[DereferenceableAttr]:

  override def parse[$: P](using Parser): P[DereferenceableAttr] =
    P(
      "<" ~ "bytes" ~ "=" ~ decimalLiteralP ~
        ("," ~ "mayBeNull" ~ "=" ~ ("true" | "false").!).? ~ ">"
    ).map { (bytes, nullable) =>
      DereferenceableAttr(
        IntData(bytes),
        IntData(if nullable.contains("true") then 1 else 0),
      )
    }

// Spelling and cases from LLVMEnums.td at llvmorg-22.1.1.
enum CConv(spelling: String) extends EnumAttr("llvm.cconv", spelling):
  case C extends CConv("ccc")
  case Fast extends CConv("fastcc")
  case Cold extends CConv("coldcc")
  case GHC extends CConv("cc_10")
  case HiPE extends CConv("cc_11")
  case AnyReg extends CConv("anyregcc")
  case PreserveMost extends CConv("preserve_mostcc")
  case PreserveAll extends CConv("preserve_allcc")
  case Swift extends CConv("swiftcc")
  case CXX_FAST_TLS extends CConv("cxx_fast_tlscc")
  case Tail extends CConv("tailcc")
  case CFGuard_Check extends CConv("cfguard_checkcc")
  case SwiftTail extends CConv("swifttailcc")
  case X86_StdCall extends CConv("x86_stdcallcc")
  case X86_FastCall extends CConv("x86_fastcallcc")
  case ARM_APCS extends CConv("arm_apcscc")
  case ARM_AAPCS extends CConv("arm_aapcscc")
  case ARM_AAPCS_VFP extends CConv("arm_aapcs_vfpcc")
  case MSP430_INTR extends CConv("msp430_intrcc")
  case X86_ThisCall extends CConv("x86_thiscallcc")
  case PTX_Kernel extends CConv("ptx_kernelcc")
  case PTX_Device extends CConv("ptx_devicecc")
  case SPIR_FUNC extends CConv("spir_funccc")
  case SPIR_KERNEL extends CConv("spir_kernelcc")
  case Intel_OCL_BI extends CConv("intel_ocl_bicc")
  case X86_64_SysV extends CConv("x86_64_sysvcc")
  case Win64 extends CConv("win64cc")
  case X86_VectorCall extends CConv("x86_vectorcallcc")
  case DUMMY_HHVM extends CConv("hhvmcc")
  case DUMMY_HHVM_C extends CConv("hhvm_ccc")
  case X86_INTR extends CConv("x86_intrcc")
  case AVR_INTR extends CConv("avr_intrcc")
  case AVR_SIGNAL extends CConv("avr_signalcc")
  case AVR_BUILTIN extends CConv("avr_builtincc")
  case AMDGPU_VS extends CConv("amdgpu_vscc")
  case AMDGPU_GS extends CConv("amdgpu_gscc")
  case AMDGPU_PS extends CConv("amdgpu_pscc")
  case AMDGPU_CS extends CConv("amdgpu_cscc")
  case AMDGPU_KERNEL extends CConv("amdgpu_kernelcc")
  case X86_RegCall extends CConv("x86_regcallcc")
  case AMDGPU_HS extends CConv("amdgpu_hscc")
  case MSP430_BUILTIN extends CConv("msp430_builtincc")
  case AMDGPU_LS extends CConv("amdgpu_lscc")
  case AMDGPU_ES extends CConv("amdgpu_escc")
  case AArch64_VectorCall extends CConv("aarch64_vectorcallcc")
  case AArch64_SVE_VectorCall extends CConv("aarch64_sve_vectorcallcc")
  case WASM_EmscriptenInvoke extends CConv("wasm_emscripten_invokecc")
  case AMDGPU_Gfx extends CConv("amdgpu_gfxcc")
  case M68k_INTR extends CConv("m68k_intrcc")

// Spelling and cases from LLVMEnums.td at llvmorg-22.1.1.
enum Linkage(spelling: String) extends EnumAttr("llvm.linkage", spelling):
  case External extends Linkage("external")
  case AvailableExternally extends Linkage("available_externally")
  case Linkonce extends Linkage("linkonce")
  case LinkonceODR extends Linkage("linkonce_odr")
  case Weak extends Linkage("weak")
  case WeakODR extends Linkage("weak_odr")
  case Appending extends Linkage("appending")
  case Internal extends Linkage("internal")
  case Private extends Linkage("private")
  case ExternWeak extends Linkage("extern_weak")
  case Common extends Linkage("common")

// Spelling and cases from LLVMEnums.td at llvmorg-22.1.1.
enum TailCallKind(spelling: String)
    extends EnumAttr("llvm.tailcallkind", spelling):
  case None extends TailCallKind("none")
  case Tail extends TailCallKind("tail")
  case MustTail extends TailCallKind("musttail")
  case NoTail extends TailCallKind("notail")

enum FramePointerKind(spelling: String)
    extends EnumAttr("llvm.framePointerKind", spelling):
  case None extends FramePointerKind("none")
  case NonLeaf extends FramePointerKind("non-leaf")
  case All extends FramePointerKind("all")
  case Reserved extends FramePointerKind("reserved")
  case NonLeafNoReserve extends FramePointerKind("non-leaf-no-reserve")

  override def printParameters(p: Printer): Unit =
    p.print("<")
    if caseName.contains("-") then p.printStringLiteral(caseName)
    else p.print(caseName)
    p.print(">")

given AttributeCompanion[FramePointerKind]:
  override def name = "llvm.framePointerKind"

  override def parse[$: P](using Parser): P[FramePointerKind] =
    P("<" ~ (stringLiteralP | bareIdP) ~ ">").flatMap { spelling =>
      FramePointerKind.values.find(_.caseName == spelling) match
        case Some(kind) => Pass(kind)
        case None       => Fail("unknown LLVM frame pointer kind")
    }

enum UWTableKind(spelling: String)
    extends EnumAttr("llvm.uwtableKind", spelling):
  case None extends UWTableKind("none")
  case Sync extends UWTableKind("sync")
  case Async extends UWTableKind("async")

// These parsers accept the canonical named-field order emitted by MLIR.
case class VScaleRangeAttr(minRange: IntegerAttr, maxRange: IntegerAttr)
    extends DerivedAttribute["llvm.vscale_range"] derives AttrDefs:

  override def printParameters(p: Printer): Unit =
    p.print("<minRange = ", minRange, ", maxRange = ", maxRange, ">")

given AttributeCustomParser[VScaleRangeAttr]:

  override def parse[$: P](using Parser): P[VScaleRangeAttr] =
    P(
      "<" ~ "minRange" ~ "=" ~ integerAttrP ~ "," ~ "maxRange" ~ "=" ~
        integerAttrP ~ ">"
    ).map((min, max) => VScaleRangeAttr(min, max))

// The derived single-array parameter format is already MLIR-compatible.
case class TargetFeaturesAttr(features: ArrayAttribute[StringData])
    extends DerivedAttribute["llvm.target_features"] derives AttrDefs

case class VecTypeHintAttr(hint: TypeAttribute, is_signed: IntData = IntData(0))
    extends DerivedAttribute["llvm.vec_type_hint"] derives AttrDefs:

  override def printParameters(p: Printer): Unit =
    p.print("<hint = ", hint)
    if is_signed.data != 0 then p.print(", is_signed = true")
    p.print(">")

given AttributeCustomParser[VecTypeHintAttr]:

  override def parse[$: P](using Parser): P[VecTypeHintAttr] =
    P(
      "<" ~ "hint" ~ "=" ~ typeP ~
        ("," ~ "is_signed" ~ "=" ~ ("true" | "false").!).? ~ ">"
    ).map { (hint, signed) =>
      VecTypeHintAttr(
        hint.asInstanceOf[TypeAttribute],
        IntData(if signed.contains("true") then 1 else 0),
      )
    }

case class MemoryEffectsAttr(
    other: StringData,
    argMem: StringData,
    inaccessibleMem: StringData,
    errnoMem: StringData,
    targetMem0: StringData,
    targetMem1: StringData,
) extends DerivedAttribute["llvm.memory_effects"] derives AttrDefs:

  override def printParameters(p: Printer): Unit =
    p.print(
      "<other = ",
      other.data,
      ", argMem = ",
      argMem.data,
      ", inaccessibleMem = ",
      inaccessibleMem.data,
      ", errnoMem = ",
      errnoMem.data,
      ", targetMem0 = ",
      targetMem0.data,
      ", targetMem1 = ",
      targetMem1.data,
      ">",
    )

given AttributeCustomParser[MemoryEffectsAttr]:

  override def parse[$: P](using Parser): P[MemoryEffectsAttr] =
    P(
      "<" ~ "other" ~ "=" ~ bareIdP ~ "," ~ "argMem" ~ "=" ~ bareIdP ~ "," ~
        "inaccessibleMem" ~ "=" ~ bareIdP ~ "," ~ "errnoMem" ~ "=" ~ bareIdP ~
        "," ~ "targetMem0" ~ "=" ~ bareIdP ~ "," ~ "targetMem1" ~ "=" ~
        bareIdP ~ ">"
    ).map { (other, argMem, inaccessibleMem, errnoMem, targetMem0, targetMem1) =>
      MemoryEffectsAttr(
        StringData(other),
        StringData(argMem),
        StringData(inaccessibleMem),
        StringData(errnoMem),
        StringData(targetMem0),
        StringData(targetMem1),
      )
    }
