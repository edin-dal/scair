package scair.dialects.llvm

import fastparse.*
import scair.clair.*
import scair.dialects.builtin.*
import scair.enums.*
import scair.ir.*
import scair.parse.*
import scair.parse.given
import scair.print.Printer
import scair.utils.*

case class Ptr() extends DerivedAttribute["llvm.ptr"] with TypeAttribute
    derives AttrDefs

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

object ICmpPredicate:

  def fromString(value: String): Option[ICmpPredicate] =
    values.find(_.name == value)

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

object FCmpPredicate:

  def fromString(value: String): Option[FCmpPredicate] =
    values.find(_.name == value)

final case class StructType(
    elems: ArrayAttribute[TypeAttribute]
) extends ParametrizedAttribute
    with TypeAttribute:
  override def name: String = "llvm.struct"
  override def parameters: Seq[Attribute] = Seq(elems)

  override def printParameters(p: Printer): Unit =
    given indentLevel: Int = 0
    p.print("<(")
    p.printListF(elems, p.print, sep = ", ")
    p.print(")>")

given AttributeCompanion[StructType]:
  override def name: String = "llvm.struct"

  override def parse[$: P](using Parser): P[StructType] =
    P("<" ~ "(" ~ typeP.rep(sep = ",") ~ ")" ~ ">")
      .map(elems => StructType(elems.map(_.asInstanceOf[TypeAttribute])))

final case class ArrayType(
    size: IntData,
    elem: TypeAttribute,
) extends ParametrizedAttribute
    with TypeAttribute:
  override def name: String = "llvm.array"
  override def parameters: Seq[Attribute] = Seq(size, elem)

  override def printParameters(p: Printer): Unit =
    given indentLevel: Int = 0
    p.print("<", size, " x ", elem, ">")

given AttributeCompanion[ArrayType]:
  override def name: String = "llvm.array"

  override def parse[$: P](using Parser): P[ArrayType] =
    P("<" ~ decimalLiteralP ~ "x" ~ typeP ~ ">").map((size, elem) =>
      ArrayType(IntData(size), elem.asInstanceOf[TypeAttribute])
    )

case class Constant(
    value: Attribute,
    res: Result[Attribute],
) extends DerivedOperation["llvm.mlir.constant"]
    with NoMemoryEffect derives OpDefs

case class Zero(
    res: Result[Attribute]
) extends DerivedOperation["llvm.mlir.zero"]
    with NoMemoryEffect derives OpDefs

case class Poison(
    res: Result[Attribute]
) extends DerivedOperation["llvm.mlir.poison"]
    with AssemblyFormat["attr-dict `:` type($res)"]
    with NoMemoryEffect derives OpDefs

case class Add(
    lhs: Operand[IntegerType | IndexType],
    rhs: Operand[IntegerType | IndexType],
    res: Result[IntegerType | IndexType],
    overflowFlags: Option[ArrayAttribute[StringData]] = None,
) extends DerivedOperation["llvm.add"] derives OpDefs

case class Mul(
    lhs: Operand[IntegerType | IndexType],
    rhs: Operand[IntegerType | IndexType],
    res: Result[IntegerType | IndexType],
    overflowFlags: Option[ArrayAttribute[StringData]] = None,
) extends DerivedOperation["llvm.mul"] derives OpDefs

case class FAdd(
    lhs: Operand[FloatType],
    rhs: Operand[FloatType],
    res: Result[FloatType],
) extends DerivedOperation["llvm.fadd"] derives OpDefs

case class FMul(
    lhs: Operand[FloatType],
    rhs: Operand[FloatType],
    res: Result[FloatType],
) extends DerivedOperation["llvm.fmul"] derives OpDefs

case class ICmp(
    lhs: Operand[IntegerType | IndexType],
    rhs: Operand[IntegerType | IndexType],
    res: Result[IntegerType],
    predicate: ICmpPredicate,
) extends DerivedOperation["llvm.icmp"] derives OpDefs:

  override def customPrint(printer: Printer): Unit =
    printer.print(name, " ")
    printer.print("\"", predicate.name, "\" ")
    printer.print(lhs, ", ", rhs, " : ", lhs.typ)

given OperationCustomParser[ICmp]:

  def parse[$: P](resNames: Seq[String])(using Parser): P[ICmp] =
    P(
      stringLiteralP ~ operandNameP ~ "," ~ operandNameP ~ ":" ~
        (typeOfP[IntegerType] | typeOfP[IndexType])
    ).flatMap((pred, lhsName, rhsName, typ) =>
      ICmpPredicate.fromString(pred) match
        case None            => Fail(s"unknown llvm.icmp predicate '$pred'")
        case Some(predicate) =>
          operandP(lhsName, typ).flatMap(lhs =>
            operandP(rhsName, typ).flatMap(rhs =>
              resultP(resNames.head, I1).map(res =>
                ICmp(
                  lhs.asInstanceOf[Operand[IntegerType | IndexType]],
                  rhs.asInstanceOf[Operand[IntegerType | IndexType]],
                  res,
                  predicate,
                )
              )
            )
          )
    )

case class Load(
    addr: Operand[Ptr],
    res: Result[Attribute],
) extends DerivedOperation["llvm.load"]
    with AssemblyFormat["$addr attr-dict `:` type($addr) `->` type($res)"]
    derives OpDefs

case class Store(
    value: Operand[Attribute],
    addr: Operand[Ptr],
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
    base: Operand[Ptr],
    dynamicIndices: Seq[Operand[IntegerType | IndexType]],
    res: Result[Ptr],
    rawConstantIndices: DenseArrayAttr,
    elem_type: Attribute,
) extends DerivedOperation["llvm.getelementptr"]
    with NoMemoryEffect derives OpDefs:

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
    container: Operand[Attribute],
    position: DenseArrayAttr,
    res: Result[Attribute],
) extends DerivedOperation["llvm.extractvalue"] derives OpDefs:

  override def customPrint(printer: Printer): Unit =
    printer.print(name, " ", container, "[")
    printer.printListF(
      position.data.data.collect { case i: IntegerAttr => i.value.value },
      idx => printer.print(idx.toString),
      sep = ", ",
    )
    printer.print("] : ", container.typ)

case class InsertValue(
    value: Operand[Attribute],
    container: Operand[Attribute],
    position: DenseArrayAttr,
    res: Result[Attribute],
) extends DerivedOperation["llvm.insertvalue"] derives OpDefs:

  override def customPrint(printer: Printer): Unit =
    printer.print(name, " ", value, ", ", container, "[")
    printer.printListF(
      position.data.data.collect { case i: IntegerAttr => i.value.value },
      idx => printer.print(idx.toString),
      sep = ", ",
    )
    printer.print("] : ", res.typ)

case class PtrToInt(
    in: Operand[Ptr],
    out: Result[IntegerType | IndexType],
) extends DerivedOperation["llvm.ptrtoint"] derives OpDefs

case class IntToPtr(
    in: Operand[IntegerType | IndexType],
    out: Result[Ptr],
) extends DerivedOperation["llvm.inttoptr"] derives OpDefs

case class Call(
    callee: SymbolRefAttr,
    operandss: Seq[Operand[Attribute]],
    resultss: Seq[Result[Attribute]],
) extends DerivedOperation["llvm.call"] derives OpDefs:

  override def customPrint(printer: Printer): Unit =
    printer.print(name, " @", callee.rootRef.data, "(")
    printer.printList(operandss)
    printer.print(") : (")
    printer.printListF(operandss.map(_.typ), printer.print, sep = ", ")
    printer.print(") -> ")
    resultss.map(_.typ) match
      case Seq() =>
        printer.print("()")
      case Seq(single) =>
        printer.print(single)
      case many =>
        printer.printListF(many, printer.print, "(", ", ", ")")

case class Br(
    args: Seq[Operand[Attribute]],
    dest: Block,
) extends DerivedOperation["llvm.br"]
    with IsTerminator derives OpDefs

case class CondBr(
    condition: Operand[IntegerType],
    trueArgs: Seq[Operand[Attribute]],
    falseArgs: Seq[Operand[Attribute]],
    trueDest: Block,
    falseDest: Block,
) extends DerivedOperation["llvm.cond_br"]
    with IsTerminator derives OpDefs

case class Return(
    args: Seq[Operand[Attribute]]
) extends DerivedOperation["llvm.return"]
    with IsTerminator derives OpDefs

given OperationCustomParser[Func]:

  def parseResultTypes[$: P](using Parser): P[Seq[Attribute]] =
    ("->" ~ (parenTypeListP | typeP.map(Seq(_)))).orElse(Seq.empty)

  def parse[$: P](resNames: Seq[String])(using Parser): P[Func] =
    ("private".!.? ~ symbolRefAttrP ~
      (("(" ~ valueIdAndTypeP.rep(sep = ",") ~ ")")
        .flatMap((args: Seq[(String, Attribute)]) =>
          Pass(args.map(_._2)) ~ parseResultTypes ~
            ("attributes" ~ attributeDictionaryP).orElse(Map()) ~ regionP(args)
        ) |
        (
          parenTypeListP ~ parseResultTypes ~
            ("attributes" ~ attributeDictionaryP).orElse(Map()) ~ Pass(Region())
        ))).map {
      case (visibility, symbol, (argTypes, resTypes, attributes, body)) =>
        val f = Func(
          sym_name = symbol.rootRef,
          function_type = FunctionType(argTypes, resTypes),
          sym_visibility = visibility.map(StringData(_)),
          body = body,
        )
        f.attributes.addAll(attributes)
        f
    }

case class Func(
    sym_name: StringData,
    function_type: FunctionType,
    sym_visibility: Option[StringData],
    body: Region,
) extends DerivedOperation["llvm.func"]
    with IsolatedFromAbove
    with Symbol
    with SymbolTable derives OpDefs:

  override def customPrint(printer: Printer): Unit =
    val lprinter = printer.scoped
    lprinter.print("llvm.func ")
    sym_visibility.foreach { visibility =>
      lprinter.print(visibility.data)
      lprinter.print(" ")
    }
    lprinter.print("@")
    lprinter.print(sym_name.data)
    if body.blocks.isEmpty then
      lprinter.print("(")
      lprinter.printListF(function_type.inputs, lprinter.print, sep = ", ")
      lprinter.print(")")
      if function_type.outputs.nonEmpty then
        lprinter.print(" -> ")
        function_type.outputs match
          case ArrayAttribute(single) => lprinter.print(single)
          case many => lprinter.printList(many, "(", ", ", ")")
    else
      val entry = body.blocks.head
      lprinter.printListF(
        entry.arguments,
        lprinter.printArgument,
        "(",
        ", ",
        ")",
      )
      if function_type.outputs.nonEmpty then
        lprinter.print(" -> ")
        function_type.outputs match
          case ArrayAttribute(single) => lprinter.print(single)
          case many => lprinter.printList(many, "(", ", ", ")")
    if attributes.nonEmpty then
      lprinter.print(" attributes")
      lprinter.printOptionalAttrDict(attributes.toMap)
    if body.blocks.nonEmpty then
      val entry = body.blocks.head
      val others = body.blocks.tail
      lprinter.print(" {\n")
      lprinter.indented(entry.operations.foreach(lprinter.print))
      others.foreach(lprinter.print)
      lprinter.withIndent(lprinter.print("}"))

case class Sub(
    lhs: Operand[IntegerType | IndexType],
    rhs: Operand[IntegerType | IndexType],
    res: Result[IntegerType | IndexType],
    overflowFlags: Option[ArrayAttribute[StringData]] = None,
) extends DerivedOperation["llvm.sub"]
    with NoMemoryEffect derives OpDefs

case class SDiv(
    lhs: Operand[IntegerType | IndexType],
    rhs: Operand[IntegerType | IndexType],
    res: Result[IntegerType | IndexType],
) extends DerivedOperation["llvm.sdiv"]
    with NoMemoryEffect derives OpDefs

case class SRem(
    lhs: Operand[IntegerType | IndexType],
    rhs: Operand[IntegerType | IndexType],
    res: Result[IntegerType | IndexType],
) extends DerivedOperation["llvm.srem"]
    with NoMemoryEffect derives OpDefs

case class FSub(
    lhs: Operand[FloatType],
    rhs: Operand[FloatType],
    res: Result[FloatType],
) extends DerivedOperation["llvm.fsub"]
    with NoMemoryEffect derives OpDefs

case class FDiv(
    lhs: Operand[FloatType],
    rhs: Operand[FloatType],
    res: Result[FloatType],
) extends DerivedOperation["llvm.fdiv"]
    with NoMemoryEffect derives OpDefs

case class FRem(
    lhs: Operand[FloatType],
    rhs: Operand[FloatType],
    res: Result[FloatType],
) extends DerivedOperation["llvm.frem"]
    with NoMemoryEffect derives OpDefs

case class FNeg(
    operand: Operand[FloatType],
    res: Result[FloatType],
) extends DerivedOperation["llvm.fneg"]
    with NoMemoryEffect derives OpDefs

case class FCmp(
    lhs: Operand[FloatType],
    rhs: Operand[FloatType],
    res: Result[IntegerType],
    predicate: FCmpPredicate,
) extends DerivedOperation["llvm.fcmp"]
    with NoMemoryEffect derives OpDefs

case class And(
    lhs: Operand[IntegerType | IndexType],
    rhs: Operand[IntegerType | IndexType],
    res: Result[IntegerType | IndexType],
) extends DerivedOperation["llvm.and"]
    with NoMemoryEffect derives OpDefs

case class Or(
    lhs: Operand[IntegerType | IndexType],
    rhs: Operand[IntegerType | IndexType],
    res: Result[IntegerType | IndexType],
) extends DerivedOperation["llvm.or"]
    with NoMemoryEffect derives OpDefs

case class XOr(
    lhs: Operand[IntegerType | IndexType],
    rhs: Operand[IntegerType | IndexType],
    res: Result[IntegerType | IndexType],
) extends DerivedOperation["llvm.xor"]
    with NoMemoryEffect derives OpDefs

case class Unreachable()
    extends DerivedOperation["llvm.unreachable"]
    with IsTerminator derives OpDefs

case class Alloca(
    arraySize: Operand[IntegerType | IndexType],
    res: Result[Ptr],
    elem_type: Attribute,
    alignment: Option[IntegerAttr] = None,
) extends DerivedOperation["llvm.alloca"] derives OpDefs

case class Trunc(
    in: Operand[IntegerType | IndexType],
    out: Result[IntegerType],
) extends DerivedOperation["llvm.trunc"]
    with NoMemoryEffect derives OpDefs

case class ZExt(
    in: Operand[IntegerType | IndexType],
    out: Result[IntegerType | IndexType],
) extends DerivedOperation["llvm.zext"]
    with NoMemoryEffect derives OpDefs

case class SExt(
    in: Operand[IntegerType | IndexType],
    out: Result[IntegerType | IndexType],
) extends DerivedOperation["llvm.sext"]
    with NoMemoryEffect derives OpDefs

case class SIToFP(
    in: Operand[IntegerType | IndexType],
    out: Result[FloatType],
) extends DerivedOperation["llvm.sitofp"]
    with NoMemoryEffect derives OpDefs

case class FPToSI(
    in: Operand[FloatType],
    out: Result[IntegerType | IndexType],
) extends DerivedOperation["llvm.fptosi"]
    with NoMemoryEffect derives OpDefs

case class FPTrunc(
    in: Operand[FloatType],
    out: Result[FloatType],
) extends DerivedOperation["llvm.fptrunc"]
    with NoMemoryEffect derives OpDefs

case class FPExt(
    in: Operand[FloatType],
    out: Result[FloatType],
) extends DerivedOperation["llvm.fpext"]
    with NoMemoryEffect derives OpDefs

case class Shl(
    lhs: Operand[IntegerType | IndexType],
    rhs: Operand[IntegerType | IndexType],
    res: Result[IntegerType | IndexType],
    overflowFlags: Option[ArrayAttribute[StringData]] = None,
) extends DerivedOperation["llvm.shl"]
    with NoMemoryEffect derives OpDefs

case class LShr(
    lhs: Operand[IntegerType | IndexType],
    rhs: Operand[IntegerType | IndexType],
    res: Result[IntegerType | IndexType],
) extends DerivedOperation["llvm.lshr"]
    with NoMemoryEffect derives OpDefs

case class AShr(
    lhs: Operand[IntegerType | IndexType],
    rhs: Operand[IntegerType | IndexType],
    res: Result[IntegerType | IndexType],
) extends DerivedOperation["llvm.ashr"]
    with NoMemoryEffect derives OpDefs

case class Select(
    condition: Operand[IntegerType],
    trueValue: Operand[Attribute],
    falseValue: Operand[Attribute],
    res: Result[Attribute],
) extends DerivedOperation["llvm.select"]
    with NoMemoryEffect derives OpDefs

case class CallIndirect(
    callee: Operand[Ptr],
    operandss: Seq[Operand[Attribute]],
    resultss: Seq[Result[Attribute]],
) extends DerivedOperation["llvm.call_indirect"] derives OpDefs

val LLVMDialect = summonDialect[
  (Ptr, StructType, ArrayType),
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
      CallIndirect,
      Alloca,
      Br,
      CondBr,
      Return,
      Unreachable,
  ),
]
