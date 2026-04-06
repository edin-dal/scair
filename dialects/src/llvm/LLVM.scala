package scair.dialects.llvm

import fastparse.*
import scair.Printer
import scair.clair.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.parse.*
import scair.parse.given
import scair.utils.*

case class Ptr() extends DerivedAttribute["llvm.ptr"] with TypeAttribute
    derives AttrDefs

final case class StructType(
    elems: Seq[TypeAttribute]
) extends ParametrizedAttribute
    with TypeAttribute:
  override def name: String = "llvm.struct"
  override def parameters: Seq[Attribute | Seq[Attribute]] = Seq(elems)

  override def printParameters(p: Printer): Unit =
    given indentLevel: Int = 0
    p.print("<(")
    p.printListF(elems, p.print, sep = ", ")
    p.print(")>")

given AttributeCompanion[StructType]:
  override def name: String = "llvm.struct"

  override def parse[$: P](using Parser): P[StructType] =
    P("<" ~ "(" ~ typeP.rep(sep = ",") ~ ")" ~ ">").map(elems =>
      StructType(elems.map(_.asInstanceOf[TypeAttribute]))
    )

final case class ArrayType(
    size: IntData,
    elem: TypeAttribute,
) extends ParametrizedAttribute
    with TypeAttribute:
  override def name: String = "llvm.array"
  override def parameters: Seq[Attribute | Seq[Attribute]] = Seq(size, elem)

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
    predicate: StringData,
    res: Result[IntegerType],
) extends DerivedOperation["llvm.icmp"] derives OpDefs:

  override def customPrint(printer: Printer): Unit =
    printer.print(name, " ")
    printer.print("\"", predicate.data, "\" ")
    printer.print(lhs, ", ", rhs, " : ", lhs.typ)

given OperationCustomParser[ICmp]:
  def parse[$: P](resNames: Seq[String])(using Parser): P[ICmp] =
    P(
      stringLiteralP ~ operandNameP ~ "," ~ operandNameP ~ ":" ~
        (typeOfP[IntegerType] | typeOfP[IndexType])
    ).flatMap((pred, lhsName, rhsName, typ) =>
      operandP(lhsName, typ).flatMap(lhs =>
        operandP(rhsName, typ).flatMap(rhs =>
          resultP(resNames.head, I1).map(res =>
            ICmp(
              lhs.asInstanceOf[Operand[IntegerType | IndexType]],
              rhs.asInstanceOf[Operand[IntegerType | IndexType]],
              StringData(pred),
              res,
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

private val gepDynamicIndexSentinel: BigInt = BigInt(Int.MinValue)

private def isDynamicGEPIndex(attr: IntegerAttr): Boolean =
  attr.value.value == gepDynamicIndexSentinel

case class GetElementPtr(
    base: Operand[Ptr],
    dynamicIndices: Seq[Operand[IntegerType | IndexType]],
    res: Result[Ptr],
    rawConstantIndices: DenseArrayAttr,
    elem_type: Attribute,
    gepFlags: Option[ArrayAttribute[StringData]] = None,
) extends DerivedOperation["llvm.getelementptr"]
    with NoMemoryEffect derives OpDefs:

  override def customVerify(): OK[Operation] =
    val rawIndices = rawConstantIndices.data.collect { case i: IntegerAttr => i }
    val numDynamicMarkers = rawIndices.count(isDynamicGEPIndex)
    if numDynamicMarkers != dynamicIndices.size then
      Err(
        s"llvm.getelementptr: rawConstantIndices contain $numDynamicMarkers dynamic markers but op has ${dynamicIndices.size} dynamic indices"
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
      position.data.collect { case i: IntegerAttr => i.value.value },
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
      position.data.collect { case i: IntegerAttr => i.value.value },
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
    val lprinter = printer.copy()
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
          case Seq(single) => lprinter.print(single)
          case many        => lprinter.printList(many, "(", ", ", ")")
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
          case Seq(single) => lprinter.print(single)
          case many        => lprinter.printList(many, "(", ", ", ")")
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

val LLVMDialect = summonDialect[
  (Ptr, StructType, ArrayType),
  (
      Func,
      Constant,
      Zero,
      Poison,
      Add,
      Mul,
      FAdd,
      FMul,
      ICmp,
      Load,
      Store,
      GetElementPtr,
      ExtractValue,
      InsertValue,
      PtrToInt,
      IntToPtr,
      Call,
      Br,
      CondBr,
      Return,
  ),
]
