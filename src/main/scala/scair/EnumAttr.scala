package scair.EnumAttr
import fastparse._
import scair.dialects.builtin.{IntegerType, Signless, I32, I64}
import java.lang.StringBuilder
import scair.ir._

// ==--------------------------== //
//   Enum Attribute Inhertiance   //
// ==--------------------------== //

abstract class EnumAttrCase[T <: Attribute](
    val symbol: String,
    val typ: T
) extends ParametrizedAttribute(symbol, Seq(typ)) {
  def parse[$: P]: P[Attribute] = P(symbol.!).map(_ => this)
  override def custom_print = symbol
}

abstract class EnumAttr[T <: Attribute](
    val name: String,
    val cases: Seq[EnumAttrCase[T]],
    val typ: T
) {

  def parser[$: P](seq: Seq[EnumAttrCase[T]]): P[Attribute] = seq match {
    case x +: xs => P(x.parse | parser(xs))
    case Nil     => P(Fail)
  }

  def caseParser[$: P]: P[Attribute] = {
    // we want to order here by length in descending order to ensure
    // we hit all cases:
    //  if "v = P("asc" | "ascc")"
    //  then "parse("ascc", v(_))"
    //  returns "Success("asc")"
    //  but we need "Success("ascc")"
    parser(cases.sortBy(_.symbol.length)(Ordering[Int].reverse))
  }
}

// ==----------------------------== //
//   Specialized Case Inhertiance   //
// ==----------------------------== //

abstract class I32EnumAttrCase(override val symbol: String)
    extends EnumAttrCase[IntegerType](symbol, I32)

abstract class I32EnumAttr(
    override val name: String,
    override val cases: Seq[I32EnumAttrCase]
) extends EnumAttr[IntegerType](name, cases, I32)

abstract class I64EnumAttrCase(override val symbol: String)
    extends EnumAttrCase[IntegerType](symbol, I64)

abstract class I64EnumAttr(
    override val name: String,
    override val cases: Seq[I64EnumAttrCase]
) extends EnumAttr[IntegerType](name, cases, I64)

/*

class EnumAttrCaseInfo<string sym, int intVal, string strVal> {
  string symbol = sym;
  int value = intVal;
  string str = strVal;
}


class IntEnumAttrCaseBase<I intType, string sym, string strVal, int intVal> :
    EnumAttrCaseInfo<sym, intVal, strVal>,
    SignlessIntegerAttrBase<intType, "case " # strVal> {
  let predicate =
    CPred<"::llvm::cast<::mlir::IntegerAttr>($_self).getInt() == " # intVal>;
}


class I32EnumAttrCase<string sym, int val, string str = sym>
    : IntEnumAttrCaseBase<I32, sym, str, val>;

class I64EnumAttrCase<string sym, int val, string str = sym>
    : IntEnumAttrCaseBase<I64, sym, str, val>;

class IntEnumAttrBase<I intType, list<IntEnumAttrCaseBase> cases, string summary> :
    SignlessIntegerAttrBase<intType, summary> {

  let predicate = And<[
    SignlessIntegerAttrBase<intType, summary>.predicate,
    Or<!foreach(case, cases, case.predicate)>]>;
}


class IntEnumAttr<I intType, string name, string summary,
                  list<IntEnumAttrCaseBase> cases> :
  EnumAttrInfo<name, cases,
    IntEnumAttrBase<intType, cases,
      !if(!empty(summary), "allowed " # intType.summary # " cases: " #
          !interleave(!foreach(case, cases, case.value), ", "),
          summary)>> {

  let parameterParser = [{[&]() -> ::mlir::FailureOr<}] # cppType # [{> {
    auto loc = $_parser.getCurrentLocation();
    ::llvm::StringRef enumKeyword;
    if (::mlir::failed($_parser.parseKeyword(&enumKeyword)))
      return ::mlir::failure();
    auto maybeEnum = }] # cppNamespace # "::" #
                          stringToSymbolFnName # [{(enumKeyword);
    if (maybeEnum)
      return *maybeEnum;
    return {(::mlir::LogicalResult)($_parser.emitError(loc) << "expected " }] #
    [{<< "}] # cppType # [{" << " to be one of: " << }] #
    !interleave(!foreach(enum, enumerants, "\"" # enum.str # "\""),
                [{ << ", " << }]) # [{)};
  }()}];

  let parameterPrinter = "$_printer << " # symbolToStringFnName # "($_self)";
}


class I32EnumAttr<string name, string summary, list<I32EnumAttrCase> cases> :
    IntEnumAttr<I32, name, summary, cases> {

  let underlyingType = "uint32_t";
}
class I64EnumAttr<string name, string summary, list<I64EnumAttrCase> cases> :
    IntEnumAttr<I64, name, summary, cases> {

  let underlyingType = "uint64_t";
}


class EnumParameter<EnumAttrInfo enumInfo>
    : AttrParameter<enumInfo.cppNamespace # "::" # enumInfo.className,
                    "an enum of type " # enumInfo.className> {

  let parser = enumInfo.parameterParser;
  let printer = enumInfo.parameterPrinter;
}


class EnumAttr<Dialect dialect, EnumAttrInfo enumInfo, string name = "",
               list <Trait> traits = []>
    : AttrDef<dialect, enumInfo.className, traits> {

  let summary = enumInfo.summary;
  let description = enumInfo.description;

  // The backing enumeration.
  EnumAttrInfo enum = enumInfo;

  // Inherit the C++ namespace from the enum.
  let cppNamespace = enumInfo.cppNamespace;

  // Define a constant builder for the attribute to convert from C++ enums.
  let constBuilderCall = cppNamespace # "::" # cppClassName #
                         "::get($_builder.getContext(), $0)";

  // Op attribute getters should return the underlying C++ enum type.
  let returnType = enumInfo.cppNamespace # "::" # enumInfo.className;

  // Convert from attribute to the underlying C++ type in op getters.
  let convertFromStorage = "$_self.getValue()";

  // The enum attribute has one parameter: the C++ enum value.
  let parameters = (ins EnumParameter<enumInfo>:$value);

  // If a mnemonic was provided, use it to generate a custom assembly format.
  let mnemonic = name;

  // The default assembly format for enum attributes. Selected to best work with
  // operation assembly formats.
  let assemblyFormat = "$value";
}

 */
