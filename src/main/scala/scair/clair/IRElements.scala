package scair.clair

import scala.collection.mutable

// ██╗ ██████╗░
// ██║ ██╔══██╗
// ██║ ██████╔╝
// ██║ ██╔══██╗
// ██║ ██║░░██║
// ╚═╝ ╚═╝░░╚═╝

// ███████╗ ██╗░░░░░ ███████╗ ███╗░░░███╗ ███████╗ ███╗░░██╗ ████████╗ ░██████╗
// ██╔════╝ ██║░░░░░ ██╔════╝ ████╗░████║ ██╔════╝ ████╗░██║ ╚══██╔══╝ ██╔════╝
// █████╗░░ ██║░░░░░ █████╗░░ ██╔████╔██║ █████╗░░ ██╔██╗██║ ░░░██║░░░ ╚█████╗░
// ██╔══╝░░ ██║░░░░░ ██╔══╝░░ ██║╚██╔╝██║ ██╔══╝░░ ██║╚████║ ░░░██║░░░ ░╚═══██╗
// ███████╗ ███████╗ ███████╗ ██║░╚═╝░██║ ███████╗ ██║░╚███║ ░░░██║░░░ ██████╔╝
// ╚══════╝ ╚══════╝ ╚══════╝ ╚═╝░░░░░╚═╝ ╚══════╝ ╚═╝░░╚══╝ ░░░╚═╝░░░ ╚═════╝░

/*≡≡=---=≡≡≡=---=≡≡*\
||      UTILS      ||
\*≡==----=≡=----==≡*/

val DictType = mutable.Map
type DictType[A, B] = mutable.Map[A, B]

val ListType = mutable.ListBuffer
type ListType[A] = mutable.ListBuffer[A]

/*≡≡=--=≡≡≡=--=≡≡*\
||     TYPES     ||
\*≡==---=≡=---==≡*/

abstract class Type(val id: String) {
  def get_import(): String
}

case class RegularType(val dialect: String, override val id: String)
    extends Type(id) {
  override def get_import(): String = s"import scair.dialects.${dialect}._\n"
}
// case class AnonType(override val id: String, val typ: RegularType) extends Type(id) {
//   override def get_import(): String = typ.get_import()
// }

/*≡≡=---==≡≡≡==---=≡≡*\
||    CONSTRAINTS    ||
\*≡==----==≡==----==≡*/

abstract class Constraint {
  def print(indent: Int): String
  def get_imports(): String
}

case class Equal(val typ: Type) extends Constraint {
  override def print(indent: Int): String =
    s"val ${typ.id.toLowerCase()}_check = EqualAttr(${typ.id})\n"

  override def get_imports(): String = typ.get_import()
}
case class Base(val typ: Type) extends Constraint {
  override def print(indent: Int): String =
    s"val ${typ.id.toLowerCase()}_check = BaseAttr[${typ.id}]()\n"

  override def get_imports(): String = typ.get_import()
}
case class Any(val typ: Seq[Type]) extends Constraint {
  override def print(indent: Int): String =
    s"val ${(for (x <- typ) yield x.id).mkString("_").toLowerCase()}_check = AnyOf(Seq(${(for (x <- typ)
        yield x.id).mkString(", ")}))\n"

  override def get_imports(): String =
    (for (x <- typ) yield x.get_import()).mkString("\n")
}
// case class Anon(val typ: Type) extends Constraint {
//   override def print(indent: Int): String =
//     s"val ${typ.id.toLowerCase()}_check = VarConstraint(${typ.id}, )\n"

//   override def get_imports(): String =

// }

/*≡≡=---===≡≡≡≡===---=≡≡*\
||  TYPES & CONTAINERS  ||
\*≡==----===≡≡===----==≡*/

// def get_imports_from(constraint: Constraint): String =
//   constraint match {
//     case Equal(typ) =>
//     case Base(typ) =>
//     case Any(typ) =>
//     case Anon(typ) =>
//   }

// def get_imports(op: OpInput): String = {
//   op match {
//     case x: Operand =>
//     case x: Result =>
//     case x: Region => ""
//     case x: Successor => ""
//     case x: OpProperty =>
//     case x: OpAttribute =>
//   }
// }

abstract class OpInput {}

case class Operand(val id: String, val const: Constraint) extends OpInput {}
case class Result(val id: String, val const: Constraint) extends OpInput {}
case class Region(val no: Int) extends OpInput {}
case class Successor(val no: Int) extends OpInput {}
case class OpProperty(val id: String, val const: Constraint) extends OpInput {}
case class OpAttribute(val id: String, val const: Constraint) extends OpInput {}

class Dialect(
    val name: String,
    val operations: ListType[Operation] = ListType(),
    val attributes: ListType[Attribute] = ListType()
) {
  def print(indent: Int): String = ""
}

/*≡≡=---=≡≡≡=---=≡≡*\
||   IR ELEMENTS   ||
\*≡==----=≡=----==≡*/

class Operation(
    val name: String,
    val className: String,
    val operands: Seq[Operand],
    val results: Seq[Result],
    val region_no: Region,
    val successor_no: Successor,
    val OpProperty: Seq[OpProperty],
    val OpAttribute: Seq[OpAttribute]
) {

  def get_imports(): Set[String] = {
    ((for (x <- operands) yield x.const.get_imports()) ++
      (for (x <- results) yield x.const.get_imports()) ++
      (for (x <- OpProperty) yield x.const.get_imports()) ++
      (for (x <- OpAttribute) yield x.const.get_imports())).toSet
  }

  def constraint_printer(indent: Int): String = {
    ((for (x <- operands) yield x.const.print(indent)) ++
      (for (x <- results) yield x.const.print(indent)) ++
      (for (x <- OpProperty) yield x.const.print(indent)) ++
      (for (x <- OpAttribute) yield x.const.print(indent))).toSet.mkString("\n")
  }

  def print(indent: Int): String = {
    val op =
      s"object ${className} extends DialectOperation {" +
        s"  override def name: String = \"${name}\"" +
        s"  override def factory: FactoryType = ${className}.apply" +
        s"}" +
        s"case class ${className}(" +
        s"  override val operands: ListType[Value[Attribute]] = ListType()," +
        s"  override val successors: ListType[Block] = ListType()," +
        s"  override val results: ListType[Value[Attribute]] = ListType()," +
        s"  override val regions: ListType[Region] = ListType()," +
        s"  override val dictionaryProperties: DictType[String, Attribute] =" +
        s"    DictType.empty[String, Attribute]," +
        s"  override val dictionaryAttributes: DictType[String, Attribute] =" +
        s"    DictType.empty[String, Attribute]" +
        s") extends RegisteredOperation(name = \"${name}\") {" +
        s"\n" +
        "  " + constraint_printer(indent) +
        s"\n" +
        s"  override def custom_verify(): Unit = (" +
        s"    successors.length," +
        s"    results.length," +
        s"    regions.length," +
        s"    dictionaryProperties.size," +
        s"    dictionaryAttributes.size" +
        s"  ) match {" +
        s"    case (0, 1, 0, 0, 1) =>" +
        s"      for (x <- operands) index_check.verify(x.typ, new ConstraintContext())" +
        s"      index_check.verify(results(0).typ, new ConstraintContext())" +
        s"      map_check.verify(" +
        s"        dictionaryAttributes.checkandget(\"map\", name, \"affine_map\")," +
        s"        new ConstraintContext()" +
        s"      )" +
        s"    case _ =>" +
        s"      throw new Exception(" +
        s"        \"Apply Operation must only contain at least 1 operand and exaclty 1 result of 'index' type, \" +" +
        s"        \"as well as attribute in a dictionary called 'map' of 'affine_map' type.\"" +
        s"      )" +
        s"  }" +
        s"}"
    op
  }
}

class Attribute(
    val name: String,
    val className: String,
    val operands: Seq[Operand],
    val typee: Int
) {
  def print(indent: Int): String = ""
}
