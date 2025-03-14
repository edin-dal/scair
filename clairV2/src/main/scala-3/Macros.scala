package scair.clairV2.macros

import scala.quoted.*
import scair.ir.*
import scala.collection.mutable

// ░█████╗░ ██╗░░░░░ ░█████╗░ ██╗ ██████╗░ ██╗░░░██╗ ██████╗░
// ██╔══██╗ ██║░░░░░ ██╔══██╗ ██║ ██╔══██╗ ██║░░░██║ ╚════██╗
// ██║░░╚═╝ ██║░░░░░ ███████║ ██║ ██████╔╝ ╚██╗░██╔╝ ░░███╔═╝
// ██║░░██╗ ██║░░░░░ ██╔══██║ ██║ ██╔══██╗ ░╚████╔╝░ ██╔══╝░░
// ╚█████╔╝ ███████╗ ██║░░██║ ██║ ██║░░██║ ░░╚██╔╝░░ ███████╗
// ░╚════╝░ ╚══════╝ ╚═╝░░╚═╝ ╚═╝ ╚═╝░░╚═╝ ░░░╚═╝░░░ ╚══════╝

// ███╗░░░███╗ ░█████╗░ ░█████╗░ ██████╗░ ░█████╗░ ░██████╗
// ████╗░████║ ██╔══██╗ ██╔══██╗ ██╔══██╗ ██╔══██╗ ██╔════╝
// ██╔████╔██║ ███████║ ██║░░╚═╝ ██████╔╝ ██║░░██║ ╚█████╗░
// ██║╚██╔╝██║ ██╔══██║ ██║░░██╗ ██╔══██╗ ██║░░██║ ░╚═══██╗
// ██║░╚═╝░██║ ██║░░██║ ╚█████╔╝ ██║░░██║ ╚█████╔╝ ██████╔╝
// ╚═╝░░░░░╚═╝ ╚═╝░░╚═╝ ░╚════╝░ ╚═╝░░╚═╝ ░╚════╝░ ╚═════╝░

inline def typeToString[T]: String = ${ typeToStringImpl[T] }

def typeToStringImpl[T: Type](using Quotes): Expr[String] = {
  Expr(Type.show[T])
}

inline def getClassPath[T]: String = ${ getClassPathImpl[T] }

def getClassPathImpl[T: Type](using Quotes): Expr[String] = {
  import quotes.reflect.*
  val classSymbol = TypeRepr.of[T].typeSymbol
  val classPath = classSymbol.fullName
  Expr(classPath)
}

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||  Hook for UnverifiedOp Constructor  ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

inline def constructUnverifiedOpHook[T <: ADTOperation](
    operands: ListType[Value[Attribute]],
    successors: ListType[scair.ir.Block],
    results_types: ListType[Attribute],
    regions: ListType[Region],
    dictionaryProperties: DictType[String, Attribute],
    dictionaryAttributes: DictType[String, Attribute]
): UnverifiedOp[T] =
  ${
    constructUnverifiedOpHookMacro[T](
      'operands,
      'successors,
      'results_types,
      'regions,
      'dictionaryProperties,
      'dictionaryAttributes
    )
  }

def constructUnverifiedOpHookMacro[T <: ADTOperation: Type](
    operands: Expr[ListType[Value[Attribute]]],
    successors: Expr[ListType[scair.ir.Block]],
    results_types: Expr[ListType[Attribute]],
    regions: Expr[ListType[Region]],
    dictionaryProperties: Expr[DictType[String, Attribute]],
    dictionaryAttributes: Expr[DictType[String, Attribute]]
)(using Quotes): Expr[UnverifiedOp[T]] = {
  import quotes.reflect.*

  val typeRepr = TypeRepr.of[T]
  val name = Expr(typeRepr.typeSymbol.name.toLowerCase())

  '{
    UnverifiedOp[T](
      name = $name,
      operands = $operands,
      successors = $successors,
      results_types = $results_types,
      regions = $regions,
      dictionaryProperties = $dictionaryProperties,
      dictionaryAttributes = $dictionaryAttributes
    )
  }
}

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||  ADT to Unverified conversion Macro  ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

inline def fromADTOperation[T <: ADTOperation](gen: T): UnverifiedOp[T] =
  ${ fromADTOperationMacro[T]('gen) }

def fromADTOperationMacro[T <: ADTOperation: Type](adtOpExpr: Expr[T])(using
    Quotes
): Expr[UnverifiedOp[T]] =
  import quotes.reflect.*

  val typeRepr = TypeRepr.of[T]
  val typeSymbol = typeRepr.typeSymbol

  if (!typeSymbol.flags.is(Flags.Case))
    report.errorAndAbort("T must be a case class extending SpecificMachine")

  val params = typeSymbol.primaryConstructor.paramSymss.flatten

  /*______________*\
  \*-- OPERANDS --*/

  // partitioning ADT parameters by Operand
  val (operandParams, restParams0) = params.partition { sym =>
    sym.termRef.widenTermRefByName match
      case AppliedType(tycon, _) => tycon =:= TypeRepr.of[Operand]
      case _                     => false
  }

  // extracting operand instances from the ADT
  val operandExprs = operandParams.map { param =>
    val select = Select.unique(adtOpExpr.asTerm, param.name).asExpr
    '{ ${ select }.asInstanceOf[Operand[Attribute]] }
  }.toSeq

  // constructing a sequence of operands to construct the UnverifiedOp with
  val operandSeqExpr =
    if (operandExprs.isEmpty) '{
      ListType.empty[Operand[Attribute]]
    }
    else
      '{
        ListType[Operand[Attribute]](${ Varargs(operandExprs) }: _*)
      }

  /*________________*\
  \*-- SUCCESSORS --*/

  // partitioning ADT parameters by Successor
  val (successorParams, restParams1) = restParams0.partition { sym =>
    sym.termRef.widenTermRefByName match
      case tycon => tycon =:= TypeRepr.of[scair.ir.Block]
  }

  // extracting successor instances from the ADT
  val successorExprs = successorParams.map { param =>
    val select = Select.unique(adtOpExpr.asTerm, param.name).asExpr
    '{ ${ select }.asInstanceOf[scair.ir.Block] }
  }.toSeq

  // constructing a sequence of successors to construct the UnverifiedOp with
  val successorSeqExpr =
    if (successorExprs.isEmpty) '{
      ListType.empty[scair.ir.Block]
    }
    else
      '{
        ListType[scair.ir.Block](${ Varargs(successorExprs) }: _*)
      }

  /*_____________*\
  \*-- RESULTS --*/

  // partitioning ADT parameters by Result
  val (resultParams, restParams2) = restParams1.partition { sym =>
    sym.termRef.widenTermRefByName match
      case AppliedType(tycon, _) => tycon =:= TypeRepr.of[Result]
      case x                     => false
  }

  // extracting result instances from the ADT
  val resultExprs = resultParams.map { param =>
    val select = Select.unique(adtOpExpr.asTerm, param.name).asExpr
    '{ ${ select }.asInstanceOf[Result[Attribute]] }
  }.toSeq

  // constructing a sequence of results to construct the UnverifiedOp with
  val resultSeqExpr =
    if (resultExprs.isEmpty) '{
      ListType.empty[Result[Attribute]]
    }
    else
      '{
        ListType[Result[Attribute]](${ Varargs(resultExprs) }: _*)
      }

  /*_____________*\
  \*-- REGIONS --*/

  // partitioning ADT parameters by Regions
  val (regionParams, restParams3) = restParams2.partition { sym =>
    sym.termRef.widenTermRefByName match
      case tycon => tycon =:= TypeRepr.of[Region]
  }

  // extracting region instances from the ADT
  val regionExprs = regionParams.map { param =>
    val select = Select.unique(adtOpExpr.asTerm, param.name).asExpr
    '{ ${ select }.asInstanceOf[Region] }
  }.toSeq

  // constructing a sequence of regions to construct the UnverifiedOp with
  val regionSeqExpr =
    if (regionExprs.isEmpty) '{
      ListType.empty[Region]
    }
    else
      '{
        ListType[Region](${ Varargs(regionExprs) }: _*)
      }

  /*________________*\
  \*-- PROPERTIES --*/

  // partitioning ADT parameters by Properties
  val (propertyParams, _) = restParams3.partition { sym =>
    sym.termRef.widenTermRefByName match
      case AppliedType(tycon, _) => tycon =:= TypeRepr.of[Property]
      case x                     => false
  }

  // extracting property instances from the ADT
  val propertyExprs = propertyParams.map { param =>
    val select = Select.unique(adtOpExpr.asTerm, param.name).asExpr
    val name = Expr(param.name)
    val y = '{ ${ select }.asInstanceOf[Property[Attribute]] }
    '{ (${ name }, ${ y }.typ) }
  }.toSeq

  // constructing a sequence of properties to construct the UnverifiedOp with
  val propertySeqExpr =
    if (propertyExprs.isEmpty) '{
      DictType.empty[String, Attribute]
    }
    else
      '{
        DictType(${ Varargs(propertyExprs) }: _*)
      }

  /*_________________*\
  \*-- CONSTRUCTOR --*/

  val opName = Expr(typeSymbol.name.toLowerCase())

  '{
    val x = UnverifiedOp[T](
      name = $opName,
      operands = $operandSeqExpr,
      successors = $successorSeqExpr,
      results_types = ListType.empty[Attribute],
      regions = $regionSeqExpr,
      dictionaryProperties = $propertySeqExpr,
      dictionaryAttributes = DictType.empty[String, Attribute]
    )
    x.results.addAll($resultSeqExpr)
    x
  }

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||  Unverified to ADT conversion Macro  ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

inline def fromUnverifiedOperation[T <: ADTOperation](gen: UnverifiedOp[T]): T =
  ${ fromUnverifiedOperationMacro[T]('gen) }

def fromUnverifiedOperationMacro[T <: ADTOperation: Type](
    genExpr: Expr[UnverifiedOp[T]]
)(using Quotes): Expr[T] =
  import quotes.reflect.*

  val typeRepr = TypeRepr.of[T]
  val typeSymbol = typeRepr.typeSymbol

  // checking if the class is a case class
  if (!typeSymbol.flags.is(Flags.Case))
    report.errorAndAbort("T must be a case class extending SpecificMachine")

  // getting the parameters out of the constructor
  val params = typeSymbol.primaryConstructor.paramSymss.flatten

  /*_____________*\
  \*-- HELPERS --*/

  // extract generic type T from Operand[T], Result[T] etc.
  def extractGenericType(applied: AppliedType): TypeRepr =
    applied match
      case AppliedType(_, List(targ)) => targ
      case _ =>
        report.errorAndAbort(
          s"Could not extract generic type from ${applied.show}"
        )

  // type checking and casting an operand
  def generateCheckedOperandArgument[A <: Attribute: Type](
      list: Expr[Seq[Operand[Attribute]]],
      index: Int,
      typeName: String
  ): Expr[Operand[A]] =
    '{
      val item = $list(${ Expr(index) })
      val value = item.typ

      if (!value.isInstanceOf[A]) {
        throw new IllegalArgumentException(
          s"Type mismatch for operand at index ${${ Expr(index) }}: " +
            s"expected ${${ Expr(typeName) }}, " +
            s"but found ${value.getClass.getSimpleName}"
        )
      }
      item.asInstanceOf[Operand[A]]
    }

  def generateCheckedResultArgument[A <: Attribute: Type](
      list: Expr[Seq[Result[Attribute]]],
      index: Int,
      typeName: String
  ): Expr[Result[A]] =
    '{
      val item = $list(${ Expr(index) })
      val value = item.typ

      if (!value.isInstanceOf[A]) {
        throw new IllegalArgumentException(
          s"Type mismatch for operand at index ${${ Expr(index) }}: " +
            s"expected ${${ Expr(typeName) }}, " +
            s"but found ${value.getClass.getSimpleName}"
        )
      }
      item.asInstanceOf[Result[A]]
    }

  def generateCheckedPropertyArgument[A <: Attribute: Type](
      list: Expr[DictType[String, Attribute]],
      propName: String,
      typeName: String
  ): Expr[Property[A]] =
    '{
      val value = $list(${ Expr(propName) })

      if (!value.isInstanceOf[A]) {
        throw new IllegalArgumentException(
          s"Type mismatch for property \"${${ Expr(propName) }}\": " +
            s"expected ${${ Expr(typeName) }}, " +
            s"but found ${value.getClass.getSimpleName}"
        )
      }
      Property[A](value.asInstanceOf[A])
    }

  /*_____________*\
  \*-- OPERAND --*/

  // grouping parameters by their type
  val (operandParams, restParams0) = params.partition { sym =>
    sym.termRef.widenTermRefByName match
      case AppliedType(tycon, _) => tycon =:= TypeRepr.of[Operand]
      case _                     => false
  }

  // get the expected types for all input parameters
  val operandExpectedTypes = operandParams.map { sym =>
    sym.termRef.widenTermRefByName match
      case applied: AppliedType => extractGenericType(applied)
      case _ =>
        report.errorAndAbort(
          s"Unexpected non-applied type: ${sym.termRef.widenTermRefByName.show}"
        )
  }

  // extracting and validating each input, the re-creating the Input instance
  val operandArgs = operandExpectedTypes.zipWithIndex.map {
    case (expectedType, idx) =>
      val typeName = expectedType.typeSymbol.name

      expectedType.asType match
        case '[t] if TypeRepr.of[t] <:< TypeRepr.of[Attribute] =>
          generateCheckedOperandArgument[t & Attribute](
            '{ $genExpr.operands.toSeq },
            idx,
            typeName
          )
  }

  /*________________*\
  \*-- SUCCESSORS --*/

  val (successorParams, restParams1) = restParams0.partition { sym =>
    sym.termRef.widenTermRefByName match
      case tycon => tycon =:= TypeRepr.of[scair.ir.Block]
  }

  val successorArgs = successorParams.zipWithIndex.map { (param, idx) =>
    '{ $genExpr.successors(${ Expr(idx) }) }
  }

  /*_____________*\
  \*-- RESULTS --*/

  val (resultParams, restParams2) = restParams1.partition { sym =>
    sym.termRef.widenTermRefByName match
      case AppliedType(tycon, _) => tycon =:= TypeRepr.of[Result]
      case x                     => false
  }

  val resultExpectedTypes = resultParams.map { sym =>
    sym.termRef.widenTermRefByName match
      case applied: AppliedType => extractGenericType(applied)
      case _ =>
        report.errorAndAbort(
          s"Unexpected non-applied type: ${sym.termRef.widenTermRefByName.show}"
        )
  }

  // extracting and validating each input, the re-creating the Input instance
  val resultArgs = resultExpectedTypes.zipWithIndex.map {
    case (expectedType, idx) =>
      val typeName = expectedType.typeSymbol.name

      expectedType.asType match
        case '[t] if TypeRepr.of[t] <:< TypeRepr.of[Attribute] =>
          generateCheckedResultArgument[t & Attribute](
            '{ $genExpr.results.toSeq },
            idx,
            typeName
          )
  }

  /*_____________*\
  \*-- REGIONS --*/

  val (regionParams, restParams3) = restParams2.partition { sym =>
    sym.termRef.widenTermRefByName match
      case tycon => tycon =:= TypeRepr.of[Region]
  }

  val regionsArgs = regionParams.zipWithIndex.map { (param, idx) =>
    '{ $genExpr.regions(${ Expr(idx) }) }
  }

  /*________________*\
  \*-- PROPERTIES --*/

  val (propertyParams, _) = restParams3.partition { sym =>
    sym.termRef.widenTermRefByName match
      case AppliedType(tycon, _) => tycon =:= TypeRepr.of[Property]
      case x                     => false
  }

  val propertyExpectedTypes = propertyParams.map { sym =>
    sym.termRef.widenTermRefByName match
      case applied: AppliedType =>
        (sym.name, extractGenericType(applied))
      case _ =>
        report.errorAndAbort(
          s"Unexpected non-applied type: ${sym.termRef.widenTermRefByName.show}"
        )
  }

  // extracting and validating each input, the re-creating the Input instance
  val propertyArgs = propertyExpectedTypes.map {
    case (propName, expectedType) =>
      val typeName = expectedType.typeSymbol.name

      expectedType.asType match
        case '[t] if TypeRepr.of[t] <:< TypeRepr.of[Attribute] =>
          generateCheckedPropertyArgument[t & Attribute](
            '{ $genExpr.dictionaryProperties },
            propName,
            typeName
          )
  }

  /*__________________*\
  \*-- LENGTH CHECK --*/

  // checking that lengths of inputs in generalized machine and case class are the same
  val lengthCheck = '{

    val operands = $genExpr.operands
    val successors = $genExpr.successors
    val results = $genExpr.results
    val regions = $genExpr.regions
    val properties = $genExpr.dictionaryProperties

    if (operands.length != ${ Expr(operandParams.length) }) {
      throw new IllegalArgumentException(
        s"Expected ${${ Expr(operandParams.length) }} operands, but got ${operands.length}"
      )
    }

    if (successors.length != ${ Expr(successorParams.length) }) {
      throw new IllegalArgumentException(
        s"Expected ${${ Expr(successorParams.length) }} successors, but got ${successors.length}"
      )
    }

    if (results.length != ${ Expr(resultParams.length) }) {
      throw new IllegalArgumentException(
        s"Expected ${${ Expr(resultParams.length) }} results, but got ${results.length}"
      )
    }

    if (regions.length != ${ Expr(regionParams.length) }) {
      throw new IllegalArgumentException(
        s"Expected ${${ Expr(regionParams.length) }} regions, but got ${regions.length}"
      )
    }

    if (properties.size != ${ Expr(propertyParams.length) }) {
      throw new IllegalArgumentException(
        s"Expected ${${ Expr(propertyParams.length) }} properties, but got ${properties.size}"
      )
    }
  }

  // Combine all parameters in the correct order
  val allArgs = params.map { param =>
    val paramType = param.termRef.widenTermRefByName
    paramType match
      case AppliedType(tycon, _) =>
        if (tycon =:= TypeRepr.of[Operand]) {
          val idx = operandParams.indexOf(param)
          operandArgs(idx)
        } else if (tycon =:= TypeRepr.of[Result]) {
          val idx = resultParams.indexOf(param)
          resultArgs(idx)
        } else if (tycon =:= TypeRepr.of[Property]) {
          val idx = propertyParams.indexOf(param)
          propertyArgs(idx)
        } else {
          report.errorAndAbort(s"Unexpected type constructor: ${tycon.show}")
        }
      case tycon =>
        if (tycon =:= TypeRepr.of[scair.ir.Block]) {
          val idx = successorParams.indexOf(param)
          successorArgs(idx)
        } else if (tycon =:= TypeRepr.of[Region]) {
          val idx = regionParams.indexOf(param)
          regionsArgs(idx)
        } else {
          report.errorAndAbort(s"Unexpected parameter type: ${tycon.show}")
        }
  }

  // creates a new instance of the ADT op
  val args = allArgs.map(_.asTerm)
  val constructorExpr =
    Select(New(TypeTree.of[T]), typeSymbol.primaryConstructor)
      .appliedToArgs(args)
      .asExprOf[T]

  '{
    $lengthCheck
    $constructorExpr
  }

  /*≡==--==≡≡≡≡==--=≡≡*\
||    MLIR TRAIT    ||
\*≡==---==≡≡==---==≡*/

object MLIRTrait {

  inline def derived[T <: ADTOperation]: MLIRTrait[T] =
    new MLIRTrait[T]:

      def constructUnverifiedOp(
          operands: ListType[Value[Attribute]] = ListType(),
          successors: ListType[scair.ir.Block] = ListType(),
          results_types: ListType[Attribute] = ListType(),
          regions: ListType[Region] = ListType(),
          dictionaryProperties: DictType[String, Attribute] =
            DictType.empty[String, Attribute],
          dictionaryAttributes: DictType[String, Attribute] =
            DictType.empty[String, Attribute]
      ): UnverifiedOp[T] =
        constructUnverifiedOpHook(
          operands,
          successors,
          results_types,
          regions,
          dictionaryProperties,
          dictionaryAttributes
        )

      def unverify(adtOp: T): UnverifiedOp[T] = fromADTOperation[T](adtOp)

      def verify(unverOp: UnverifiedOp[T]): T =
        fromUnverifiedOperation[T](unverOp)

}

trait MLIRTrait[T <: ADTOperation] {

  def constructUnverifiedOp(
      operands: ListType[Value[Attribute]] = ListType(),
      successors: ListType[scair.ir.Block] = ListType(),
      results_types: ListType[Attribute] = ListType(),
      regions: ListType[Region] = ListType(),
      dictionaryProperties: DictType[String, Attribute] =
        DictType.empty[String, Attribute],
      dictionaryAttributes: DictType[String, Attribute] =
        DictType.empty[String, Attribute]
  ): UnverifiedOp[T]

  def unverify(adtOp: T): UnverifiedOp[T]

  def verify(unverOp: UnverifiedOp[T]): T

}
