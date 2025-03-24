package scair.clairV2.macros

import scala.quoted.*
import scair.ir.*
import scala.collection.mutable
import scair.scairdl.constraints.*
import scair.dialects.builtin.*
import scala.collection.mutable.ListBuffer

import scala.compiletime._
import scair.clairV2.mirrored.getDefImpl
import scala.deriving.Mirror
import scair.clairV2.codegen._

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

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||  ADT to Unverified conversion Macro  ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

import scala.quoted.*

inline def symbolicField[T](inline obj: T, inline fieldName: String): Any =
  ${ symbolicFieldImpl('obj, 'fieldName) }

def symbolicFieldImpl[T: Type](obj: Expr[T], fieldName: Expr[String])(using
    Quotes
): Expr[Any] = {
  import quotes.reflect.*

  fieldName.value match {
    case Some(name) =>
      val symbol = obj.asTerm.tpe.typeSymbol.fieldMember(name)
      Select(obj.asTerm, symbol).asExpr
    case None =>
      report.errorAndAbort(
        s"Field name ${fieldName.show} must be a known string at compile-time"
      )
  }
}

def ADTFlatInputMacro[Def <: OpInputDef: Type, T: Type](
    opInputDefs: Seq[Def],
    adtOpExpr: Expr[T]
)(using Quotes): Expr[ListType[DefinedInput[Def]]] = {
  import quotes.reflect.*
  val stuff = Expr.ofList(
    opInputDefs.map((d: Def) =>
      (d match
        case d: MayVariadicOpInputDef
            if (d.variadicity == Variadicity.Variadic) =>
          symbolicFieldImpl(adtOpExpr, Expr(d.name))
        case _ =>
          '{
            Seq(${
              symbolicFieldImpl(adtOpExpr, Expr(d.name))
                .asExprOf[DefinedInput[Def]]
            })
          }
      ).asExprOf[Seq[DefinedInput[Def]]]
    )
  )
  '{ ListType.from(${ stuff }.flatten) }
}

def fromADTOperationMacro[T: Type](
    opDef: OperationDef,
    adtOpExpr: Expr[T]
)(using
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

  val flatOperands = ADTFlatInputMacro(opDef.operands, adtOpExpr)

  /*________________*\
  \*-- SUCCESSORS --*/

  val flatSuccessors = ADTFlatInputMacro(opDef.successors, adtOpExpr)

  /*_____________*\
  \*-- RESULTS --*/

  val flatResults = ADTFlatInputMacro(opDef.results, adtOpExpr)

  /*_____________*\
  \*-- REGIONS --*/

  val flatRegions = ADTFlatInputMacro(opDef.regions, adtOpExpr)

  /*________________*\
  \*-- PROPERTIES --*/

  // partitioning ADT parameters by Properties
  val (propertyParams, _) = params.partition { sym =>
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

  '{
    val x = UnverifiedOp[T](
      name = ${ Expr(opDef.name) },
      operands = $flatOperands,
      successors = $flatSuccessors,
      results_types = ListType.empty[Attribute],
      regions = $flatRegions,
      dictionaryProperties = $propertySeqExpr,
      dictionaryAttributes = DictType.empty[String, Attribute]
    )
    x.results.addAll($flatResults)
    x
  }

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||  Unverified to ADT conversion Macro  ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

def fromUnverifiedOperationMacro[T: Type](
    opDef: OperationDef,
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
      item: Expr[Operand[Attribute]],
      index: Int
  ): Expr[Operand[A]] =
    val typeName = Type.of[A].toString()
    '{
      val value = $item.typ

      if (!value.isInstanceOf[A]) {
        throw new IllegalArgumentException(
          s"Type mismatch for operand at index ${${ Expr(index) }}: " +
            s"expected ${${ Expr(typeName) }}, " +
            s"but found ${value.getClass.getSimpleName}"
        )
      }
      $item.asInstanceOf[Operand[A]]
    }

  def generateCheckedOperandArgumentOfVariadic[A <: Attribute: Type](
      list: Expr[ListBuffer[Operand[Attribute]]],
      index: Int
  ): Expr[Seq[Operand[A]]] =
    val typeName = Type.of[A].toString()
    '{
      (for (item <- $list) yield {
        val value = item.typ

        if (!value.isInstanceOf[A]) {
          throw new IllegalArgumentException(
            s"Type mismatch for operand at index ${${ Expr(index) }}: " +
              s"expected ${${ Expr(typeName) }}, " +
              s"but found ${value.getClass.getSimpleName}"
          )
        }
        item.asInstanceOf[Operand[A]]
      }).toSeq
    }

  def generateCheckedResultArgument[A <: Attribute: Type](
      item: Expr[Result[Attribute]],
      index: Int,
      typeName: String
  ): Expr[Result[A]] =
    '{
      val value = $item.typ

      if (!value.isInstanceOf[A]) {
        throw new IllegalArgumentException(
          s"Type mismatch for result at index ${${ Expr(index) }}: " +
            s"expected ${${ Expr(typeName) }}, " +
            s"but found ${value.getClass.getSimpleName}"
        )
      }
      $item.asInstanceOf[Result[A]]
    }

  def generateCheckedResultArgumentOfVariadic[A <: Attribute: Type](
      list: Expr[ListBuffer[Result[Attribute]]],
      index: Int,
      typeName: String
  ): Expr[Seq[Operand[A]]] =
    '{
      (for (item <- $list) yield {
        val value = item.typ

        if (!value.isInstanceOf[A]) {
          throw new IllegalArgumentException(
            s"Type mismatch for variadic result at index ${${ Expr(index) }}: " +
              s"expected ${${ Expr(typeName) }}, " +
              s"but found ${value.getClass.getSimpleName}"
          )
        }
        item.asInstanceOf[Result[A]]
      }).toSeq
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

  // used to get (from, to) indices for single variadic cases, do not use with multivariadics
  def getPartitions(x: Seq[String], fullLength: Int): (Int, Int) =
    val (preceeding, following) = x.splitAt(x.indexOf("Var"))
    (preceeding.length, fullLength - following.length)

  def operandSegmentSizes(
      variadicsNumber: Int,
      noOfOperands: Int
  ): Expr[Seq[Int]] =
    if (variadicsNumber > 1)
      '{
        val dictAttributes = $genExpr.dictionaryProperties

        if (!dictAttributes.contains("operandSegmentSizes"))
        then throw new Exception("Expected operandSegmentSizes property")

        val operandSegmentSizes_attr =
          dictAttributes("operandSegmentSizes") match {
            case right: DenseArrayAttr => right
            case _ =>
              throw new Exception(
                "Expected operandSegmentSizes to be a DenseArrayAttr"
              )
          }

        ParametrizedAttrConstraint[DenseArrayAttr](
          Seq(
            EqualAttr(IntegerType(IntData(32), Signless)),
            AllOf(
              Seq(
                BaseAttr[IntegerAttr](),
                ParametrizedAttrConstraint[IntegerAttr](
                  Seq(
                    BaseAttr[IntData](),
                    EqualAttr(IntegerType(IntData(32), Signless))
                  )
                )
              )
            )
          )
        ).verify(operandSegmentSizes_attr, ConstraintContext())

        if (operandSegmentSizes_attr.length != ${ Expr(noOfOperands) }) then
          throw new Exception(
            s"Expected operandSegmentSizes to have ${${ Expr(noOfOperands) }} elements, got ${operandSegmentSizes_attr.length}"
          )

        for (s <- operandSegmentSizes_attr) yield s match {
          case right: IntegerAttr => right.value.data.toInt
          case _ =>
            throw new Exception(
              "Unreachable exception as per above constraint check."
            )
        }
      }
    else '{ Seq() }

  def resultSegmentSizes(
      variadicsNumber: Int,
      noOfResults: Int
  ): Expr[Seq[Int]] =
    if (variadicsNumber > 1)
      '{
        val dictAttributes = $genExpr.dictionaryProperties

        if (!dictAttributes.contains("resultSegmentSizes"))
        then throw new Exception("Expected resultSegmentSizes property")

        val resultSegmentSizes_attr = dictAttributes(
          "resultSegmentSizes"
        ) match {
          case right: DenseArrayAttr => right
          case _ =>
            throw new Exception(
              "Expected resultSegmentSizes to be a DenseArrayAttr"
            )
        }

        ParametrizedAttrConstraint[DenseArrayAttr](
          Seq(
            EqualAttr(IntegerType(IntData(32), Signless)),
            AllOf(
              Seq(
                BaseAttr[IntegerAttr](),
                ParametrizedAttrConstraint[IntegerAttr](
                  Seq(
                    BaseAttr[IntData](),
                    EqualAttr(IntegerType(IntData(32), Signless))
                  )
                )
              )
            )
          )
        ).verify(resultSegmentSizes_attr, ConstraintContext())

        if (resultSegmentSizes_attr.length != ${ Expr(noOfResults) }) then
          throw new Exception(
            s"Expected resultSegmentSizes to have ${${ Expr(noOfResults) }} elements, got $${resultSegmentSizes_attr.length}"
          )

        for (s <- resultSegmentSizes_attr) yield s match {
          case right: IntegerAttr => right.value.data.toInt
          case _ =>
            throw new Exception(
              "Unreachable exception as per above constraint check."
            )
        }
      }
    else '{ Seq() }

  /*_____________*\
  \*-- OPERAND --*/

  // extracting and validating each input, the casting the Operand instance
  val operands = opDef.operands
  val variadicOperands = operands.count(_.variadicity != Variadicity.Single)
  val operandArgs = operands.zipWithIndex.map {
    case (OperandDef(name, tpe, variadicity), idx) =>
      tpe match
        case '[t] if TypeRepr.of[t] <:< TypeRepr.of[Attribute] =>
          variadicity match
            case Variadicity.Single =>
              generateCheckedOperandArgument[t & Attribute](
                '{ $genExpr.operands(${ Expr(idx) }) },
                idx
              )
            case Variadicity.Variadic =>
              if variadicOperands > 1 then
                val from = '{
                  ${
                    operandSegmentSizes(
                      variadicOperands,
                      operands.length
                    )
                  }
                    .slice(0, ${ Expr(idx) })
                    .fold(0)(_ + _)
                }
                val to = '{
                  $from + ${
                    operandSegmentSizes(
                      variadicOperands,
                      operands.length
                    )
                  }(
                    ${ Expr(idx) }
                  )
                }
                generateCheckedOperandArgumentOfVariadic[t & Attribute](
                  '{
                    $genExpr.operands
                      .slice($from, $to)
                  },
                  idx
                )
              else
                generateCheckedOperandArgumentOfVariadic[t & Attribute](
                  {
                    val (preceeding, following) =
                      operands
                        .map(_.variadicity)
                        .splitAt(opDef.operands.indexOf(Variadicity.Variadic))
                    '{
                      val (from, to) =
                        (
                          ${ Expr(preceeding.length) },
                          fullOperandsLength - ${
                            Expr(following.length - 1)
                          } // spliceAt Index keeps the element at that index
                        )
                      val fullOperandsLength = $genExpr.operands.length
                      $genExpr.operands
                        .slice(from, to)
                    }
                  },
                  idx
                )

  }
  /*________________*\
  \*-- SUCCESSORS --*/

  val (successorParams, restParams1) = params.partition { sym =>
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
      case AppliedType(_, List(AppliedType(tycon, _))) =>
        tycon =:= TypeRepr.of[Result]
      case AppliedType(tycon, _) =>
        tycon =:= TypeRepr.of[Result]
      case x => false
  }

  val variadicResultParams = resultParams.filter { sym =>
    sym.termRef.widenTermRefByName match
      case AppliedType(tycon, _) => tycon =:= TypeRepr.of[Variadic]
      case _                     => false
  }

  val resultExpectedTypes = resultParams.zipWithIndex.map { (sym, idx) =>
    sym.termRef.widenTermRefByName match
      // for Variadic[Operand[targ]]
      case AppliedType(tycon, List(AppliedType(_, List(targ)))) =>
        ("Var", targ, idx)
      // for Operand[targ]
      case AppliedType(tycon, List(targ)) => ("Sin", targ, idx)
      case _ =>
        report.errorAndAbort(
          s"Unexpected non-applied type: ${sym.termRef.widenTermRefByName.show}"
        )
  }

  val resultArgs = resultExpectedTypes.map {
    case (variadicity, expectedType, idx) =>
      val typeName = expectedType.typeSymbol.name
      val idxExpr = Expr(idx)

      variadicity match {
        case "Sin" =>
          expectedType.asType match
            case '[t] if TypeRepr.of[t] <:< TypeRepr.of[Attribute] =>
              generateCheckedResultArgument[t & Attribute](
                '{ $genExpr.results($idxExpr) },
                idx,
                typeName
              )

        case "Var" =>
          if (variadicResultParams.length > 1) {
            expectedType.asType match
              case '[t] if TypeRepr.of[t] <:< TypeRepr.of[Attribute] =>
                val from = '{
                  ${
                    resultSegmentSizes(
                      variadicResultParams.length,
                      resultParams.length
                    )
                  }
                    .slice(0, $idxExpr)
                    .fold(0)(_ + _)
                }
                val to = '{
                  $from + ${
                    resultSegmentSizes(
                      variadicResultParams.length,
                      resultParams.length
                    )
                  }(
                    $idxExpr
                  )
                }
                generateCheckedResultArgumentOfVariadic[t & Attribute](
                  '{
                    $genExpr.results
                      .slice($from, $to)
                  },
                  idx,
                  typeName
                )
          } else {
            val x = Expr(resultExpectedTypes.map(_._1))
            expectedType.asType match
              case '[t] if TypeRepr.of[t] <:< TypeRepr.of[Attribute] =>
                generateCheckedResultArgumentOfVariadic[t & Attribute](
                  '{
                    val fullResultsLength = $genExpr.results.length
                    val (preceeding, following) =
                      $x.splitAt($x.indexOf("Var"))
                    val (from, to) =
                      (
                        preceeding.length,
                        fullResultsLength - (following.length - 1) // spliceAt Index keeps the element at that index
                      )
                    $genExpr.results
                      .slice(from, to)
                  },
                  idx,
                  typeName
                )
          }
      }
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
  val lengthCheck = {
    val nOperands = opDef.operands.length
    val varOperLen = Expr(variadicOperands)
    val varResLen = Expr(variadicResultParams.length)

    '{
      val operands = $genExpr.operands
      val successors = $genExpr.successors
      val results = $genExpr.results
      val regions = $genExpr.regions
      val properties = $genExpr.dictionaryProperties

      $varOperLen match {
        case 0 =>
          if (operands.length != ${ Expr(variadicOperands) }) {
            throw new IllegalArgumentException(
              s"Expected ${${ Expr(variadicOperands) }} operands, but got ${operands.length}"
            )
          }
        case 1 =>
          if (${ Expr(variadicOperands) } < operands.length - 1) {
            throw new IllegalArgumentException(
              s"Expected ${${ Expr(variadicOperands) }} operands, but got ${operands.length}"
            )
          }
        case _ =>
          val operandSegmentSizesSum = ${
            operandSegmentSizes(
              variadicOperands,
              nOperands
            )
          }.fold(0)(_ + _)
          if (operandSegmentSizesSum != operands.length) then
            throw new Exception(
              s"Expected ${operandSegmentSizesSum} operands, got ${operands.length}"
            )
      }

      if (successors.length != ${ Expr(successorParams.length) }) {
        throw new IllegalArgumentException(
          s"Expected ${${ Expr(successorParams.length) }} successors, but got ${successors.length}"
        )
      }

      $varResLen match {
        case 0 =>
          if (results.length != ${ Expr(resultParams.length) }) {
            throw new IllegalArgumentException(
              s"Expected ${${ Expr(resultParams.length) }} results, but got ${results.length} a"
            )
          }
        case 1 =>
          if (${ Expr(resultParams.length) } < results.length - 1) {
            throw new IllegalArgumentException(
              s"Expected ${${ Expr(resultParams.length) }} results, but got ${results.length} b"
            )
          }
        case _ =>
          val resultSegmentSizesSum = ${
            resultSegmentSizes(
              variadicOperands,
              resultParams.length
            )
          }.fold(0)(_ + _)
          if (resultSegmentSizesSum != results.length) then
            throw new Exception(
              s"Expected ${resultSegmentSizesSum} results, got ${results.length} c"
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
  }

  // Combine all parameters in the correct order
  val allArgs = params.map { param =>
    val paramType = param.termRef.widenTermRefByName
    paramType match
      case AppliedType(tycon, List(AppliedType(tycon2, _))) =>
        if (tycon2 =:= TypeRepr.of[Operand]) {
          val idx = operands.indexWhere(_.name == param.name)
          operandArgs(idx)
        } else if (tycon2 =:= TypeRepr.of[Result]) {
          val idx = resultParams.indexOf(param)
          resultArgs(idx)
        } else {
          report.errorAndAbort(
            s"Unexpected Variadic type constructor: ${tycon2.show}"
          )
        }
      case AppliedType(tycon, List(targ)) =>
        if (tycon =:= TypeRepr.of[Operand]) {
          val idx = operands.indexWhere(_.name == param.name)
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

trait MLIRTrait[T] extends MLIRTraitI[T] {
  extension (op: T) override def MLIRTrait = this
}

object MLIRTrait {

  inline def derived[T]: MLIRTrait[T] = ${ derivedImpl[T] }

  def derivedImpl[T: Type](using Quotes): Expr[MLIRTrait[T]] =
    val opDef = getDefImpl[T]
    '{

      new MLIRTrait[T]:

        def getName: String = ${ Expr(opDef.name) }

        def constructUnverifiedOp(
            operands: ListType[Value[Attribute]] = ListType(),
            successors: ListType[scair.ir.Block] = ListType(),
            results_types: ListType[Attribute] = ListType(),
            regions: ListType[Region] = ListType(),
            dictionaryProperties: DictType[String, Attribute] =
              DictType.empty[String, Attribute],
            dictionaryAttributes: DictType[String, Attribute] =
              DictType.empty[String, Attribute]
        ): UnverifiedOp[T] = UnverifiedOp[T](
          name = ${ Expr(opDef.name) },
          operands = operands,
          successors = successors,
          results_types = results_types,
          regions = regions,
          dictionaryProperties = dictionaryProperties,
          dictionaryAttributes = dictionaryAttributes
        )

        def unverify(adtOp: T): UnverifiedOp[T] =
          ${ fromADTOperationMacro[T](opDef, '{ adtOp }) }

        def verify(unverOp: UnverifiedOp[T]): T =
          ${ fromUnverifiedOperationMacro[T](opDef, '{ unverOp }) }

        extension (op: T) override def MLIRTrait: MLIRTrait[T] = this

    }

}

inline def summonMLIRTraits[T <: Tuple]: Seq[MLIRTrait[_]] =
  inline erasedValue[T] match
    case _: (t *: ts) =>
      MLIRTrait.derived[t] +: summonMLIRTraits[ts]
    case _: EmptyTuple => Seq()

inline def summonDialect[T <: Tuple](
    attributes: Seq[AttributeObject]
): DialectV2 =
  new DialectV2(summonMLIRTraits[T], attributes)
