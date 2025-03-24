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

  // extracting property instances from the ADT
  val propertyExprs = opDef.properties.map { case OpPropertyDef(name, tpe) =>
    val select = Select.unique(adtOpExpr.asTerm, name).asExpr
    val y = '{ ${ select }.asInstanceOf[Property[Attribute]] }
    '{ (${ Expr(name) }, ${ y }.typ) }
  }.toSeq

  // constructing a sequence of properties to construct the UnverifiedOp with
  val propertySeqExpr = '{ DictType(${ Varargs(propertyExprs) }: _*) }

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
      index: Int
  ): Expr[Result[A]] =
    val typeName = Type.of[A].toString()
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
      index: Int
  ): Expr[Seq[Operand[A]]] =
    val typeName = Type.of[A].toString()
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
      propName: String
  ): Expr[Property[A]] =
    val typeName = Type.of[A].toString()
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

  val successors = opDef.successors

  val successorArgs = successors.zipWithIndex.map { (param, idx) =>
    '{ $genExpr.successors(${ Expr(idx) }) }
  }

  /*_____________*\
  \*-- RESULTS --*/

  val results = opDef.results
  val variadicResults = results.count(_.variadicity != Variadicity.Single)
  val resultArgs = results.zipWithIndex.map {
    case (ResultDef(name, tpe, variadicity), idx) =>
      tpe match
        case '[t] if TypeRepr.of[t] <:< TypeRepr.of[Attribute] =>
          variadicity match
            case Variadicity.Single =>
              generateCheckedResultArgument[t & Attribute](
                '{ $genExpr.results(${ Expr(idx) }) },
                idx
              )
            case Variadicity.Variadic =>
              if variadicResults > 1 then
                val from = '{
                  ${
                    resultSegmentSizes(
                      variadicResults,
                      results.length
                    )
                  }
                    .slice(0, ${ Expr(idx) })
                    .fold(0)(_ + _)
                }
                val to = '{
                  $from + ${
                    resultSegmentSizes(
                      variadicResults,
                      results.length
                    )
                  }(
                    ${ Expr(idx) }
                  )
                }
                generateCheckedResultArgumentOfVariadic[t & Attribute](
                  '{
                    $genExpr.results
                      .slice($from, $to)
                  },
                  idx
                )
              else
                generateCheckedResultArgumentOfVariadic[t & Attribute](
                  {
                    val (preceeding, following) =
                      results
                        .map(_.variadicity)
                        .splitAt(opDef.results.indexOf(Variadicity.Variadic))
                    '{
                      val (from, to) =
                        (
                          ${ Expr(preceeding.length) },
                          fullResultsLength - ${
                            Expr(following.length - 1)
                          } // spliceAt Index keeps the element at that index
                        )
                      val fullResultsLength = $genExpr.results.length
                      $genExpr.results
                        .slice(from, to)
                    }
                  },
                  idx
                )
  }

  /*_____________*\
  \*-- REGIONS --*/

  val regions = opDef.regions

  val regionsArgs = regions.zipWithIndex.map { (param, idx) =>
    '{ $genExpr.regions(${ Expr(idx) }) }
  }

  /*________________*\
  \*-- PROPERTIES --*/

  val properties = opDef.properties

  // extracting and validating each input, the re-creating the Input instance
  val propertyArgs = properties.map { case OpPropertyDef(name, tpe) =>
    tpe match
      case '[t] if TypeRepr.of[t] <:< TypeRepr.of[Attribute] =>
        generateCheckedPropertyArgument[t & Attribute](
          '{ $genExpr.dictionaryProperties },
          name
        )
  }

  /*__________________*\
  \*-- LENGTH CHECK --*/

  // checking that lengths of inputs in generalized machine and case class are the same
  val lengthCheck = {
    val OperLen = operands.length
    val varOperLen = Expr(variadicOperands)
    val ResLen = results.length
    val varResLen = Expr(variadicResults)
    val SuccLen = successors.length
    val RegLen = regions.length
    val PropLen = properties.length

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
              OperLen
            )
          }.fold(0)(_ + _)
          val SuccLen = successors.length
          if (operandSegmentSizesSum != operands.length) then
            throw new Exception(
              s"Expected ${operandSegmentSizesSum} operands, got ${operands.length}"
            )
      }

      if (successors.length != ${ Expr(SuccLen) }) {
        throw new IllegalArgumentException(
          s"Expected ${${ Expr(SuccLen) }} successors, but got ${successors.length}"
        )
      }

      $varResLen match {
        case 0 =>
          if (results.length != ${ Expr(ResLen) }) {
            throw new IllegalArgumentException(
              s"Expected ${${ Expr(ResLen) }} results, but got ${results.length} a"
            )
          }
        case 1 =>
          if (${ Expr(ResLen) } < results.length - 1) {
            throw new IllegalArgumentException(
              s"Expected ${${ Expr(ResLen) }} results, but got ${results.length} b"
            )
          }
        case _ =>
          val resultSegmentSizesSum = ${
            resultSegmentSizes(
              variadicOperands,
              ResLen
            )
          }.fold(0)(_ + _)
          if (resultSegmentSizesSum != results.length) then
            throw new Exception(
              s"Expected ${resultSegmentSizesSum} results, got ${results.length} c"
            )
      }

      if (regions.length != ${ Expr(RegLen) }) {
        throw new IllegalArgumentException(
          s"Expected ${${ Expr(RegLen) }} regions, but got ${regions.length}"
        )
      }

      if (properties.size != ${ Expr(PropLen) }) {
        throw new IllegalArgumentException(
          s"Expected ${${ Expr(PropLen) }} properties, but got ${properties.size}"
        )
      }
    }
  }

  // Combine all parameters in the correct order
  val args = opDef.allDefsWithIndex.map { (d: OpInputDef, index: Int) =>
    d match
      case OperandDef(name, tpe, variadicity) =>
        NamedArg(name, operandArgs(index).asTerm)
      case ResultDef(name, tpe, variadicity) =>
        NamedArg(name, resultArgs(index).asTerm)
      case RegionDef(name, variadicity) =>
        NamedArg(name, regionsArgs(index).asTerm)
      case SuccessorDef(name, variadicity) =>
        NamedArg(name, successorArgs(index).asTerm)
      case OpPropertyDef(name, tpe) =>
        NamedArg(name, propertyArgs(index).asTerm)
  }

  // creates a new instance of the ADT op
  val constructorExpr =
    Apply(
      Select(New(TypeTree.of[T]), TypeRepr.of[T].typeSymbol.primaryConstructor),
      List.from(args)
    )
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
