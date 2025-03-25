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
extension [A: Type, B: Type](es: Expr[Iterable[A]])

  def map(f: Expr[A] => Expr[B])(using Quotes): Expr[Iterable[B]] = {
    '{
      $es.map(a => ${ f('a) })
    }
  }

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||  ADT to Unverified conversion Macro  ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

import scala.quoted.*

def selectMember[T: Type](obj: Expr[T], name: String)(using
    Quotes
): Expr[Any] = {
  import quotes.reflect.*

  Select.unique(obj.asTerm, name).asExpr
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
          selectMember(adtOpExpr, d.name)
        case _ =>
          '{
            Seq(${
              selectMember(adtOpExpr, d.name)
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

/*_____________*\
\*-- HELPERS --*/

def generateCheckedPropertyArgument[A <: Attribute: Type](
    list: Expr[DictType[String, Attribute]],
    propName: String
)(using Quotes): Expr[Property[A]] =
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

def getConstructSeq[Def <: MayVariadicOpInputDef: Type](
    op: Expr[UnverifiedOp[_]]
)(using Quotes) =
  Type.of[DefinedInput[Def]] match
    case '[Result[_]]  => '{ ${ op }.results }
    case '[Operand[_]] => '{ ${ op }.operands }
    case '[Region]     => '{ ${ op }.regions }
    case '[Successor]  => '{ ${ op }.successors }

def getConstructName[Def <: MayVariadicOpInputDef: Type](using Quotes) =
  Type.of[DefinedInput[Def]] match
    case '[Result[_]]  => "result"
    case '[Operand[_]] => "operand"
    case '[Region]     => "region"
    case '[Successor]  => "successor"

def expectSegmentSizes[Def <: MayVariadicOpInputDef: Type](
    op: Expr[UnverifiedOp[_]]
)(using Quotes) =
  val segmentSizesName = s"${getConstructName[Def]}SegmentSizes"
  '{
    val dense =
      ${ op }.dictionaryProperties.get(s"${${ Expr(segmentSizesName) }}") match
        case Some(segmentSizes) =>
          segmentSizes match
            case dense: DenseArrayAttr => dense
            case _ =>
              throw new Exception(
                s"Expected ${${ Expr(segmentSizesName) }} to be a DenseArrayAttr"
              )

        case None =>
          throw new Exception(
            s"Expected ${${ Expr(segmentSizesName) }} property"
          )

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
    ).verify(dense, ConstraintContext())

    for (s <- dense) yield s match {
      case right: IntegerAttr => right.value.data.toInt
      case _ =>
        throw new Exception(
          "Unreachable exception as per above constraint check."
        )
    }
  }

def partitionConstruct[Def <: MayVariadicOpInputDef: Type](
    defs: Seq[Def],
    op: Expr[UnverifiedOp[_]]
)(using Quotes) =
  val flat = getConstructSeq[Def](op)
  defs.count(_.variadicity != Variadicity.Single) match
    case 0 => defs.zipWithIndex.map((d, i) => '{ ${ flat }(${ Expr(i) }) })

    case 1 =>
      val preceeding = defs.indexWhere(_.variadicity == Variadicity.Variadic)
      val following = defs.length - preceeding - 1
      val preceeding_exprs = defs
        .slice(0, preceeding)
        .zipWithIndex
        .map((d, i) => '{ ${ flat }.apply(${ Expr(i) }) })
      println(preceeding_exprs.map(_.show).mkString("\n"))
      val variadic_expr = '{
        ${ flat }
          .slice(${ Expr(preceeding) }, ${ flat }.length - ${ Expr(following) })
      }
      println(variadic_expr.show)
      val following_exprs = defs
        .slice(preceeding + 1, defs.length)
        .zipWithIndex
        .map((d, i) =>
          '{
            ${ flat }.apply(${ flat }.length - ${ Expr(following) } + ${
              Expr(i)
            })
          }
        )
      println(following_exprs.map(_.show).mkString("\n"))
      (preceeding_exprs :+ variadic_expr) ++ following_exprs
    case _ =>
      val segmentSizes = '{
        val sizes = ${ expectSegmentSizes[Def](op) }
        val segments = sizes.length
        val total = sizes.sum
        if (segments != ${ Expr(defs.length) }) then
          throw new Exception(
            s"Expected ${${ Expr(defs.length) }} entries in ${${
                Expr(getConstructName[Def])
              }}SegmentSizes, got ${segments}."
          )
        if (total != ${ flat }.length) then
          throw new Exception(
            s"${${ Expr(getConstructName[Def]) }}'s sum does not match the op's ${${
                flat
              }.length} ${${ Expr(getConstructName[Def]) }}s."
          )
        sizes
      }
      defs.zipWithIndex.map { case (d, i) =>
        d.variadicity match
          case Variadicity.Single => '{ ${ flat }(${ Expr(i) }) }
          case Variadicity.Variadic =>
            val e = '{
              val sizes = $segmentSizes
              val start = sizes.slice(0, ${ Expr(i) }).sum
              val end = start + sizes(${ Expr(i) })
              ${ flat }.slice(start, end)
            }
            println(e.show)
            e
      }

def verifyConstructs[Def <: MayVariadicOpInputDef: Type](
    defs: Seq[Def],
    op: Expr[UnverifiedOp[_]]
)(using Quotes) = {
  (partitionConstruct[Def](defs, op) zip defs).map { (c, d) =>
    val tpe = d match
      case OperandDef(name, tpe, variadicity) => tpe
      case ResultDef(name, tpe, variadicity)  => tpe
      case RegionDef(name, variadicity)       => Type.of[Attribute]
      case SuccessorDef(name, variadicity)    => Type.of[Attribute]
    tpe match
      case '[t] =>
        d.variadicity match
          case Variadicity.Single =>
            '{
              if (!${ c }.isInstanceOf[DefinedInputOf[Def, t & Attribute]]) then
                throw new Exception(
                  s"Expected ${${ Expr(d.name) }} to be of type ${${
                      Expr(Type.show[DefinedInputOf[Def, t & Attribute]])
                    }}, got ${${ c }}"
                )
              ${ c }.asInstanceOf[DefinedInputOf[Def, t & Attribute]]
            }
          case Variadicity.Variadic =>
            '{
              if (
                !${ c }
                  .isInstanceOf[ListType[DefinedInputOf[Def, t & Attribute]]]
              ) then
                throw new Exception(
                  s"Expected ${${ Expr(d.name) }} to be of type ${${
                      Expr(Type.show[ListType[DefinedInputOf[Def, t & Attribute]]])
                    }}, got ${${ c }}"
                )
              ${ c }
                .asInstanceOf[ListType[DefinedInputOf[Def, t & Attribute]]]
                .toSeq
            }
  }
}

def constructorArgs[Def <: MayVariadicOpInputDef: Type](
    opDef: OperationDef,
    op: Expr[UnverifiedOp[_]]
)(using Quotes) =
  import quotes.reflect._
  (verifyConstructs(opDef.operands, op) zip opDef.operands).map((e, d) =>
    NamedArg(d.name, e.asTerm)
  ) ++
    (verifyConstructs(opDef.results, op) zip opDef.results).map((e, d) =>
      NamedArg(d.name, e.asTerm)
    ) ++
    (verifyConstructs(opDef.regions, op) zip opDef.regions).map((e, d) =>
      NamedArg(d.name, e.asTerm)
    ) ++
    (verifyConstructs(opDef.successors, op) zip opDef.successors).map((e, d) =>
      NamedArg(d.name, e.asTerm)
    )

def fromUnverifiedOperationMacro[T: Type](
    opDef: OperationDef,
    genExpr: Expr[UnverifiedOp[T]]
)(using Quotes): Expr[T] =
  import quotes.reflect.*

  val args = constructorArgs(opDef, genExpr)

  /*________________*\
  \*-- PROPERTIES --*/

  val properties = opDef.properties

  // extracting and validating each input, the re-creating the Input instance
  val propertyArgs = properties.map { case OpPropertyDef(name, tpe) =>
    tpe match
      case '[t] if TypeRepr.of[t] <:< TypeRepr.of[Attribute] =>
        val property = generateCheckedPropertyArgument[t & Attribute](
          '{ $genExpr.dictionaryProperties },
          name
        )
        NamedArg(name, property.asTerm)
  }

  // println((args ++ propertyArgs).map(_.asExpr.show).mkString("\n"))

  // creates a new instance of the ADT op
  Apply(
    Select(New(TypeTree.of[T]), TypeRepr.of[T].typeSymbol.primaryConstructor),
    List.from(args ++ propertyArgs)
  )
    .asExprOf[T]

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
