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

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||  ADT to Unverified conversion Macro  ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

import scala.quoted.*

/** Small helper to select a member of an expression.
  * @param obj
  *   The object to select the member from.
  * @param name
  *   The name of the member to select.
  */
def selectMember(obj: Expr[_], name: String)(using
    Quotes
): Expr[Any] = {
  import quotes.reflect.*

  Select.unique(obj.asTerm, name).asExpr
}

//TODO: handle multi-variadic segmentSizes creation!
/** Get all constructs of the specified type flattened from the ADT expression.
  * @tparam Def
  *   The construct definition type.
  * @param opInputDefs
  *   The construct definitions.
  * @param adtOpExpr
  *   The ADT expression.
  */
def ADTFlatInputMacro[Def <: OpInputDef: Type](
    opInputDefs: Seq[Def],
    adtOpExpr: Expr[_]
)(using Quotes): Expr[ListType[DefinedInput[Def]]] = {
  import quotes.reflect.*
  val stuff = Expr.ofList(
    opInputDefs.map((d: Def) =>
      getConstructVariadicity(d) match
        case Variadicity.Variadic =>
          selectMember(adtOpExpr, d.name).asExprOf[Seq[DefinedInput[Def]]]
        case Variadicity.Single =>
          '{
            Seq(${
              selectMember(adtOpExpr, d.name)
                .asExprOf[DefinedInput[Def]]
            })
          }
    )
  )
  '{ ListType.from(${ stuff }.flatten) }
}

/** Create an UnverifiedOp instance from an ADT expression.
  * @tparam T
  *   The ADT type.
  * @param opDef
  *   The OperationDef derived from the ADT.
  * @param adtOpExpr
  *   The ADT expression.
  */
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
  val propertyExprs = ADTFlatInputMacro(opDef.properties, adtOpExpr)

  // Populating a Dictionarty with the properties
  val propertyNames = Expr.ofList(opDef.properties.map((d) => Expr(d.name)))
  val propertiesDict = '{
    DictType.from(${ propertyNames } zip ${ propertyExprs }.map(_.typ))
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
      dictionaryProperties = $propertiesDict,
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
/** Helper to verify a property argument.
  */
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

/** Type helper to get the defined input type of a construct definition.
  */
type DefinedInputOf[T <: OpInputDef, A <: Attribute] = T match {
  case OperandDef    => Operand[A]
  case ResultDef     => Result[A]
  case RegionDef     => Region
  case SuccessorDef  => Successor
  case OpPropertyDef => Property[A]
}

/** Type helper to get the defined input type of a construct definition.
  */
type DefinedInput[T <: OpInputDef] = DefinedInputOf[T, Attribute]

/** Helper to access the right sequence of constructs from an UnverifiedOp,
  * given a construct definition type.
  */
def getConstructSeq[Def <: OpInputDef: Type](
    op: Expr[UnverifiedOp[_]]
)(using Quotes) =
  Type.of[Def] match
    case '[ResultDef]     => '{ ${ op }.results }
    case '[OperandDef]    => '{ ${ op }.operands }
    case '[RegionDef]     => '{ ${ op }.regions }
    case '[SuccessorDef]  => '{ ${ op }.successors }
    case '[OpPropertyDef] => '{ ${ op }.dictionaryProperties.toSeq }

/** Helper to get the name of a construct definition type.
  */
def getConstructName[Def <: OpInputDef: Type](using Quotes) =
  Type.of[Def] match
    case '[ResultDef]     => "result"
    case '[OperandDef]    => "operand"
    case '[RegionDef]     => "region"
    case '[SuccessorDef]  => "successor"
    case '[OpPropertyDef] => "property"

/** Helper to get the expected type of a construct definition's construct.
  */
def getConstructConstraint(_def: OpInputDef)(using Quotes) =
  _def match
    case OperandDef(name, tpe, variadicity) => tpe
    case ResultDef(name, tpe, variadicity)  => tpe
    case RegionDef(name, variadicity)       => Type.of[Attribute]
    case SuccessorDef(name, variadicity)    => Type.of[Attribute]
    case OpPropertyDef(name, tpe)           => tpe

/** Helper to get the variadicity of a construct definition's construct.
  */
def getConstructVariadicity(_def: OpInputDef)(using Quotes) =
  _def match
    case OperandDef(name, tpe, variadicity) => variadicity
    case ResultDef(name, tpe, variadicity)  => variadicity
    case RegionDef(name, variadicity)       => variadicity
    case SuccessorDef(name, variadicity)    => variadicity
    case OpPropertyDef(name, tpe)           => Variadicity.Single

/*__________________*\
\*-- VERIFICATION --*/

/** Expect a segmentSizes property of DenseArrayAttr type, and return it as a
  * list of integers.
  *
  * @tparam Def
  *   The construct definition type.
  * @param op
  *   The UnverifiedOp expression.
  */
def expectSegmentSizes[Def <: OpInputDef: Type](
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

/** Partion constructs of a specified type. That is, verify that they are in a
  * coherent quantity, and partition them into the provided definitions.
  *
  * @tparam Def
  *   The construct definition type.
  * @param defs
  *   The construct definitions.
  * @param op
  *   The UnverifiedOp expression.
  */
def partitionedConstructs[Def <: OpInputDef: Type](
    defs: Seq[Def],
    op: Expr[UnverifiedOp[_]]
)(using Quotes) =
  // Get the flat list of constructs from the UnverifiedOp
  val flat = getConstructSeq[Def](op)
  // Check the number of variadic constructs
  defs.count(getConstructVariadicity(_) != Variadicity.Single) match
    // If there is no variadic defintion, partionning is just about taking individual elements
    case 0 => defs.zipWithIndex.map((d, i) => '{ ${ flat }(${ Expr(i) }) })
    // If there is a single variadic definition, partionning is about stripping the preceeding and following single elements, and taking the rest as elements of the variadic construct
    case 1 =>
      val preceeding =
        defs.indexWhere(getConstructVariadicity(_) == Variadicity.Variadic)
      val following = defs.length - preceeding - 1
      val preceeding_exprs = defs
        .slice(0, preceeding)
        .zipWithIndex
        .map((d, i) => '{ ${ flat }.apply(${ Expr(i) }) })

      val variadic_expr = '{
        ${ flat }
          .slice(${ Expr(preceeding) }, ${ flat }.length - ${ Expr(following) })
      }

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

      (preceeding_exprs :+ variadic_expr) ++ following_exprs
    // If there are multiple variadic definitions, a corresponding segmentSizes property is expected to disambiguate the partition.
    case _ =>
      // Expect a coherent segmentSizes and interpret it as a list of integers.
      val segmentSizes = '{
        val sizes = ${ expectSegmentSizes[Def](op) }
        val segments = sizes.length
        val total = sizes.sum
        // Check the segmentSizes define a segment for each definition
        if (segments != ${ Expr(defs.length) }) then
          throw new Exception(
            s"Expected ${${ Expr(defs.length) }} entries in ${${
                Expr(getConstructName[Def])
              }}SegmentSizes, got ${segments}."
          )
        // Check the segmentSizes' sum is coherent with the number of constructs
        if (total != ${ flat }.length) then
          throw new Exception(
            s"${${ Expr(getConstructName[Def]) }}'s sum does not match the op's ${${
                flat
              }.length} ${${ Expr(getConstructName[Def]) }}s."
          )
        sizes
      }
      // Partition the constructs according to the segmentSizes
      defs.zipWithIndex.map { case (d, i) =>
        getConstructVariadicity(d) match
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

/** Get all verified constructs of a specified type from an UnverifiedOp. That
  * is, of the expected types and variadicities, as specified by the
  * OperationDef.
  *
  * @tparam Def
  *   The construct definition type.
  * @param defs
  *   The constructs definitions.
  * @param op
  *   The UnverifiedOp expression.
  */
def verifiedConstructs[Def <: OpInputDef: Type](
    defs: Seq[Def],
    op: Expr[UnverifiedOp[_]]
)(using Quotes) = {
  // For each partitioned construct
  (partitionedConstructs(defs, op) zip defs).map { (c, d) =>
    // Get the expected type and variadicity of the construct
    val tpe = getConstructConstraint(d)
    val variadicity = getConstructVariadicity(d)
    tpe match
      case '[t] =>
        variadicity match
          // If the construct is not variadic, just check if it is of the expected type
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
          // If the construct is variadic, check if it is a list of the expected type
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

/** Return all named arguments for the primary constructor of an ADT. Those are
  * verified, in the sense that they are checked to be of the correct types and
  * numbers.
  *
  * @param opDef
  *   The OperationDef derived from the ADT.
  * @param op
  *   The UnverifiedOp instance.
  * @return
  *   The verified named arguments for the primary constructor of the ADT.
  */

def constructorArgs(
    opDef: OperationDef,
    op: Expr[UnverifiedOp[_]]
)(using Quotes) =
  import quotes.reflect._
  (verifiedConstructs(opDef.operands, op) zip opDef.operands).map((e, d) =>
    NamedArg(d.name, e.asTerm)
  ) ++
    (verifiedConstructs(opDef.results, op) zip opDef.results).map((e, d) =>
      NamedArg(d.name, e.asTerm)
    ) ++
    (verifiedConstructs(opDef.regions, op) zip opDef.regions).map((e, d) =>
      NamedArg(d.name, e.asTerm)
    ) ++
    (verifiedConstructs(opDef.successors, op) zip opDef.successors).map(
      (e, d) => NamedArg(d.name, e.asTerm)
    ) ++
    opDef.properties.map { case OpPropertyDef(name, tpe) =>
      tpe match
        case '[t] if TypeRepr.of[t] <:< TypeRepr.of[Attribute] =>
          val property = generateCheckedPropertyArgument[t & Attribute](
            '{ $op.dictionaryProperties },
            name
          )
          NamedArg(name, property.asTerm)
    }

  /** Attempt to create an ADT from an UnverifiedO[ADT]
    *
    * @tparam T
    *   The ADT Type.
    * @param opDef
    *   The OperationDef derived from the ADT.
    * @param genExpr
    *   The expression of the UnverifiedOp[ADT].
    * @return
    *   The ADT instance.
    * @raises
    *   Exception if the UnverifiedOp[ADT] is not valid to represent by the ADT.
    */

def fromUnverifiedOperationMacro[T: Type](
    opDef: OperationDef,
    genExpr: Expr[UnverifiedOp[T]]
)(using Quotes): Expr[T] =
  import quotes.reflect.*

  // Create named arguments for all of the ADT's constructor arguments.
  val args = constructorArgs(opDef, genExpr)

  // Return a call to the primary constructor of the ADT.
  Apply(
    Select(New(TypeTree.of[T]), TypeRepr.of[T].typeSymbol.primaryConstructor),
    List.from(args)
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
