package scair.clair.macros

import fastparse.*
import scair.AttrParser
import scair.Parser.*
import scair.clair.codegen.*
import scair.clair.mirrored.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.scairdl.constraints.*

import scala.quoted.*

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

/** Small helper to select a member of an expression.
  * @param obj
  *   The object to select the member from.
  * @param name
  *   The name of the member to select.
  */
def selectMember(obj: Expr[?], name: String)(using
    Quotes
): Expr[Any] = {
  import quotes.reflect.*

  Select.unique(obj.asTerm, name).asExpr
}

def makeSegmentSizes[T <: MayVariadicOpInputDef: Type](
    hasMultiVariadic: Boolean,
    defs: Seq[T],
    adtOpExpr: Expr[?]
)(using Quotes): Expr[Map[String, Attribute]] = {
  val name = Expr(s"${getConstructName[T]}SegmentSizes")
  hasMultiVariadic match {
    case true =>
      val arrayAttr: Expr[Seq[Int]] =
        Expr.ofList(
          defs.map((d) =>
            d.variadicity match {
              case Variadicity.Single                          => Expr(1)
              case Variadicity.Variadic | Variadicity.Optional =>
                '{
                  ${ selectMember(adtOpExpr, d.name).asExprOf[Seq[?]] }.length
                }
            }
          )
        )
      '{
        Map(
          $name -> DenseArrayAttr(
            IntegerType(IntData(32), Signless),
            ${ arrayAttr }.map(x =>
              IntegerAttr(
                IntData(x),
                IntegerType(IntData(32), Signless)
              )
            )
          )
        )
      }
    case false => '{ Map.empty[String, Attribute] }
  }
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
    adtOpExpr: Expr[?]
)(using Quotes): Expr[Seq[DefinedInput[Def]]] = {
  val stuff = Expr.ofList(
    opInputDefs.map((d: Def) =>
      getConstructVariadicity(d) match
        case Variadicity.Optional =>
          selectMember(adtOpExpr, d.name).asExprOf[Option[DefinedInput[Def]]]
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
  '{ ${ stuff }.flatten }
}

def operandsMacro(
    opDef: OperationDef,
    adtOpExpr: Expr[?]
)(using Quotes): Expr[Seq[Operand[Attribute]]] =
  ADTFlatInputMacro(opDef.operands, adtOpExpr)

def successorsMacro(
    opDef: OperationDef,
    adtOpExpr: Expr[?]
)(using Quotes): Expr[Seq[Successor]] =
  ADTFlatInputMacro(opDef.successors, adtOpExpr)

def resultsMacro(
    opDef: OperationDef,
    adtOpExpr: Expr[?]
)(using Quotes): Expr[Seq[Result[Attribute]]] =
  ADTFlatInputMacro(opDef.results, adtOpExpr)

def regionsMacro(
    opDef: OperationDef,
    adtOpExpr: Expr[?]
)(using Quotes): Expr[Seq[Region]] =
  ADTFlatInputMacro(opDef.regions, adtOpExpr)

def propertiesMacro(
    opDef: OperationDef,
    adtOpExpr: Expr[?]
)(using Quotes): Expr[Map[String, Attribute]] =
  // extracting property instances from the ADT
  val propertyExprs = ADTFlatInputMacro(opDef.properties, adtOpExpr)

  val opSegSizeProp = makeSegmentSizes(
    opDef.hasMultiVariadicOperands,
    opDef.operands,
    adtOpExpr
  )
  val resSegSizeProp = makeSegmentSizes(
    opDef.hasMultiVariadicResults,
    opDef.results,
    adtOpExpr
  )
  val regSegSizeProp = makeSegmentSizes(
    opDef.hasMultiVariadicRegions,
    opDef.regions,
    adtOpExpr
  )
  val succSegSizeProp = makeSegmentSizes(
    opDef.hasMultiVariadicSuccessors,
    opDef.successors,
    adtOpExpr
  )
  // Populating a Dictionarty with the properties
  val propertyNames = Expr.ofList(opDef.properties.map((d) => Expr(d.name)))
  '{
    Map.from(${ propertyNames } zip ${ propertyExprs })
      ++ ${ opSegSizeProp }
      ++ ${ resSegSizeProp }
      ++ ${ regSegSizeProp }
      ++ ${ succSegSizeProp }
  }

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||  Unverified to ADT conversion Macro  ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

/*_____________*\
\*-- HELPERS --*/
/** Helper to verify a property argument.
  */
def generateCheckedPropertyArgument[A <: Attribute: Type](
    list: Expr[Map[String, Attribute]],
    propName: String
)(using Quotes): Expr[A] =
  val typeName = Type.of[A].toString()
  '{
    val value: Option[Attribute] = $list.get(${ Expr(propName) })

    if (value.isEmpty) {
      throw new IllegalArgumentException(
        s"Missing required property \"${${ Expr(propName) }}\" of type ${${
            Expr(typeName)
          }}"
      )
    } else if (!value.get.isInstanceOf[A]) {
      throw new IllegalArgumentException(
        s"Type mismatch for property \"${${ Expr(propName) }}\": " +
          s"expected ${${ Expr(typeName) }}, " +
          s"but found ${value.getClass.getSimpleName}"
      )
    }
    value.get.asInstanceOf[A]
  }

def generateOptionalCheckedPropertyArgument[A <: Attribute: Type](
    list: Expr[Map[String, Attribute]],
    propName: String
)(using Quotes): Expr[Option[A]] =
  val typeName = Type.of[A].toString()
  '{
    val value: Option[Attribute] = $list.get(${ Expr(propName) })

    if (value.isEmpty) {
      value.asInstanceOf[None.type]
    } else if (value.get.isInstanceOf[A]) {
      value.asInstanceOf[Option[A]]
    } else
      throw new IllegalArgumentException(
        s"Type mismatch for property \"${${ Expr(propName) }}\": " +
          s"expected ${${ Expr(typeName) }}, " +
          s"but found ${value.getClass}"
      )
  }

/** Type helper to get the defined input type of a construct definition.
  */
type DefinedInputOf[T <: OpInputDef, A <: Attribute] = T match {
  case OperandDef    => Operand[A]
  case ResultDef     => Result[A]
  case RegionDef     => Region
  case SuccessorDef  => Successor
  case OpPropertyDef => A
}

/** Type helper to get the defined input type of a construct definition.
  */
type DefinedInput[T <: OpInputDef] = DefinedInputOf[T, Attribute]

/** Helper to access the right sequence of constructs from an UnverifiedOp,
  * given a construct definition type.
  */
def getConstructSeq[Def <: OpInputDef: Type as d](
    op: Expr[DerivedOperationCompanion[?]#UnverifiedOp]
)(using Quotes) =
  (d match
    case '[ResultDef]     => '{ ${ op }.results }
    case '[OperandDef]    => '{ ${ op }.operands }
    case '[RegionDef]     => '{ ${ op }.regions.map(_.detached) }
    case '[SuccessorDef]  => '{ ${ op }.successors }
    case '[OpPropertyDef] => '{ ${ op }.properties.toSeq }
  ).asExprOf[Seq[DefinedInput[Def]]]

/** Helper to get the name of a construct definition type.
  */
def getConstructName[Def <: OpInputDef: Type as d](using Quotes) =
  d match
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
    case OpPropertyDef(name, tpe, _)        => tpe

/** Helper to get the variadicity of a construct definition's construct.
  */
def getConstructVariadicity(_def: OpInputDef)(using Quotes) =
  _def match
    case OperandDef(name, tpe, variadicity) => variadicity
    case ResultDef(name, tpe, variadicity)  => variadicity
    case RegionDef(name, variadicity)       => variadicity
    case SuccessorDef(name, variadicity)    => variadicity
    case OpPropertyDef(name, tpe, false)    => Variadicity.Single
    case OpPropertyDef(name, tpe, _)        => Variadicity.Optional

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
    op: Expr[DerivedOperationCompanion[?]#UnverifiedOp]
)(using Quotes) =
  val segmentSizesName = s"${getConstructName[Def]}SegmentSizes"
  '{
    val dense =
      ${ op }.properties.get(s"${${ Expr(segmentSizesName) }}") match
        case Some(segmentSizes) =>
          segmentSizes match
            case dense: DenseArrayAttr => dense
            case _                     =>
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
      case _                  =>
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
def partitionConstructs[Def <: OpInputDef: Type](
    defs: Seq[Def],
    op: Expr[DerivedOperationCompanion[?]#UnverifiedOp],
    flat: Expr[Seq[DefinedInput[Def]]]
)(using Quotes) =
  // Check the number of variadic constructs
  defs.count(getConstructVariadicity(_) != Variadicity.Single) match
    // If there is no variadic defintion, partionning is just about taking individual elements
    case 0 =>
      defs.zipWithIndex.map((d, i) =>
        val defLength = Expr(defs.length)
        '{
          if ($flat.length != $defLength) then
            throw new Exception(
              s"Expected ${${ Expr(defs.length) }} ${${
                  Expr(getConstructName[Def])
                }}s, got ${$flat.length}."
            )
          ${ flat }(${ Expr(i) })
        }
      )
    // If there is a single variadic definition, partionning is about stripping the preceeding and following single elements, and taking the rest as elements of the variadic construct
    case 1 =>
      val preceeding =
        defs.indexWhere(x =>
          val a = getConstructVariadicity(x)
          a == Variadicity.Variadic ||
          a == Variadicity.Optional
        )
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
          case Variadicity.Variadic | Variadicity.Optional =>
            '{
              val sizes = $segmentSizes
              val start = sizes.slice(0, ${ Expr(i) }).sum
              val end = start + sizes(${ Expr(i) })
              ${ flat }.slice(start, end)
            }
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
def verifiyConstructs[Def <: OpInputDef: Type](
    defs: Seq[Def],
    op: Expr[DerivedOperationCompanion[?]#UnverifiedOp],
    partitioned: Seq[Expr[DefinedInput[Def] | Seq[DefinedInput[Def]]]]
)(using Quotes) = {
  (partitioned zip defs).map { (c, d) =>
    // Get the expected type and variadicity of the construct
    val tpe = getConstructConstraint(d)
    val variadicity = getConstructVariadicity(d)
    tpe match
      case '[type t <: Attribute; `t`] =>
        variadicity match
          case Variadicity.Optional =>
            // If the construct is optional, check if it is defined and if so, verify its type
            '{
              if (!${ c }.isInstanceOf[Seq[DefinedInputOf[Def, t]]])
              then
                throw new Exception(
                  s"Expected ${${ Expr(d.name) }} to be of type ${${
                      Expr(Type.show[DefinedInputOf[Def, t]])
                    }}, got ${${ c }}"
                )
              ${ c }
                .asInstanceOf[Seq[DefinedInputOf[Def, t]]]
                .headOption
            }
          // If the construct is not variadic, just check if it is of the expected type
          case Variadicity.Single =>
            '{
              if (!${ c }.isInstanceOf[DefinedInputOf[Def, t]]) then
                throw new Exception(
                  s"Expected ${${ Expr(d.name) }} to be of type ${${
                      Expr(Type.show[DefinedInputOf[Def, t]])
                    }}, got ${${ c }}"
                )

              ${ c }.asInstanceOf[DefinedInputOf[Def, t]]
            }
          // If the construct is variadic, check if it is a list of the expected type
          case Variadicity.Variadic =>
            '{
              if (
                  !${ c }
                    .isInstanceOf[Seq[DefinedInputOf[Def, t]]]
                )
              then
                throw new Exception(
                  s"Expected ${${ Expr(d.name) }} to be of type ${${
                      Expr(Type.show[Seq[DefinedInputOf[Def, t]]])
                    }}, got ${${ c }}"
                )
              ${ c }
                .asInstanceOf[Seq[DefinedInputOf[Def, t]]]
                .toSeq
            }
  }
}

def verifiedConstructs[Def <: OpInputDef: Type](
    defs: Seq[Def],
    op: Expr[DerivedOperationCompanion[?]#UnverifiedOp]
)(using Quotes) = {
  // Get the flat sequence of these constructs
  val flat = getConstructSeq(op)
  // partition the constructs according to their definitions
  val partitioned = partitionConstructs(
    defs,
    op,
    flat
  )

  // Verify the constructs
  verifiyConstructs(defs, op, partitioned)
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
    op: Expr[DerivedOperationCompanion[?]#UnverifiedOp]
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
    opDef.properties.map { case OpPropertyDef(name, tpe, optionality) =>
      tpe match
        case '[type t <: Attribute; `t`] =>
          val property =
            if optionality then
              generateOptionalCheckedPropertyArgument[t](
                '{ $op.properties },
                name
              )
            else
              generateCheckedPropertyArgument[t](
                '{ $op.properties },
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
    genExpr: Expr[DerivedOperationCompanion[T]#UnverifiedOp]
)(using Quotes): Expr[T] =
  import quotes.reflect.*

  // Create named arguments for all of the ADT's constructor arguments.
  val args = constructorArgs(opDef, genExpr)

  // Return a call to the primary constructor of the ADT.
  Apply(
    Select(New(TypeTree.of[T]), TypeRepr.of[T].typeSymbol.primaryConstructor),
    List.from(args)
  ).asExprOf[T]

def getAttrConstructor[T: Type](
    attrDef: AttributeDef,
    attributes: Expr[Seq[Attribute]]
)(using
    Quotes
): Expr[T] = {
  import quotes.reflect.*

  val lengthCheck = Type.of[T] match
    case '[type t <: Attribute; `t`] =>
      '{
          if ${ Expr(attrDef.attributes.length) } != $attributes.length then
            throw new Exception(
              s"Number of attributes ${${ Expr(attrDef.attributes.length) }} does not match the number of provided attributes ${$attributes.length}"
            )
        }
    case _ =>
      report.errorAndAbort(
        s"Type ${Type.show[T]} needs to be a subtype of Attribute"
      )

  val defs = attrDef.attributes

  val verifiedConstructs = (defs.zipWithIndex.map((d, i) =>
    '{ ${ attributes }(${ Expr(i) }) }
  ) zip defs)
    .map { (a, d) =>
      // expected type of the attribute
      val tpe = d.tpe
      tpe match
        case '[t] =>
          '{
            if (!${ a }.isInstanceOf[t]) then
              throw Exception(
                s"Expected ${${ Expr(d.name) }} to be of type ${${
                    Expr(Type.show[t])
                  }}, got ${${ a }}"
              )
            ${ a }.asInstanceOf[t]
          }
    }

  val args = (verifiedConstructs zip attrDef.attributes)
    .map((e, d) => NamedArg(d.name, e.asTerm))

  val constructorCall = Apply(
    Select(New(TypeTree.of[T]), TypeRepr.of[T].typeSymbol.primaryConstructor),
    List.from(args)
  ).asExprOf[T]

  '{
    $lengthCheck
    $constructorCall
  }
}

def ADTFlatAttrInputMacro[Def <: AttributeDef: Type](
    attrInputDefs: Seq[AttributeParamDef],
    adtAttrExpr: Expr[?]
)(using Quotes): Expr[Seq[Attribute]] = {
  Expr.ofList(
    attrInputDefs.map(d =>
      selectMember(adtAttrExpr, d.name).asExprOf[Attribute]
    )
  )
}

def parametersMacro(
    attrDef: AttributeDef,
    adtAttrExpr: Expr[?]
)(using Quotes): Expr[Seq[Attribute]] =
  ADTFlatAttrInputMacro(attrDef.attributes, adtAttrExpr)

def derivedAttributeCompanion[T: Type](using
    Quotes
): Expr[DerivedAttributeCompanion[T]] =

  val attrDef = getAttrDefImpl[T]

  '{
    new DerivedAttributeCompanion[T] {
      override def name: String = ${ Expr(attrDef.name) }
      override def parse[$: P](p: AttrParser): P[T] = P(
        ("<" ~/ p.Type.rep(sep = ",") ~ ">")
      ).orElse(Seq())
        .map(x => ${ getAttrConstructor[T](attrDef, '{ x }) })
      def parameters(attr: T): Seq[Attribute | Seq[Attribute]] = ${
        parametersMacro(attrDef, '{ attr })
      }
    }
  }

def deriveOperationCompanion[T <: Operation: Type](using
    Quotes
): Expr[DerivedOperationCompanion[T]] =
  val opDef = getDefImpl[T]

  '{

    new DerivedOperationCompanion[T]:

      def operands(adtOp: T): Seq[Value[Attribute]] =
        ${ operandsMacro(opDef, '{ adtOp }) }
      def successors(adtOp: T): Seq[Block] =
        ${ successorsMacro(opDef, '{ adtOp }) }
      def results(adtOp: T): Seq[Result[Attribute]] =
        ${ resultsMacro(opDef, '{ adtOp }) }
      def regions(adtOp: T): Seq[Region] =
        ${ regionsMacro(opDef, '{ adtOp }) }
      def properties(adtOp: T): Map[String, Attribute] =
        ${ propertiesMacro(opDef, '{ adtOp }) }

      def name: String = ${ Expr(opDef.name) }

      def apply(
          operands: Seq[Value[Attribute]] = Seq(),
          successors: Seq[Block] = Seq(),
          results: Seq[Result[Attribute]] = Seq(),
          regions: Seq[Region] = Seq(),
          properties: Map[String, Attribute] = Map.empty[String, Attribute],
          attributes: DictType[String, Attribute] =
            DictType.empty[String, Attribute]
      ): UnverifiedOp = UnverifiedOp(
        operands = operands,
        successors = successors,
        results = results,
        regions = regions,
        properties = properties,
        attributes = attributes
      )

      def unverify(adtOp: T): UnverifiedOp =
        UnverifiedOp(
          operands = operands(adtOp),
          successors = successors(adtOp),
          results = results(adtOp),
          regions = regions(adtOp).map(_.detached),
          properties = properties(adtOp),
          attributes = adtOp.attributes
        )

      def verify(unverOp: UnverifiedOp): T =
        ${
          fromUnverifiedOperationMacro[T](opDef, '{ unverOp })
        } match {
          case adt: DerivedOperation[_, T] =>
            adt.attributes.addAll(unverOp.attributes)
            adt
          case _ =>
            throw new Exception(
              s"Internal Error: Hacky did not hack -> T is not a DerivedOperation: ${unverOp}"
            )
        }

  }
