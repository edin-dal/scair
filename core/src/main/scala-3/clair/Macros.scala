package scair.clair.macros

import fastparse.*
import scair.AttrParser
import scair.AttrParser.whitespace
import scair.Parser
import scair.Parser.*
import scair.Printer
import scair.clair.codegen.*
import scair.clair.mirrored.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.macros.*
import scair.transformations.CanonicalizationPatterns
import scair.transformations.RewritePattern

import scala.annotation.switch
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
||  ADT to Unstructured conversion Macro  ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

def makeSegmentSizes[T <: MayVariadicOpInputDef: Type](
    hasMultiVariadic: Boolean,
    defs: Seq[T],
    adtOpExpr: Expr[?]
)(using Quotes): Option[(String, Expr[Attribute])] =
  val name = s"${getConstructName[T]}SegmentSizes"
  hasMultiVariadic match
    case true =>
      val arrayAttr: Expr[Seq[Int]] =
        Expr.ofList(
          defs.map((d) =>
            d.variadicity match
              case Variadicity.Single                          => Expr(1)
              case Variadicity.Variadic | Variadicity.Optional =>
                '{
                  ${ adtOpExpr.member[Seq[?]](d.name) }.length
                }
          )
        )
      Some(
        name -> '{
          DenseArrayAttr(
            IntegerType(IntData(32), Signless),
            ${ arrayAttr }.map(x =>
              IntegerAttr(
                IntData(x),
                IntegerType(IntData(32), Signless)
              )
            )
          )
        }
      )
    case false => None

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
)(using Quotes): Expr[Seq[DefinedInput[Def]]] =
  val stuff =
    opInputDefs.map((d: Def) =>
      adtOpExpr.member[DefinedInput[Def] | IterableOnce[DefinedInput[Def]]](
        d.name
      )
    )
  stuff.foldLeft('{ Seq.empty[DefinedInput[Def]] })((seq, next) =>
    next match
      case '{ $ne: DefinedInput[Def] }               => '{ $seq :+ $ne }
      case '{ $ns: IterableOnce[DefinedInput[Def]] } => '{ $seq :++ $ns }
  )

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
  val definedProps =
    if opDef.properties.isEmpty then '{ Map.empty[String, Attribute] }
    else
      // extracting property instances from the ADT
      val propertyExprs = ADTFlatInputMacro(opDef.properties, adtOpExpr)
      val propertyNames = Expr.ofList(opDef.properties.map((d) => Expr(d.name)))
      '{
        Map.from(${ propertyNames } zip ${ propertyExprs })
      }

  Seq(opSegSizeProp, resSegSizeProp, regSegSizeProp, succSegSizeProp).foldLeft(
    definedProps
  ) {
    case (map, Some((name, segSize))) =>
      '{ $map + (${ Expr(name) } -> $segSize) }
    case (map, None) => map
  }

def customPrintMacro(
    opDef: OperationDef,
    adtOpExpr: Expr[?],
    p: Expr[Printer],
    indentLevel: Expr[Int]
)(using Quotes): Expr[Unit] =
  opDef.assembly_format match
    case Some(format) =>
      format.print(adtOpExpr, p)
    case None =>
      '{
        $p.printGenericMLIROperation(${ adtOpExpr }.asInstanceOf[Operation])(
          using $indentLevel
        )
      }

def parseMacro(
    opDef: OperationDef,
    p: Expr[Parser],
    resNames: Expr[Seq[String]]
)(using
    Quotes
): Expr[P[Any] ?=> P[Operation]] =
  opDef.assembly_format match
    case Some(format) =>
      format.parse(opDef, p, resNames)
    case None =>
      '{
        throw new Exception(
          s"No custom Parser implemented for Operation '${${
              Expr(opDef.name)
            }}'"
        )
      }

def verifyMacro(
    opDef: OperationDef,
    adtOpExpr: Expr[?]
)(using Quotes): Expr[Either[String, Operation]] =

  val a = opDef.operands // val xyz: Seq[Expr[Either[String, Unit]]] =
    .filter(_.variadicity == Variadicity.Single)
    .collect(_ match
      case OperandDef(name, _, _, Some(constraint)) =>
        val mem = adtOpExpr.member[Operand[Attribute]](name)
        '{ (ctx: scair.core.constraints.ConstraintContext) =>
          $constraint.verify($mem.typ)(using ctx)
        })

  '{
    given ctx: scair.core.constraints.ConstraintContext =
      scair.core.constraints.ConstraintContext()
    ${
      val chain = a.foldLeft[Expr[Either[String, Unit]]](
        '{ Right(()) }
      )((res, result) => '{ $res.flatMap(_ => $result(ctx)) })
      '{ $chain.map(_ => $adtOpExpr.asInstanceOf[Operation]) }
    }
  }

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
|| Unstructured to ADT conversion Macro ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

/*_____________*\
\*-- HELPERS --*/
/** Helper to check a property argument.
  */
def generateCheckedPropertyArgument[A <: Attribute: Type](
    list: Expr[Map[String, Attribute]],
    propName: String
)(using Quotes): Expr[A] =
  val typeName = Type.of[A].toString()
  '{
    val value: Option[Attribute] = $list.get(${ Expr(propName) })
    value match
      case None =>
        throw new IllegalArgumentException(
          s"Missing required property \"${${ Expr(propName) }}\" of type ${${
              Expr(typeName)
            }}"
        )
      case Some(prop: A) => prop
      case Some(_)       =>
        throw new IllegalArgumentException(
          s"Type mismatch for property \"${${ Expr(propName) }}\": " +
            s"expected ${${ Expr(typeName) }}, " +
            s"but found ${value.getClass.getSimpleName}"
        )
  }

def generateOptionalCheckedPropertyArgument[A <: Attribute: Type](
    list: Expr[Map[String, Attribute]],
    propName: String
)(using Quotes): Expr[Option[A]] =
  val typeName = Type.of[A].toString()
  '{
    val value: Option[Attribute] = $list.get(${ Expr(propName) })
    value.map {
      case prop: A => prop
      case _       =>
        throw new IllegalArgumentException(
          s"Type mismatch for property \"${${ Expr(propName) }}\": " +
            s"expected ${${ Expr(typeName) }}, " +
            s"but found ${value.getClass}"
        )
    }
  }

/** Type helper to get the defined input type of a construct definition.
  */
type DefinedInputOf[T <: OpInputDef, A <: Attribute] = T match
  case OperandDef    => Operand[A]
  case ResultDef     => Result[A]
  case RegionDef     => Region
  case SuccessorDef  => Successor
  case OpPropertyDef => A

/** Type helper to get the defined input type of a construct definition.
  */
type DefinedInput[T <: OpInputDef] = DefinedInputOf[T, Attribute]

/** Helper to access the right sequence of constructs from an UnstructuredOp,
  * given a construct definition type.
  */
def getConstructSeq[Def <: OpInputDef: Type as d](
    op: Expr[DerivedOperationCompanion[?]#UnstructuredOp]
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
    case OperandDef(name, tpe, variadicity, _) => tpe
    case ResultDef(name, tpe, variadicity, _)  => tpe
    case RegionDef(name, variadicity)          => Type.of[Attribute]
    case SuccessorDef(name, variadicity)       => Type.of[Attribute]
    case OpPropertyDef(name, tpe, _, _)        => tpe

/** Helper to get the variadicity of a construct definition's construct.
  */
def getConstructVariadicity(_def: OpInputDef)(using Quotes) =
  _def match
    case v: MayVariadicOpInputDef => v.variadicity

/*__________________*\
\*-- STRUCTURING  --*/

/** Expect a segmentSizes property of DenseArrayAttr type, and return it as a
  * list of integers.
  *
  * @tparam Def
  *   The construct definition type.
  * @param op
  *   The UnstructuredOp expression.
  */
def expectSegmentSizes[Def <: OpInputDef: Type](using Quotes) =
  val segmentSizesName = s"${getConstructName[Def]}SegmentSizes"
  '{ (properties: Map[String, Attribute]) =>
    val dense =
      properties.get(s"${${ Expr(segmentSizesName) }}") match
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

    if dense.typ != I32 then
      throw new Exception(
        s"Expected ${${ Expr(segmentSizesName) }} to be of element type i32"
      )

    dense.map {
      case IntegerAttr(IntData(value), eltpe) if eltpe == I32 => value.toInt
      case _                                                  =>
        throw new Exception(
          s"Expected ${${ Expr(segmentSizesName) }} to contain IntegerAttr of i32"
        )
    }
  }

/** Partition a construct sequence, in the case of no variadic defintion.
  *
  * @see
  *   [[constructPartitioner]]
  */
def uniadicConstructPartitioner[Def <: OpInputDef: Type](defs: Seq[Def])(using
    Quotes
) =
  defs.zipWithIndex.map((d, i) =>
    val defLength = Expr(defs.length)
    '{
      (
          properties: Map[String, Attribute],
          flat: Seq[DefinedInput[Def]]
      ) =>
        // TODO: This does not really belong here. Bigger fishes to fry at the time of
        // writing though. Conceptually this should end up in some kind of header.
        if flat.length != $defLength then
          throw new Exception(
            s"Expected ${${ Expr(defs.length) }} ${${
                Expr(getConstructName[Def])
              }}s, got ${flat.length}."
          )
        flat(${ Expr(i) })
    }
  )

/** Partition a construct sequence, in the case of a single variadic defintion.
  *
  * @see
  *   [[constructPartitioner]]
  */
def univariadicConstructPartitioner[Def <: OpInputDef: Type](defs: Seq[Def])(
    using Quotes
) =
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
    .map((d, i) =>
      '{

        (
            properties: Map[String, Attribute],
            flat: Seq[DefinedInput[Def]]
        ) =>
          flat.apply(${ Expr(i) })
      }
    )

  val variadic_expr = '{
    (
        properties: Map[String, Attribute],
        flat: Seq[DefinedInput[Def]]
    ) =>
      flat
        .slice(${ Expr(preceeding) }, flat.length - ${ Expr(following) })
  }

  val following_exprs = defs
    .slice(preceeding + 1, defs.length)
    .zipWithIndex
    .map((d, i) =>
      '{
        (
            properties: Map[String, Attribute],
            flat: Seq[DefinedInput[Def]]
        ) =>
          flat.apply(flat.length - ${ Expr(following) } + ${
            Expr(i)
          })
      }
    )

  (preceeding_exprs :+ variadic_expr) ++ following_exprs

/** Partition a construct sequence, in the case of multiple variadic definitions
  *
  * @see
  *   [[constructPartitioner]]
  */
def multivariadicConstructPartitioner[Def <: OpInputDef: Type](
    defs: Seq[Def]
)(using Quotes) =
  // Expect a coherent segmentSizes and interpret it as a list of integers.
  val segmentSizes = '{
    // TODO: This does not really belong here. Bigger fishes to fry at the time of
    // writing thoug. Conceptually this should end up in some kind of header.
    (
        properties: Map[String, Attribute],
        flat: Seq[DefinedInput[Def]]
    ) =>
      val sizes = ${ expectSegmentSizes[Def] }(properties)
      val segments = sizes.length
      val total = sizes.sum
      // Check the segmentSizes define a segment for each definition
      if segments != ${ Expr(defs.length) } then
        throw new Exception(
          s"Expected ${${ Expr(defs.length) }} entries in ${${
              Expr(getConstructName[Def])
            }}SegmentSizes, got ${segments}."
        )
      // Check the segmentSizes' sum is coherent with the number of constructs
      if total != flat.length then
        throw new Exception(
          s"${${ Expr(getConstructName[Def]) }}'s sum does not match the op's ${flat}.length} ${${
              Expr(getConstructName[Def])
            }}s."
        )
      sizes
  }
  // Partition the constructs according to the segmentSizes
  defs.zipWithIndex.map { case (d, i) =>
    getConstructVariadicity(d) match
      case Variadicity.Single =>
        '{
          (
              properties: Map[String, Attribute],
              flat: Seq[DefinedInput[Def]]
          ) => flat(${ Expr(i) })
        }
      case Variadicity.Variadic | Variadicity.Optional =>
        '{
          (
              properties: Map[String, Attribute],
              flat: Seq[DefinedInput[Def]]
          ) =>
            val sizes = ${ segmentSizes }(properties, flat)
            val start = sizes.slice(0, ${ Expr(i) }).sum
            val end = start + sizes(${ Expr(i) })
            flat.slice(start, end)
        }
  }

/** Partion constructs of a specified type. That is, check that they are in a
  * coherent quantity, and partition them into the provided definitions.
  *
  * @tparam Def
  *   The construct definition type.
  * @param defs
  *   The construct definitions.
  * @return
  *   A function of an operation and its flat sequence of constructs, returning
  *   the sequence of partitions according to the definitions.
  */
def constructPartitioner[Def <: OpInputDef: Type](
    defs: Seq[Def]
)(using Quotes) =
  // Check the number of variadic constructs
  defs.count(getConstructVariadicity(_) != Variadicity.Single) match
    case 0 => uniadicConstructPartitioner(defs)
    case 1 => univariadicConstructPartitioner(defs)
    case _ => multivariadicConstructPartitioner(defs)

/* Return an extractor for a single-defined construct
 */
def singleConstructExtractor[Def <: OpInputDef: Type, t <: Attribute: Type](
    d: Def
)(using Quotes) =
  '{ (c: DefinedInput[Def] | Seq[DefinedInput[Def]]) =>
    (c match
        case v: DefinedInputOf[Def, t] => v
        case _                         =>
          throw new Exception(
            s"Expected ${${ Expr(d.name) }} to be of type ${${
                Expr(Type.show[DefinedInputOf[Def, t]])
              }}, got ${c}"
          )
        // This somehow fails to carry type information if not casted explicitely here.
        // Including the exact same asInstanceOf in the case above.
        // I think I'm missing something..
    ).asInstanceOf[DefinedInputOf[Def, t]]
  }

/* Return an extractor for a variadic-defined construct
 */
def variadicConstructExtractor[Def <: OpInputDef: Type, t <: Attribute: Type](
    d: Def
)(using Quotes) =
  '{ (c: DefinedInput[Def] | Seq[DefinedInput[Def]]) =>
    (c match
        case s: Seq[DefinedInput[Def]] =>
          s.map(e => ${ singleConstructExtractor(d) }(e))
        case _ =>
          throw new Exception(
            s"Expected ${${ Expr(d.name) }} to be of type ${${
                Expr(Type.show[Seq[DefinedInputOf[Def, t]]])
              }}, got ${c}"
          )
        // Idem, see `singleConstructExtractor`
    ).asInstanceOf[Seq[DefinedInputOf[Def, t]]]
  }

/* Return an extractor for an optional-defined construct
 */
def optionalConstructExtractor[Def <: OpInputDef: Type, t <: Attribute: Type](
    d: Def
)(using Quotes) =
  '{ (c: DefinedInput[Def] | Seq[DefinedInput[Def]]) =>
    val cs = ${ variadicConstructExtractor(d) }(c)
    if cs.length > 1 then
      throw new Exception(
        s"Expected ${${ Expr(d.name) }} to be of type ${${
            Expr(Type.show[DefinedInputOf[Def, t]])
          }}, got ${c}"
      )
    cs.headOption
      // Idem, see `singleConstructExtractor`
      .asInstanceOf[Option[DefinedInputOf[Def, t]]]
  }

/** Returns an extractor expression for the passed construct definition.
  *
  * @param defs
  *   The constructs definitions.
  * @returns
  *   A function of a construct(s), returning the typed, extracted construct(s)
  */
def constructExtractor[Def <: OpInputDef: Type](
    d: Def
)(using Quotes) =
  getConstructConstraint(d) match
    case '[type t <: Attribute; `t`] =>
      getConstructVariadicity(d): @switch match
        case Variadicity.Single =>
          singleConstructExtractor[Def, t](d)
        case Variadicity.Variadic =>
          variadicConstructExtractor[Def, t](d)
        case Variadicity.Optional =>
          optionalConstructExtractor[Def, t](d)

def extractedConstructs[Def <: OpInputDef: Type](
    defs: Seq[Def],
    flat: Expr[Seq[DefinedInput[Def]]],
    properties: Expr[Map[String, Attribute]]
)(using Quotes) =
  // partition the constructs according to their definitions
  val partitioned =
    constructPartitioner(defs).map(p => '{ ${ p }($properties, $flat) })

  // extract the constructs
  (partitioned zip defs).map { (c, d) =>
    '{ ${ constructExtractor(d) }(${ c }) }
  }

/** Return all named arguments for the primary constructor of an ADT. Those are
  * checked, in the sense that they are checked to be of the correct types and
  * numbers.
  *
  * @param opDef
  *   The OperationDef derived from the ADT.
  * @param op
  *   The UnstructuredOp instance.
  * @return
  *   The checked named arguments for the primary constructor of the ADT.
  */

def tryConstruct[T: Type](
    opDef: OperationDef,
    operands: Expr[Seq[Operand[Attribute]]],
    results: Expr[Seq[Result[Attribute]]],
    regions: Expr[Seq[Region]],
    successors: Expr[Seq[Successor]],
    properties: Expr[Map[String, Attribute]]
)(using Quotes) =
  import quotes.reflect.*
  val args = (extractedConstructs(
    opDef.operands,
    operands,
    properties
  ) zip opDef.operands)
    .map((e, d) => NamedArg(d.name, e.asTerm)) ++
    (extractedConstructs(opDef.results, results, properties) zip opDef.results)
      .map((e, d) => NamedArg(d.name, e.asTerm)) ++ (extractedConstructs(
      opDef.regions,
      regions,
      properties
    ) zip opDef.regions).map((e, d) =>
      NamedArg(d.name, e.asTerm)
    ) ++ (extractedConstructs(
      opDef.successors,
      successors,
      properties
    ) zip opDef.successors).map((e, d) => NamedArg(d.name, e.asTerm)) ++
    opDef.properties.map { case OpPropertyDef(name, tpe, variadicity, _) =>
      tpe match
        case '[type t <: Attribute; `t`] =>
          val property = variadicity match
            case Variadicity.Optional =>
              generateOptionalCheckedPropertyArgument[t](
                properties,
                name
              )
            case Variadicity.Single =>
              generateCheckedPropertyArgument[t](
                properties,
                name
              )
            case Variadicity.Variadic =>
              report.errorAndAbort(
                s"Properties cannot be variadic in an ADT."
              )
          NamedArg(name, property.asTerm)
    }
  // Return a call to the primary constructor of the ADT.
  Apply(
    Select(New(TypeTree.of[T]), TypeRepr.of[T].typeSymbol.primaryConstructor),
    List.from(args)
  ).asExprOf[T]

  /** Attempt to create an ADT from an UnstructuredOp[ADT]
    *
    * @tparam T
    *   The ADT Type.
    * @param opDef
    *   The OperationDef derived from the ADT.
    * @param genExpr
    *   The expression of the UnstructuredOp[ADT].
    * @return
    *   The ADT instance.
    * @raises
    *   Exception if the UnstructuredOp[ADT] is not valid to represent by the
    *   ADT.
    */

def fromUnstructuredOperationMacro[T: Type](
    opDef: OperationDef,
    genExpr: Expr[DerivedOperationCompanion[T]#UnstructuredOp]
)(using Quotes): Expr[T] =

  // Create named arguments for all of the ADT's constructor arguments.
  tryConstruct(
    opDef,
    '{ $genExpr.operands },
    '{ $genExpr.results },
    '{ $genExpr.detached_regions },
    '{ $genExpr.successors },
    '{ $genExpr.properties }
  )

def getAttrConstructor[T: Type](
    attrDef: AttributeDef,
    attributes: Expr[Seq[Attribute]]
)(using
    Quotes
): Expr[T] =
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

  val extractedConstructs = (defs.zipWithIndex.map((d, i) =>
    '{ ${ attributes }(${ Expr(i) }) }
  ) zip defs)
    .map { (a, d) =>
      // expected type of the attribute
      val tpe = d.tpe
      tpe match
        case '[t] =>
          '{
            if !${ a }.isInstanceOf[t] then
              throw Exception(
                s"Expected ${${ Expr(d.name) }} to be of type ${${
                    Expr(Type.show[t])
                  }}, got ${${ a }}"
              )
            ${ a }.asInstanceOf[t]
          }
    }

  val args = (extractedConstructs zip attrDef.attributes)
    .map((e, d) => NamedArg(d.name, e.asTerm))

  val constructorCall = Apply(
    Select(New(TypeTree.of[T]), TypeRepr.of[T].typeSymbol.primaryConstructor),
    List.from(args)
  ).asExprOf[T]

  '{
    $lengthCheck
    $constructorCall
  }

def ADTFlatAttrInputMacro[Def <: AttributeDef: Type](
    attrInputDefs: Seq[AttributeParamDef],
    adtAttrExpr: Expr[?]
)(using Quotes): Expr[Seq[Attribute]] =
  Expr.ofList(
    attrInputDefs.map(d => adtAttrExpr.member[Attribute](d.name))
  )

def parametersMacro(
    attrDef: AttributeDef,
    adtAttrExpr: Expr[?]
)(using Quotes): Expr[Seq[Attribute]] =
  ADTFlatAttrInputMacro(attrDef.attributes, adtAttrExpr)

def derivedAttributeCompanion[T <: Attribute: Type](using
    Quotes
): Expr[DerivedAttributeCompanion[T]] =

  val attrDef = getAttrDefImpl[T]

  '{
    new DerivedAttributeCompanion[T]:
      override def name: String = ${ Expr(attrDef.name) }
      override def parse[$: P as ctx](p: AttrParser): P[T] = ${
        getAttrCustomParse[T]('{ p }, '{ ctx }).getOrElse(
          '{
            P(
              ("<" ~/ p.Attribute.rep(sep = ",") ~ ">")
            ).orElse(Seq())
              .map(x => ${ getAttrConstructor[T](attrDef, '{ x }) })
          }
        )
      }
      def parameters(attr: T): Seq[Attribute | Seq[Attribute]] = ${
        parametersMacro(attrDef, '{ attr })
      }
  }

def deriveOperationCompanion[T <: Operation: Type](using
    Quotes
): Expr[DerivedOperationCompanion[T]] =
  val opDef = getDefImpl[T]

  val summonedPatterns = Expr.summon[CanonicalizationPatterns[T]] match
    case Some(canonicalizationPatterns) =>
      '{ $canonicalizationPatterns.patterns }
    case None => '{ Seq() }

  '{

    new DerivedOperationCompanion[T]:

      override def canonicalizationPatterns: Seq[RewritePattern] =
        $summonedPatterns

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

      def custom_print(adtOp: T, p: Printer)(using indentLevel: Int): Unit =
        ${ customPrintMacro(opDef, '{ adtOp }, '{ p }, '{ indentLevel }) }

      def constraint_verify(adtOp: T): Either[String, Operation] =
        ${
          verifyMacro(opDef, '{ adtOp })
        }

      override def parse[$: P as ctx](
          p: Parser,
          resNames: Seq[String]
      ): P[Operation] =
        ${
          (getOpCustomParse[T]('{ p }, '{ resNames })
            .getOrElse(parseMacro(opDef, '{ p }, '{ resNames })))
        }(using ctx)

      def apply(
          operands: Seq[Value[Attribute]] = Seq(),
          successors: Seq[Block] = Seq(),
          results: Seq[Result[Attribute]] = Seq(),
          regions: Seq[Region] = Seq(),
          properties: Map[String, Attribute] = Map.empty[String, Attribute],
          attributes: DictType[String, Attribute] =
            DictType.empty[String, Attribute]
      ): UnstructuredOp | T & Operation =
        try {
          val structured = ${
            tryConstruct(
              opDef,
              '{ operands },
              '{ results },
              '{ regions },
              '{ successors },
              '{ properties }
            )
          }
          structured.attributes.addAll(attributes)
          structured
        } catch { _ =>
          UnstructuredOp(
            operands = operands,
            successors = successors,
            results = results,
            regions = regions,
            properties = properties,
            attributes = attributes
          )
        }

      def destructure(adtOp: T): UnstructuredOp =
        UnstructuredOp(
          operands = operands(adtOp),
          successors = successors(adtOp),
          results = results(adtOp),
          regions = regions(adtOp).map(_.detached),
          properties = properties(adtOp),
          attributes = adtOp.attributes
        )

      def structure(unstrucOp: UnstructuredOp): T =
        ${
          fromUnstructuredOperationMacro[T](opDef, '{ unstrucOp })
        } match
          case adt: DerivedOperation[?, T] =>
            adt.attributes.addAll(unstrucOp.attributes)
            adt
          case _ =>
            throw new Exception(
              s"Internal Error: Hacky did not hack -> T is not a DerivedOperation: ${unstrucOp}"
            )

  }
