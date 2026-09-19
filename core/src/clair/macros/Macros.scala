package scair.clair.macros

import fastparse.*
import scair.*
import scair.clair.*
import scair.dialects.builtin.*
import scair.enums.*
import scair.ir.*
import scair.parse.*
import scair.print.Printer
import scair.transformations.CanonicalizationPatterns
import scair.transformations.RewritePattern
import scair.utils.*

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

/** Small helper to select a member of an expression.
  * @param obj
  *   The object to select the member from.
  * @param name
  *   The name of the member to select.
  */
def selectMember[T: Type](obj: Expr[?], name: String)(using
    Quotes
): Expr[T] =
  import quotes.reflect.*

  Select.unique(obj.asTerm, name).asExprOf[T]

def makeSegmentSizes[T <: MayVariadicOpInputDef: Type](
    hasMultiVariadic: Boolean,
    defs: Seq[T],
    adtOpExpr: Expr[?],
)(using Quotes): Option[(Expr[String], Expr[Attribute])] =
  val name = s"${getConstructName[T]}SegmentSizes"
  hasMultiVariadic match
    case true =>
      val arrayAttr: Expr[Seq[Int]] =
        Expr.ofList(
          defs.map((d) =>
            d.variadicity match
              case Variadicity.Single   => Expr(1)
              case Variadicity.Variadic =>
                '{
                  ${ selectMember[Seq[?]](adtOpExpr, d.name) }.length
                }
              case Variadicity.Optional =>
                '{
                  ${ selectMember[Option[?]](adtOpExpr, d.name) }.size
                }
          )
        )
      Some(
        Expr(name),
        '{
          DenseArrayAttr(
            IntegerType(IntData(32), Signless),
            ${ arrayAttr }.map(x =>
              IntegerAttr(
                IntData(x),
                IntegerType(IntData(32), Signless),
              )
            ),
          )
        },
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
    adtOpExpr: Expr[?],
)(using Quotes): Expr[Seq[DefinedInput[Def]]] =
  def variadicity(d: Def): Variadicity = d match
    case d: MayVariadicOpInputDef => d.variadicity

  opInputDefs.toList match
    case Nil =>
      // No inputs, optimized empty sequence.
      '{ Seq.empty[DefinedInput[Def]] }

    case d :: Nil =>
      variadicity(d) match
        case Variadicity.Single =>
          // One fixed input - simple constructor.
          val input = selectMember[DefinedInput[Def]](adtOpExpr, d.name)
          '{ Seq($input) }
        case Variadicity.Variadic =>
          // One variadic input - return as-is.
          selectMember[Seq[DefinedInput[Def]]](adtOpExpr, d.name)
        case Variadicity.Optional =>
          // One optional input - convert to sequence.
          val input =
            selectMember[Option[DefinedInput[Def]]](adtOpExpr, d.name)
          '{
            $input match
              case Some(value) => Seq(value)
              case None        => Seq.empty[DefinedInput[Def]]
          }

    case defs if defs.forall(variadicity(_) == Variadicity.Single) =>
      // Multiple fixed inputs - straightforward constructor.
      Expr
        .ofSeq(
          defs.map(d => selectMember[DefinedInput[Def]](adtOpExpr, d.name))
        )

    case defs =>
      // Multiple inputs with optional or variadic fields - compute the runtime total size.
      val totalSize = defs.foldLeft(Expr(0))((size, d) =>
        variadicity(d) match
          case Variadicity.Single   => '{ $size + 1 }
          case Variadicity.Variadic =>
            val input =
              selectMember[Seq[DefinedInput[Def]]](adtOpExpr, d.name)
            '{ $size + $input.length }
          case Variadicity.Optional =>
            val input =
              selectMember[Option[DefinedInput[Def]]](adtOpExpr, d.name)
            '{ $size + $input.size }
      )
      // Generate a filling of the mutable array.
      def fillArray(
          remaining: List[Def],
          array: Expr[Array[AnyRef]],
          index: Expr[Int],
      ): Expr[Unit] = remaining match
        case Nil       => '{ () }
        case d :: tail =>
          variadicity(d) match
            case Variadicity.Single =>
              // Copy one fixed input.
              val input = selectMember[DefinedInput[Def]](adtOpExpr, d.name)
              '{
                $array($index) = $input.asInstanceOf[AnyRef]
                ${ fillArray(tail, array, '{ $index + 1 }) }
              }
            case Variadicity.Variadic =>
              // Copy one variadic input.
              val input =
                selectMember[Seq[DefinedInput[Def]]](adtOpExpr, d.name)
              '{
                val copied = $input.copyToArray($array, $index)
                ${ fillArray(tail, array, '{ $index + copied }) }
              }
            case Variadicity.Optional =>
              // Copy one optional input.
              val input =
                selectMember[Option[DefinedInput[Def]]](adtOpExpr, d.name)
              '{
                val copied = $input match
                  case Some(value) =>
                    $array($index) = value.asInstanceOf[AnyRef]
                    1
                  case None => 0
                ${ fillArray(tail, array, '{ $index + copied }) }
              }

      '{
        val size = $totalSize
        // If there are no inputs, return the optimized empty sequence.
        if size == 0 then Seq.empty[DefinedInput[Def]]
        else
          // Else, preallocate an array, fill it and return.
          val array = new Array[AnyRef](size)
          ${ fillArray(defs, '{ array }, Expr(0)) }
          // Every DefinedInput alternative is a reference type, so an AnyRef array
          // can safely back the covariant immutable result sequence.
          scala.collection.immutable.ArraySeq.unsafeWrapArray(array)
            .asInstanceOf[Seq[DefinedInput[Def]]]
      }

def operandsMacro(
    opDef: OperationDef,
    adtOpExpr: Expr[?],
)(using Quotes): Expr[Seq[Operand[Attribute]]] =
  ADTFlatInputMacro(opDef.operands, adtOpExpr)

def successorsMacro(
    opDef: OperationDef,
    adtOpExpr: Expr[?],
)(using Quotes): Expr[Seq[Successor]] =
  ADTFlatInputMacro(opDef.successors, adtOpExpr)

def resultsMacro(
    opDef: OperationDef,
    adtOpExpr: Expr[?],
)(using Quotes): Expr[Seq[Result[Attribute]]] =
  ADTFlatInputMacro(opDef.results, adtOpExpr)

def regionsMacro(
    opDef: OperationDef,
    adtOpExpr: Expr[?],
)(using Quotes): Expr[Seq[Region]] =
  ADTFlatInputMacro(opDef.regions, adtOpExpr)

def propertiesMacro(
    opDef: OperationDef,
    adtOpExpr: Expr[?],
)(using Quotes): Expr[Map[String, Attribute]] =

  val opSegSizeProp = makeSegmentSizes(
    opDef.hasMultiVariadicOperands && !opDef.sameVariadicOperandSize,
    opDef.operands,
    adtOpExpr,
  )
  val resSegSizeProp = makeSegmentSizes(
    opDef.hasMultiVariadicResults && !opDef.sameVariadicResultSize,
    opDef.results,
    adtOpExpr,
  )
  val regSegSizeProp = makeSegmentSizes(
    opDef.hasMultiVariadicRegions,
    opDef.regions,
    adtOpExpr,
  )
  val succSegSizeProp = makeSegmentSizes(
    opDef.hasMultiVariadicSuccessors,
    opDef.successors,
    adtOpExpr,
  )
  // Populating a Dictionary with the properties
  val mandatoryProps =
    opDef.properties.collect {
      case OpPropertyDef(name = name, variadicity = Variadicity.Single) =>
        (Expr(name), selectMember[Attribute](adtOpExpr, name))
    } ++ opSegSizeProp ++ resSegSizeProp ++ regSegSizeProp ++ succSegSizeProp

  val optionalProps =
    opDef.properties.collect {
      case OpPropertyDef(name = name, variadicity = Variadicity.Optional) =>
        (Expr(name), selectMember[Option[Attribute]](adtOpExpr, name))
    }
  // Properties are typically few; a chain of `updated` goes through the
  // small specialized maps without a builder or tuples along the way.
  val withMandatory = mandatoryProps
    .foldLeft('{
      Map.empty[String, Attribute]
    })((props, prop) => '{ $props.updated(${ prop._1 }, ${ prop._2 }) })
  optionalProps.foldLeft(withMandatory)((props, prop) =>
    '{
      val current = $props
      ${ prop._2 } match
        case Some(value) => current.updated(${ prop._1 }, value)
        case None        => current
    }
  )

def customPrintMacro(
    opDef: OperationDef,
    adtOpExpr: Expr[?],
    p: Expr[Printer],
)(using Quotes): Expr[Unit] =
  opDef.assemblyFormat match
    case Some(format) =>
      format.print(opDef, adtOpExpr, p)
    case None =>
      '{
        $p.printGenericMLIROperation(${ adtOpExpr }.asInstanceOf[Operation])
      }

def parseMacro[O <: Operation: Type](
    opDef: OperationDef,
    p: Expr[Parser],
    resNames: Expr[Seq[String]],
)(using
    Quotes
): Expr[P[Any] ?=> P[O]] =
  opDef.assemblyFormat match
    case Some(format) =>
      format.parse(opDef, p, resNames)
    case None =>
      '{
        Fail(
          s"No custom Parser implemented for Operation '${${
              Expr(opDef.name)
            }}'"
        )
      }

/** Verify that all variadic definitions of a construct indeed hold the same
  * number of constructs, as the op declares with a `SameVariadic*Size` trait.
  *
  * @param defs
  *   The construct definitions of the marked construct.
  * @param marker
  *   The name of the trait the op is marked with, for diagnostics.
  */
def sameVariadicSizeVerifier[Def <: MayVariadicOpInputDef: Type](
    opName: String,
    defs: Seq[Def],
    marker: String,
    adtOpExpr: Expr[?],
)(using Quotes): Expr[OK[Unit]] =
  val variadics = defs.filter(_.variadicity != Variadicity.Single)
  // Each variadic definition, paired with the number of constructs it holds.
  val sizes = Expr.ofList(
    variadics.map(d =>
      '{
        (
          ${ Expr(d.name) },
          ${ selectMember[Iterable[?]](adtOpExpr, d.name) }.size,
        )
      }
    )
  )
  '{
    val named = $sizes
    if named.map(_._2).distinct.length > 1 then
      scair.utils.Err(
        s"Operation '${${ Expr(opName) }}' is marked ${${
            Expr(marker)
          }}, but its variadic ${${
            Expr(getConstructName[Def])
          }}s have differing sizes: ${named
            .map((name, size) => s"$name ($size)").mkString(", ")}",
        Some($adtOpExpr.asInstanceOf[Operation]),
      )
    else OK()
  }

/** Generate the constraint verification of an op: its `SameVariadic*Size`
  * checks, followed by its operand constraints, short-circuiting on the first
  * error and yielding the op otherwise.
  *
  * Only what the op declares is generated; an op without any check verifies to
  * itself, allocation-free.
  */
def verifyMacro(
    opDef: OperationDef,
    adtOpExpr: Expr[?],
)(using Quotes): Expr[OK[Operation]] =
  val op = '{ $adtOpExpr.asInstanceOf[Operation] }

  val sameSizes: Seq[Expr[OK[Unit]]] =
    Option
      .when(opDef.sameVariadicOperandSize)(
        sameVariadicSizeVerifier(
          opDef.name,
          opDef.operands,
          "SameVariadicOperandSize",
          adtOpExpr,
        )
      ).toSeq ++ Option.when(opDef.sameVariadicResultSize)(
      sameVariadicSizeVerifier(
        opDef.name,
        opDef.results,
        "SameVariadicResultSize",
        adtOpExpr,
      )
    )

  // Operand constraints share one context (e.g. for `Var` constraints); each
  // check is generated against the context expression it is given.
  val constraints
      : Seq[Expr[scair.constraints.ConstraintContext] => Expr[OK[Unit]]] =
    opDef.operands.collect {
      case OperandDef(name, _, Variadicity.Single, Some(constraint)) =>
        val typ = '{
          ${ selectMember[Operand[Attribute]](adtOpExpr, name) }.typ
        }
        ctx => '{ $constraint.verify($typ)(using $ctx) }
    }

  // Sequence the checks, short-circuiting on the first error, and yield the op.
  def chain(checks: Seq[Expr[OK[Unit]]]): Expr[OK[Operation]] =
    checks.reduceOption((done, next) => '{ $done.flatMap(_ => $next) }) match
      case None      => '{ OK($op) }
      case Some(all) => '{ $all.map(_ => $op) }

  if constraints.isEmpty then chain(sameSizes)
  else
    '{
      val ctx = scair.constraints.ConstraintContext()
      ${ chain(sameSizes ++ constraints.map(_('{ ctx }))) }
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
    propName: String,
    defaultValue: Option[Expr[Any]],
)(using Quotes): Expr[A] =
  val typeName = Type.of[A].toString()
  val ifAbsent = defaultValue match
    case Some(default) => default.asExprOf[A]
    case None          =>
      '{
        throw new IllegalArgumentException(
          s"Missing required property \"${${ Expr(propName) }}\" of type ${${
              Expr(typeName)
            }}"
        )
      }
  '{
    val value: Option[Attribute] = $list.get(${ Expr(propName) })
    value match
      case None          => $ifAbsent
      case Some(prop: A) => prop
      case Some(value)   =>
        throw new IllegalArgumentException(
          s"Type mismatch for property \"${${ Expr(propName) }}\": " +
            s"expected ${${ Expr(typeName) }}, " +
            s"but found ${value.getClass}"
        )
  }

def generateOptionalCheckedPropertyArgument[A <: Attribute: Type](
    list: Expr[Map[String, Attribute]],
    propName: String,
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
    op: Expr[OpDefs[?]#UnstructuredOp]
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
    case OperandDef(tpe = tpe)    => tpe
    case ResultDef(tpe = tpe)     => tpe
    case _: RegionDef             => Type.of[Attribute]
    case _: SuccessorDef          => Type.of[Attribute]
    case OpPropertyDef(tpe = tpe) => tpe

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
          flat: Seq[DefinedInput[Def]],
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
      a == Variadicity.Variadic || a == Variadicity.Optional
    )
  val following = defs.length - preceeding - 1
  val preceedingExprs = defs.slice(0, preceeding).zipWithIndex.map((d, i) =>
    '{

      (
          properties: Map[String, Attribute],
          flat: Seq[DefinedInput[Def]],
      ) => flat.apply(${ Expr(i) })
    }
  )

  val variadicExpr = '{
    (
        properties: Map[String, Attribute],
        flat: Seq[DefinedInput[Def]],
    ) => flat.slice(${ Expr(preceeding) }, flat.length - ${ Expr(following) })
  }

  val followingExprs = defs.slice(preceeding + 1, defs.length).zipWithIndex.map(
    (d, i) =>
      '{
        (
            properties: Map[String, Attribute],
            flat: Seq[DefinedInput[Def]],
        ) =>
          flat
            .apply(flat.length - ${ Expr(following) } + ${
              Expr(i)
            })
      }
  )

  (preceedingExprs :+ variadicExpr) ++ followingExprs

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
        flat: Seq[DefinedInput[Def]],
    ) =>
      val sizes = ${ expectSegmentSizes[Def] }(properties)
      val segments = sizes.length
      val total = sizes.sum
      // Check the segmentSizes define a segment for each definition
      if segments != ${ Expr(defs.length) } then
        throw new Exception(
          s"Expected ${${ Expr(defs.length) }} entries in ${${
              Expr(getConstructName[Def])
            }}SegmentSizes, got $segments."
        )
      // Check the segmentSizes' sum is coherent with the number of constructs
      if total != flat.length then
        throw new Exception(
          s"${${ Expr(getConstructName[Def]) }}'s sum does not match the op's $flat.length} ${${
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
              flat: Seq[DefinedInput[Def]],
          ) => flat(${ Expr(i) })
        }
      case Variadicity.Variadic | Variadicity.Optional =>
        '{
          (
              properties: Map[String, Attribute],
              flat: Seq[DefinedInput[Def]],
          ) =>
            val sizes = ${ segmentSizes }(properties, flat)
            val start = sizes.slice(0, ${ Expr(i) }).sum
            val end = start + sizes(${ Expr(i) })
            flat.slice(start, end)
        }
  }

/** Partition a construct sequence, in the case of multiple variadic definitions
  * declared to all hold the same number of constructs.
  *
  * The flat sequence is split evenly over the variadic definitions, rather than
  * according to a segment sizes property.
  *
  * @see
  *   [[constructPartitioner]]
  */
def sameSizeConstructPartitioner[Def <: OpInputDef: Type](
    defs: Seq[Def]
)(using Quotes) =
  val variadicities = defs.map(getConstructVariadicity(_))
  val variadics = variadicities.count(_ != Variadicity.Single)
  val singles = defs.length - variadics
  // An optional definition holds at most one construct, which caps the shared
  // size all variadic definitions must agree on.
  val hasOptional = variadicities.contains(Variadicity.Optional)

  // The number of constructs each variadic definition holds, deduced from the
  // total. Kept as an expression as it is only known at runtime.
  val sharedSize = '{ (flat: Seq[DefinedInput[Def]]) =>
    val variable = flat.length - ${ Expr(singles) }
    if variable < 0 || variable % ${ Expr(variadics) } != 0 then
      throw new Exception(
        s"Expected ${${ Expr(singles) }} ${${
            Expr(getConstructName[Def])
          }}s plus a multiple of ${${
            Expr(variadics)
          }} same-sized variadic ones, got ${flat.length}."
      )
    val size = variable / ${ Expr(variadics) }
    if ${ Expr(hasOptional) } && size > 1 then
      throw new Exception(
        s"Expected at most one ${${
            Expr(getConstructName[Def])
          }} per variadic definition, as one of them is optional, got $size."
      )
    size
  }

  // Each definition starts after the preceeding single ones, plus a shared size
  // worth of constructs for each preceeding variadic one.
  val starts = variadicities.scanLeft((0, 0))((counts, variadicity) =>
    variadicity match
      case Variadicity.Single => (counts._1 + 1, counts._2)
      case Variadicity.Variadic | Variadicity.Optional =>
        (counts._1, counts._2 + 1)
  )

  (defs zip starts).map { case (d, (singlesBefore, variadicsBefore)) =>
    val start = '{ (flat: Seq[DefinedInput[Def]]) =>
      ${ Expr(singlesBefore) } +
        ${ Expr(variadicsBefore) } * ${ sharedSize }(flat)
    }
    getConstructVariadicity(d) match
      case Variadicity.Single =>
        '{
          (
              properties: Map[String, Attribute],
              flat: Seq[DefinedInput[Def]],
          ) => flat(${ start }(flat))
        }
      case Variadicity.Variadic | Variadicity.Optional =>
        '{
          (
              properties: Map[String, Attribute],
              flat: Seq[DefinedInput[Def]],
          ) =>
            val from = ${ start }(flat)
            flat.slice(from, from + ${ sharedSize }(flat))
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
    defs: Seq[Def],
    sameVariadicSize: Boolean = false,
)(using Quotes) =
  // Check the number of variadic constructs
  defs.count(getConstructVariadicity(_) != Variadicity.Single) match
    case 0                     => uniadicConstructPartitioner(defs)
    case 1                     => univariadicConstructPartitioner(defs)
    case _ if sameVariadicSize => sameSizeConstructPartitioner(defs)
    case _                     => multivariadicConstructPartitioner(defs)

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
              }}, got $c"
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
              }}, got $c"
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
          }}, got $c"
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
    properties: Expr[Map[String, Attribute]],
    sameVariadicSize: Boolean = false,
)(using Quotes) =
  // partition the constructs according to their definitions
  val partitioned =
    constructPartitioner(defs, sameVariadicSize)
      .map(p => '{ ${ p }($properties, $flat) })

  // extract the constructs
  (partitioned zip defs).map((c, d) => '{ ${ constructExtractor(d) }(${ c }) })

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
    properties: Expr[Map[String, Attribute]],
)(using Quotes) =
  import quotes.reflect.*
  val args =
    (extractedConstructs(
      opDef.operands,
      operands,
      properties,
      opDef.sameVariadicOperandSize,
    ) zip opDef.operands).map((e, d) => NamedArg(d.name, e.asTerm)) ++
      (extractedConstructs(
        opDef.results,
        results,
        properties,
        opDef.sameVariadicResultSize,
      ) zip opDef.results).map((e, d) => NamedArg(d.name, e.asTerm)) ++
      (extractedConstructs(
        opDef.regions,
        regions,
        properties,
      ) zip opDef.regions).map((e, d) => NamedArg(d.name, e.asTerm)) ++
      (extractedConstructs(
        opDef.successors,
        successors,
        properties,
      ) zip opDef.successors).map((e, d) => NamedArg(d.name, e.asTerm)) ++
      opDef.properties.map {
        case OpPropertyDef(name, tpe, variadicity, _, defaultValue) =>
          val namedArg = tpe match
            case '[type t <: scala.reflect.Enum & IntegerEnumAttr; `t`] =>
              val property = variadicity match
                case Variadicity.Optional =>
                  enumFromPropertyOption[t](
                    properties,
                    name,
                  )
                case Variadicity.Single =>
                  enumFromProperty[t](
                    properties,
                    name,
                  )
              NamedArg(name, property.asTerm)
            case '[type t <: Attribute; `t`] =>
              val property = variadicity match
                case Variadicity.Optional =>
                  generateOptionalCheckedPropertyArgument[t](
                    properties,
                    name,
                  )
                case Variadicity.Single =>
                  generateCheckedPropertyArgument[t](
                    properties,
                    name,
                    defaultValue,
                  )
              NamedArg(name, property.asTerm)
          namedArg
      }
  // Return a call to the primary constructor of the ADT.
  Apply(
    Select(New(TypeTree.of[T]), TypeRepr.of[T].typeSymbol.primaryConstructor),
    List.from(args),
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

def fromUnstructuredOperationMacro[T <: Operation: Type](
    opDef: OperationDef,
    genExpr: Expr[OpDefs[T]#UnstructuredOp],
)(using Quotes): Expr[T] =

  // Create named arguments for all of the ADT's constructor arguments.
  tryConstruct(
    opDef,
    '{ $genExpr.operands },
    '{ $genExpr.results },
    '{ $genExpr.detachedRegions },
    '{ $genExpr.successors },
    '{ $genExpr.properties },
  )

def getAttrConstructor[T: Type](
    attrDef: AttributeDef,
    attributes: Expr[Seq[Attribute]],
)(using
    Quotes
): Expr[T] =
  import quotes.reflect.*

  val lengthCheck = Type.of[T] match
    case '[type t <: Attribute; `t`] =>
      '{
        if ${ Expr(attrDef.attributes.length) } != $attributes.length then
          throw new Exception(
            s"Number of attributes ${${ Expr(attrDef.attributes.length) }} does not match the number of provided attributes ${$attributes
                .length}"
          )
      }
    case _ =>
      report
        .errorAndAbort(
          s"Type ${Type.show[T]} needs to be a subtype of Attribute"
        )

  val defs = attrDef.attributes

  val extractedConstructs =
    (defs.zipWithIndex.map((d, i) => '{ ${ attributes }(${ Expr(i) }) }) zip
      defs).map { (a, d) =>
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
    List.from(args),
  ).asExprOf[T]

  '{
    $lengthCheck
    $constructorCall
  }

def ADTFlatAttrInputMacro[Def <: AttributeDef: Type](
    attrInputDefs: Seq[AttributeParamDef],
    adtAttrExpr: Expr[?],
)(using Quotes): Expr[Seq[Attribute]] =
  Expr
    .ofList(
      attrInputDefs.map(d => selectMember[Attribute](adtAttrExpr, d.name))
    )

def parametersMacro(
    attrDef: AttributeDef,
    adtAttrExpr: Expr[?],
)(using Quotes): Expr[Seq[Attribute]] =
  ADTFlatAttrInputMacro(attrDef.attributes, adtAttrExpr)

/** Builds `self.p1 == that.p1 && ... && self.pN == that.pN`.
  *
  * Each parameter is selected at its declared type rather than as `Any`, so the
  * comparisons call `Attribute.equals` directly instead of going through
  * `BoxesRunTime.equals`. An attribute with no parameters compares equal to any
  * other instance of its class, which is what its single inhabitant warrants.
  */
def equalMacro[T <: Attribute: Type](
    attrDef: AttributeDef,
    selfExpr: Expr[T],
    otherExpr: Expr[Any],
)(using Quotes): Expr[Boolean] =
  '{
    val that = $otherExpr.asInstanceOf[T]
    ${
      attrDef.attributes
        .map(d =>
          d.tpe match
            case '[t] =>
              '{
                ${ selectMember[t](selfExpr, d.name) } ==
                  ${ selectMember[t]('{ that }, d.name) }
              }
        )
        .reduceOption((l, r) => '{ $l && $r })
        .getOrElse('{ true })
    }
  }

def deriveAttrDefs[T <: Attribute: Type](using
    Quotes
): Expr[AttrDefs[T]] =

  val attrDef = getAttrDefImpl[T]

  '{
    new AttrDefs[T]:
      override def name: String = ${ Expr(attrDef.name) }
      override def parse[$: P as ctx](using p: Parser): P[T] = ${
        getAttrCustomParse[T]('{ p }, '{ ctx }).getOrElse(
          '{
            given Whitespace = scair.parse.whitespace
            given Parser = p
            ("<" ~/ attributeP.rep(sep = ",")(using scair.parse.listRepeater) ~
              ">").orElse(Seq())
              .map(x => ${ getAttrConstructor[T](attrDef, '{ x }) })
          }
        )
      }
      def parameters(attr: T): Seq[Attribute] = ${
        parametersMacro(attrDef, '{ attr })
      }
      def equal(self: T, other: Any): Boolean = ${
        equalMacro[T](attrDef, '{ self }, '{ other })
      }
  }

def deriveOpDefs[T <: Operation: Type](using
    Quotes
): Expr[OpDefs[T]] =
  val opDef = getDefImpl[T]

  val summonedPatterns = Expr.summon[CanonicalizationPatterns[T]] match
    case Some(canonicalizationPatterns) =>
      '{ $canonicalizationPatterns.patterns }
    case None => '{ Seq() }

  '{

    new OpDefs[T]:

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

      def customPrint(adtOp: T, p: Printer): Unit =
        ${ customPrintMacro(opDef, '{ adtOp }, '{ p }) }

      def constraintVerify(adtOp: T): OK[Operation] =
        ${
          verifyMacro(opDef, '{ adtOp })
        }

      override def parse[$: P as ctx](
          resNames: Seq[String]
      )(using p: Parser): P[T] =
        ${
          (getOpCustomParse[T]('{ p }, '{ resNames })
            .getOrElse(parseMacro[T](opDef, '{ p }, '{ resNames })))
        }(using ctx)

      def apply(
          operands: Seq[Value[Attribute]] = Seq(),
          successors: Seq[Block] = Seq(),
          results: Seq[Result[Attribute]] = Seq(),
          regions: Seq[Region] = Seq(),
          properties: Map[String, Attribute] = Map.empty[String, Attribute],
          attributes: Map[String, Attribute] = Map.empty[String, Attribute],
      ): UnstructuredOp | T & Operation =
        try {
          val structured = ${
            tryConstruct(
              opDef,
              '{ operands },
              '{ results },
              '{ regions },
              '{ successors },
              '{ properties },
            )
          }
          structured.withAttributes(attributes)
        } catch { _ =>
          UnstructuredOp(
            operands = operands,
            successors = successors,
            results = results,
            regions = regions,
            properties = properties,
            attributes = attributes,
          )
        }

      def destructure(adtOp: T): UnstructuredOp =
        UnstructuredOp(
          operands = operands(adtOp),
          successors = successors(adtOp),
          results = results(adtOp),
          regions = regions(adtOp).map(_.detached),
          properties = properties(adtOp),
          attributes = adtOp.attributes,
        )

      def structure(unstrucOp: UnstructuredOp): T =
        ${
          fromUnstructuredOperationMacro[T](opDef, '{ unstrucOp })
        } match
          case adt: DerivedOperation[?] =>
            adt.withAttributes(unstrucOp.attributes)
          case _ =>
            throw new Exception(
              s"Internal Error: Hacky did not hack -> T is not a DerivedOperation: $unstrucOp"
            )

  }
