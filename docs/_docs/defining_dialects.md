---
title: "Defining a Dialect"
---

[Attribute]: scair.ir.Attribute
[TypeAttribute]: scair.ir.TypeAttribute
[Operation]: scair.ir.Operation
[NoMemoryEffect]: scair.ir.NoMemoryEffect
[IsTerminator]: scair.ir.IsTerminator
[IsolatedFromAbove]: scair.ir.IsolatedFromAbove
[DerivedAttribute]: scair.clair.DerivedAttribute
[DerivedOperation]: scair.clair.DerivedOperation
[derives OpDefs]: scair.clair.DerivedOperationCompanion
[derives AttrDefs]: scair.clair.AttrDefs

# Defining a Dialect
This tutorial explains how to define new attributes and operations in ScaIR and how to package these into a dialect.

## Defining Attributes

Attributes represent compile-time information in the IR. They are immutable and may appear:

* as SSA types
* as constant values
* as metadata attached to operations

In ScaIR, all attributes extend the base [Attribute] hierarchy.

### Attributes vs Types

In MLIR (and conceptually in ScaIR), types are a specialized kind of attribute. Attributes represent general compile-time information. Type attributes are used to describe the types of SSA values.

In MLIR every SSA value has exactly one type. In ScaIR the distinction between `Attribute` and `TypeAttribute` is primarily maintained for MLIR IR compatibility (e.g., printing `#` vs `!` for MLIR dialects). 
SSA values in ScaIR do not strictly require a `TypeAttribute`.

This distinction is reflected in the IR syntax:

- `#dialect.attr<...>` — general attributes
- `!dialect.type<...>` — type attributes

Although both are implemented as attributes internally, only type attributes should appear in SSA value type positions.

In ScaIR, this distinction is expressed explicitly in Scala: type attributes extend `TypeAttribute`, while other attributes do not.

### Type Attributes

`TypeAttribute` describes the types of SSA values. While MLIR requires every SSA value to have exactly one type attribute, ScaIR allows SSA values to be typed using regular attributes as well.

```scala sc-name:MyType
//{
import scair.ir.TypeAttribute
import scair.clair.*
//}
final case class MyType()
  extends DerivedAttribute["mydialect.type"]
  with TypeAttribute
  derives AttrDefs
```

Type attributes are printed in the IR type position:

```mlir
%0 : !mydialect.type
```

[DerivedAttribute] is the typed base for attributes whose IR name and parameters are provided by a derived companion. 

[derives AttrDefs] generates the glue code needed for printing/parsing and parameter handling.

### Data Attributes

Data attributes store constant compile-time data, such as numbers or structured constants.

ScaIR provides many built-in examples (e.g. `IntData`, `FloatData`). You can define your own:

```scala sc-name:RangeAttr
//{
import scair.dialects.builtin.IntData
import scair.ir.DataAttribute
//}
case class RangeAttr(min: IntData, max: IntData)
  extends DataAttribute[(IntData, IntData)]("mydialect.range", (min, max))
```

Use data attributes for:

* constants
* annotations
* configuration metadata

### Parametrized Attributes

Parametrized attributes are composed of other attributes.

```scala sc-name:FunctionType
//{
import scair.dialects.builtin.ArrayAttribute
import scair.clair.macros.*
import scair.ir.*
//}
final case class FunctionType(
    inputs: ArrayAttribute[Attribute],
    outputs: ArrayAttribute[Attribute],
) extends ParametrizedAttribute
    with TypeAttribute:

  override def name: String = "builtin.function_type"

  override def parameters: Seq[Attribute] =
    Seq(inputs, outputs)
```

These are ideal for:

* function types
* container types
* composite metadata

## Defining Operations

Operations represent units of computation in the IR.

Every Operation has:

* a name
* results
* operands
* successors
* regions
* properties
* attributes

### Typed Operations and the DerivedOperationCompanion

As with attributes, ScaIR defines operations using a typed definition plus a derived companion. Operations are defined as strongly typed Scala case classes.

Each Operation definition consists of two parts:

- **`DerivedOperation`**: defines the typed shape of the Operation (its name, operands, results, regions, and verification logic).
- **`DerivedOperationCompanion`**: connects the typed Scala definition to the generic IR, providing construction, parsing, and printing support.

Together, these two parts bridge the typed Scala API and the generic IR representation used by parsers, printers, and transformation passes.

In most cases, the companion is derived automatically using macros:

```scala sc:nocompile
case class Add(...) 
  extends DerivedOperation["mydialect.add"]
  derives OpDefs
```

This derived companion plays the same role as MLIR’s TableGen-generated boilerplate, but without a separate code-generation step.

### A Simple Operation

```scala sc-name:Add
//{
import scair.ir.*
import scair.dialects.builtin.*
import scair.clair.*
//}
case class Add(
  lhs: Operand[IntegerType],
  rhs: Operand[IntegerType],
  res: Result[IntegerType]
) extends DerivedOperation["mydialect.add"]
  derives OpDefs
```

This defines an operation printed as:

```mlir
%r = "mydialect.add"(%a, %b) : (i32, i32) -> i32
```

### Operations with Regions

Operations may contain regions, which define nested scopes.

```scala sc-name:MyIf
//{
import scair.ir.*
import scair.clair.*
import scair.dialects.builtin.*
//}
case class MyIf(
  cond: Operand[IntegerType],
  thenRegion: Region,
  elseRegion: Region
) extends DerivedOperation["mydialect.if"]
  derives OpDefs
```

Regions are commonly used for control flow and loops.

### Traits

Traits in ScaIR are simply Scala traits. Most operation traits extend `Operation` directly. When an operation mixes in such a trait, the operation itself becomes an instance of that trait. This allows trait implementations to directly access operation properties such as operands, results, and the containing block via `this`.

Traits are used to attach semantics or constraints (including structural properties and shared behavior) to operations and may optionally participate in operation verification.

Common examples:

* [NoMemoryEffect]
* [IsTerminator]
* [IsolatedFromAbove]

```scala sc-name:PureOp
//{
import scair.ir.*
import scair.clair.*
import scair.dialects.builtin.*
//}
case class PureOp(
  res: Result[IntegerType]
) extends DerivedOperation["mydialect.pure"]
  with NoMemoryEffect
  derives OpDefs
```

Example trait Implementation:

```scala
//{
import scair.ir.Operation
import scair.utils.*
//}
trait IsTerminator extends Operation:

  override def traitVerify(): OK[Operation] =
    val verified =
      this.containerBlock match
        case Some(b) =>
          if this ne b.operations.last then
            Err(
              s"Operation '$name' marked as a terminator, but is not the last operation within its container block"
            )
          else OK(this)
        case None =>
          Err(
            s"Operation '$name' marked as a terminator, but is not contained in any block."
          )

    verified.flatMap(_ => super.traitVerify())
```

Traits are commonly used by [transformations](transformations.md) and verification passes.

### Verification

Operations can define a `verify()` method to enforce invariants:

```scala
///{
import scair.ir.*
import scair.clair.*
import scair.utils.*
///}
case class ExampleOp(lhs: Operand[Attribute], rhs: Operand[Attribute]) extends DerivedOperation["example"] derives OpDefs:
  override def verify() =
    if lhs.typ == rhs.typ then OK(this)
    else Err("type mismatch")
```

Verification is run automatically during parsing and transformation passes. Verification combines generic IR checks with operation- and trait-specific constraints.

### Constraints

Hand-written `verify()` is the escape hatch. Most invariants are better stated
declaratively, as a *constraint* attached to an operand or result with `!>`:

```scala
///{
import scair.ir.*
import scair.clair.*
import scair.constraints.*
import scair.dialects.builtin.*
///}
val i32 = IntegerType(IntData(32), Signless)

case class MulI(
    lhs: Operand[IntegerType !> EqAttr[i32.type]],
    rhs: Operand[IntegerType !> EqAttr[i32.type]],
    result: Result[IntegerType],
) extends DerivedOperation["example.muli"] derives OpDefs
```

Constraints are types, and they are interpreted entirely at compile time: the
macro that derives `OpDefs` compiles them into straight-line code inside the
operation's verifier. Nothing about the constraint survives into the generated
program -- there is no constraint object, no context and no lookup table at run
time.

The algebra:

| Constraint | Holds when |
|---|---|
| `AnyAttr` | always |
| `Base[A]` | the attribute is an `A` |
| `EqAttr[a.type]` | the attribute equals `a` (which must be a stable `val`) |
| `Var["T"]` | the attribute equals whatever else `"T"` matched |
| `Param[A, (C1, ..., Cn)]` | it is an `A` whose parameters satisfy `C1`..`Cn` |
| `Msg["...", C]` | `C` holds; otherwise report the given message |
| `C1 && C2` | both hold |
| `C1 \|\| C2` | either holds |

#### Tying components together

`Var` is the interesting one. Its first occurrence binds, later ones must agree,
so sharing a variable between operands and results is MLIR's
`SameOperandsAndResultType`:

```scala
///{
import scair.ir.*
import scair.clair.*
import scair.constraints.*
import scair.dialects.builtin.*
///}
type T = Var["T"]

case class AddF(
    lhs: Operand[Attribute !> (Base[FloatType] && T)],
    rhs: Operand[Attribute !> T],
    result: Result[Attribute !> T],
) extends DerivedOperation["example.addf"] derives OpDefs
```

Sharing a variable costs nothing: the generated check for `rhs` is a direct
comparison against `lhs.typ`, because the compiler knows where the variable was
bound.

#### Inferred result types

A result whose type is determined by a constraint variable need not appear in
the assembly format at all -- it is inferred while parsing:

```scala
///{
import scair.ir.*
import scair.clair.*
import scair.constraints.*
import scair.dialects.builtin.*
type T = Var["T"]
///}
case class NegF(
    operand: Operand[Attribute !> T],
    result: Result[Attribute !> T],
) extends DerivedOperation["example.negf"]
    with AssemblyFormat["$operand attr-dict `:` type($operand)"]
    derives OpDefs
```

If a result's type is neither spelled out in the format nor inferable, that is a
compile error naming the result, rather than a failure at parse time.

#### Reusing and extending constraints

A constraint you use often is just a type alias, and one that takes parameters
is just a parameterised alias:

```scala
///{
import scair.ir.*
import scair.constraints.*
import scair.dialects.builtin.*
///}
type AnyFloat = Base[Float16Type] || Base[Float32Type] || Base[Float64Type]
type SignlessInt = Param[IntegerType, (AnyAttr, EqAttr[Signless.type])]
```

Constraints are types, and each one -- built-in or not -- is a trait plus a
companion object carrying a `ConstraintGen`, the thing that says how to compile
it. `EqAttr` and `AnyOf` are defined exactly this way, so a constraint you define
downstream is not a second-class citizen: it can bind constraint variables,
override failure messages and take part in inference, because it uses the same
interface the built-ins do.

For a constraint that genuinely cannot be composed -- one that has to *compute*
over the attribute -- write it out:

```scala
///{
import scair.constraints.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.utils.*
import scala.quoted.*
///}
trait Width[N <: Int] extends Constraint

object Width extends ConstraintGen:
  def verify(c: Type[?], attr: Expr[Attribute], path: String)(using
      Quotes, GenCtx, ErrCtx
  ) =
    c match
      case '[Width[n]] =>
        val expected = Expr(Type.valueOfConstant[n].get.asInstanceOf[Int])
        Some('{
          $attr match
            case IntegerType(IntData(bits), _) if bits == BigInt($expected) => OK()
            case a => ${ fail(path, '{ "an integer of width " + $expected }, 'a) }
        })
```

`ConstraintGen` has two more methods, both optional: `infer`, to let the
constraint pin down a result type an assembly format omits, and `bind`, to seed
inference from a value already known to satisfy it. Returning `None` from
`verify` means the constraint needs no code at all -- that is how `AnyAttr` and a
variable's first occurrence stay genuinely free. A constraint built out of other
constraints should extend `CompositeGen` and say only what its `parts` are.

A generator runs at compile time, so it must live in a module compiled *before*
the operations that use it -- the same rule that already applies to macros and
`derives OpDefs`.

## What is a Dialect?

A dialect is a namespace that groups:

* operations
* attributes
* types

Dialects represent a coherent abstraction level in the IR.

Examples:

* `arith` — arithmetic operations
* `scf` — structured control flow

### Declare the Dialect

In ScaIR, dialects are declared using `summonDialect`.

```scala sc:nocompile
val MyDialect = summonDialect[
  // Attributes
  (MyType, FunctionType, RangeAttr),

  // Operations
  (Add, PureOp)
]
```

Calling `summonDialect` constructs a dialect definition, describing its attributes, operations, and associated parsing and printing logic. By itself, however, this does not make the dialect available to any tool or pass.

### Register a Dialect

ScaIR tools typically inherit from `ScairOptBase`, which defines the set of available dialects via the dialects field:

```scala
//{
import scair.tools.opt.*
import scair.tools.ScairToolBase
//}
trait ScairOptBase extends ScairToolBase[ScairOptArgs]:
  override def dialects = scair.dialects.allDialects
```

which defaults to:

```scala sc:nocompile
val allDialects: Seq[Dialect] =
  Seq(
    BuiltinDialect,
    ...,
    MyDialect
  )
```

A dialect becomes usable once it is included in the sequence returned by dialects.

There are two common ways to register a dialect:

* When using ScaIR as a library: Create a custom `Opt` class inheriting from `ScairOptBase` and override `dialects` to include your dialect.

* When working within ScaIR itself: Add the dialect directly to the `allDialects` sequence.

Once a dialect is registered with a tool, the IR parser and printer can recognize:

* attribute names
* operation names
* dialect-specific parsing and printing logic

### How to Connect a Dialect

To make a dialect available:

1. Import the dialect object
2. Ensure it is linked into the binary

```scala sc:nocompile
import scair.dialects.mydialect.*
```

After this, IR containing `"mydialect.*"` operations can be parsed and printed.
