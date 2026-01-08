---
title: "Defining a Dialect"
---

[Attribute]: scair.ir.Attribute
[TypeAttribute]: scair.ir.TypeAttribute
[Operation]: scair.ir.Operation
[NoMemoryEffect]: scair.ir.NoMemoryEffect
[IsTerminator]: scair.ir.IsTerminator
[IsolatedFromAbove]: scair.ir.IsolatedFromAbove
[DerivedAttribute]: scair.clair.macros.DerivedAttribute
[DerivedOperation]: scair.clair.macros.DerivedOperation
[derives DerivedOperationCompanion]: scair.clair.macros.DerivedOperationCompanion
[derives DerivedAttributeCompanion]: scair.clair.macros.DerivedAttributeCompanion

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

```scala
final case class MyType()
  extends DerivedAttribute["mydialect.type", MyType]
  with TypeAttribute
  derives DerivedAttributeCompanion
```

Type attributes are printed in the IR type position:

```mlir
%0 : !mydialect.type
```

[DerivedAttribute] is the typed base for attributes whose IR name and parameters are provided by a derived companion. 

[derives DerivedAttributeCompanion] generates the glue code needed for printing/parsing and parameter handling.

### Data Attributes

Data attributes store constant compile-time data, such as numbers or structured constants.

ScaIR provides many built-in examples (e.g. `IntData`, `FloatData`). You can define your own:

```scala
case class RangeAttr(min: IntData, max: IntData)
  extends DataAttribute[(IntData, IntData)]("mydialect.range", (min, max))
```

Use data attributes for:

* constants
* annotations
* configuration metadata

### Parametrized Attributes

Parametrized attributes are composed of other attributes.

```scala
final case class FunctionType(
    inputs: Seq[Attribute],
    outputs: Seq[Attribute],
) extends ParametrizedAttribute
    with TypeAttribute:

  override def name: String = "builtin.function_type"

  override def parameters: Seq[Attribute | Seq[Attribute]] =
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

```scala
case class Add(...) 
  extends DerivedOperation["mydialect.add", Add]
  derives DerivedOperationCompanion
```

This derived companion plays the same role as MLIR’s TableGen-generated boilerplate, but without a separate code-generation step.

### A Simple Operation

```scala
case class Add(
  lhs: Operand[IntegerType],
  rhs: Operand[IntegerType],
  res: Result[IntegerType]
) extends DerivedOperation["mydialect.add", Add]
  derives DerivedOperationCompanion
```

This defines an operation printed as:

```mlir
%r = "mydialect.add"(%a, %b) : (i32, i32) -> i32
```

### Operations with Regions

Operations may contain regions, which define nested scopes.

```scala
case class MyIf(
  cond: Operand[IntegerType],
  thenRegion: Region,
  elseRegion: Region
) extends DerivedOperation["mydialect.if", MyIf]
  derives DerivedOperationCompanion
```

Regions are commonly used for control flow and loops.

### Traits

Traits in ScaIR are simply Scala traits. Most operation traits extend `Operation` directly. When an operation mixes in such a trait, the operation itself becomes an instance of that trait. This allows trait implementations to directly access operation properties such as operands, results, and the containing block via `this`.

Traits are used to attach semantics or constraints (including structural properties and shared behavior) to operations and may optionally participate in operation verification.

Common examples:

* [NoMemoryEffect]
* [IsTerminator]
* [IsolatedFromAbove]

```scala
case class PureOp(
  res: Result[IntegerType]
) extends DerivedOperation["mydialect.pure", PureOp]
  with NoMemoryEffect
  derives DerivedOperationCompanion
```

Example trait Implementation:

```scala
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
override def verify() =
  if lhs.typ == rhs.typ then Right(this)
  else Left("type mismatch")
```

Verification is run automatically during parsing and transformation passes. Verification combines generic IR checks with operation- and trait-specific constraints.

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

```scala
val MyDialect = summonDialect[
  // Attributes
  (MyType, VectorType, RangeAttr),

  // Operations
  (Add, PureOp)
]
```

Calling `summonDialect` constructs a dialect definition, describing its attributes, operations, and associated parsing and printing logic. By itself, however, this does not make the dialect available to any tool or pass.

### Register a Dialect

ScaIR tools typically inherit from `ScairOptBase`, which defines the set of available dialects via the dialects field:

```scala
trait ScairOptBase extends ScairToolBase[ScairOptArgs]:
  override def dialects = scair.dialects.allDialects
```

which defaults to:

```scala
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

```scala
import scair.dialects.mydialect.*
```

After this, IR containing `"mydialect.*"` operations can be parsed and printed.
