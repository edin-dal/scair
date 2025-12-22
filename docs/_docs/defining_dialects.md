---
title: "Defining a Dialect"
---

[Attribute]: scair.ir.Attibute
[Operation]: scair.ir.Operation
[Traits]: sacir.ir.Traits
[DerivedAttribute]: scair.clair.Traits
[DerivedOperation]: scair.clair.Traits
[derives DerivedOperationCompanion]: scair.clair.Traits
[derives DerivedAttributeCompanion]: scair.clair.Traits

# Defining a Dialect
This tutorial explains how to define new attributes and operations in ScaIR and how to package these into a dialect.

## Defining Attributes

Attributes represent compile-time information in the IR. They are immutable and may appear:

* as SSA types
* as constant values
* as metadata attached to operations

In ScaIR, all attributes extend the base [Attribute] hierarchy.

### Attributes vs Types

In MLIR and ScaIR types are a specialized kind of attribute.

Attributes represent general compile-time information, while type attributes are used exclusively to describe the types of SSA values. Every SSA value must have exactly one [TypeAttribute].

This distinction is reflected in the IR syntax:

- `#dialect.attr<...>` — general attributes
- `!dialect.type<...>` — type attributes

Although both are implemented as attributes internally, only type attributes may appear in SSA type positions.

In ScaIR, this distinction is expressed explicitly in Scala: type attributes extend TypeAttribute, while other attributes do not.

### Type Attributes

`TypeAttribute` describes the types of SSA values. Every SSA value must have exactly one type attribute.

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
case class RangeAttr(min: Int, max: Int)
  extends DataAttribute[(Int, Int)]("mydialect.range", (min, max))
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
* operands
* results
* optional regions
* optional traits

### Typed Operations and the DerivedOperationCompanion

In ScaIR, Operations are defined as strongly typed Scala `case class`es. This replaces MLIR’s TableGen-generated C++ with ordinary Scala code that is checked at compile time.

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

[DerivedOperation] is the typed base for Operations and fixes the operations IR name.

[derives DerivedOperationCompanion] generates the bridge between the typed case class and the generic IR (construction, printing/parsing, and additional verification constraints).

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

[Traits] attach semantic guarantees to Operations.

Common examples:

* `NoMemoryEffect`
* `IsTerminator`
* `IsolatedFromAbove`

```scala
case class PureOp(
  res: Result[IntegerType]
) extends DerivedOperation["mydialect.pure", PureOp]
  with NoMemoryEffect
  derives DerivedOperationCompanion
```

Traits are used by transformations and verification passes.

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

### Registering a Dialect

In ScaIR, dialects are registered declaratively using `summonDialect`.

```scala
val MyDialect = summonDialect[
  // Attributes
  (MyType, VectorType, RangeAttr),

  // Operations
  (Add, PureOp)
]
```

Once registered, the IR parser can recognize:

* attribute names
* operation names
* printing and parsing logic

### How to Connect a Dialect

To make a dialect available:

1. Import the dialect object
2. Ensure it is linked into the binary

```scala
import scair.dialects.mydialect.*
```

After this, IR containing `"mydialect.*"` operations can be parsed and printed.
