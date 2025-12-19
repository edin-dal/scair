---
title: "Define Attributes and Operations"
---

# Attributes and Operations

This tutorial explains how to define new attributes and operations in ScaIR. These are the fundamental building blocks used to model computation in the IR and closely follow the MLIR design.

## 1. Defining Attributes

Attributes represent **compile-time information** in the IR. They are immutable and may appear:

* as SSA types
* as constant values
* as metadata attached to operations

In ScaIR, all attributes extend the base `Attribute` hierarchy.

### 1.1 Base Attributes

Base attributes represent atomic data values. Typical examples are integers, floats, and strings.

```scala
final case class MyIntAttr(value: Int)
  extends DataAttribute[Int]("mydialect.int", value)
```

Use base attributes when the attribute:

* wraps a single value
* does not reference other attributes

---

### 1.2 Type Attributes

Type attributes describe the **types of SSA values**. Every SSA value must have exactly one type attribute.

```scala
final case class MyType()
  extends TypeAttribute
  with DerivedAttribute["mydialect.type", MyType]
  derives DerivedAttributeCompanion
```

Type attributes are printed in the IR type position:

```mlir
%0 : !mydialect.type
```

---

### 1.3 Data Attributes

Data attributes store constant compile-time data, such as numbers or structured constants.

ScaIR provides many built-in examples (e.g. `IntegerAttr`). You can define your own:

```scala
final case class RangeAttr(min: Int, max: Int)
  extends DataAttribute[(Int, Int)]("mydialect.range", (min, max))
```

Use data attributes for:

* constants
* annotations
* configuration metadata

---

### 1.4 Parametrized Attributes

Parametrized attributes are composed of **other attributes**.

```scala
final case class VectorType(
  length: Attribute,
  elementType: Attribute
) extends ParametrizedAttribute
    with TypeAttribute:

  override def name = "mydialect.vector"
  override def parameters = Seq(length, elementType)
```

These are ideal for:

* container types
* dependent or structured types
* composite metadata

---

## 2. Defining Operations

Operations represent **units of computation** in the IR.

Every operation has:

* a name
* operands
* results
* optional regions
* optional traits

### 2.1 A Simple Operation

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

---

### 2.2 Operations with Regions

Operations may contain **regions**, which define nested scopes.

```scala
case class MyIf(
  cond: Operand[IntegerType],
  thenRegion: Region,
  elseRegion: Region
) extends DerivedOperation["mydialect.if", MyIf]
  derives DerivedOperationCompanion
```

Regions are commonly used for:

* control flow
* lambdas
* loops

---

### 2.3 Traits

Traits attach semantic guarantees to operations.

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

Traits are heavily used by transformations and verification passes.

---

## 3. Verification

Operations can define a `verify()` method to enforce invariants:

```scala
override def verify() =
  if lhs.typ == rhs.typ then Right(this)
  else Left("type mismatch")
```

Verification is run automatically during parsing and transformation passes.
