---
title: "Package a Dialect and Connect Passes"
---

This tutorial explains how to package attributes and operations into a **Dialect**, and how to define and connect **Passes** to the ScaIR compilation pipeline.

---

## 1. What is a Dialect?

A **Dialect** is a namespace that groups:

* operations
* attributes
* types

Dialects represent a coherent abstraction level in the IR.

Examples:

* `arith` — arithmetic operations
* `scf` — structured control flow
* `dlam` — dependent typing

---

## 2. Registering a Dialect

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

---

## 3. How to Connect a Dialect

To make a dialect available:

1. Import the dialect object
2. Ensure it is linked into the binary

```scala
import scair.dialects.mydialect.*
```

After this, IR containing `"mydialect.*"` operations can be parsed and printed.

---

## 4. Defining Passes

A **Pass** performs analysis or transformation over the IR.

### 4.1 ModulePass

```scala
final class MyAnalysisPass(ctx: MLContext)
  extends ModulePass(ctx):

  override val name = "my-analysis"

  override def transform(op: Operation): Operation = {
    // inspect or modify IR
    op
  }
```

`ModulePass` runs once over the entire module.

---

### 4.2 WalkerPass

`WalkerPass` traverses the IR and applies rewrite patterns.

```scala
final class MyRewritePass(ctx: MLContext)
  extends WalkerPass(ctx):

  override val name = "my-rewrite"

  override val walker = PatternRewriteWalker(
    GreedyRewritePatternApplier(Seq(MyPattern))
  )
```

Walker passes are ideal for:

* optimizations
* canonicalization
* lowering

---

## 5. Writing Rewrite Patterns

Patterns match operations and produce replacements.

```scala
val MyPattern = pattern {
  case Add(lhs, rhs, _) =>
    Add(lhs, rhs, Result(lhs.typ))
}
```

Patterns can:

* replace operations
* erase operations
* rewrite operands and results

---

## 6. Integrating Passes into the Pipeline

Passes are invoked using `scair-opt`:

```bash
scair-opt -p=my-rewrite,my-analysis input.mlir
```

Passes run in the order specified.
