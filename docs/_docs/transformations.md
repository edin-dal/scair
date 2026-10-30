---
title: "Transformations and Pass"
---

[NoMemoryEffect]: scair.ir.NoMemoryEffect
[pattern]: scair.transformations.pattern
[Owner]: scair.transformations.Owner
[AddI]: scair.dialects.arith.AddI
[Constant]: scair.dialects.arith.Constant
[IntegerAttr]: scair.dialects.builtin.IntegerAttr
[IntegerAttr+]: scair.dialects.builtin.IntegerAttr.+
[WalkerPass]: scair.transformations.WalkerPass
[ModulePass]: scair.transformations.ModulePass
[PassCompanion]: scair.transformations.PassCompanion
[PatternRewriteWalker]: scair.transformations.PatternRewriteWalker
[RewritePattern]: scair.transformations.RewritePattern
[GreedyRewritePatternApplier]: scair.transformations.GreedyRewritePatternApplier
[UnregisteredOperation]: scair.ir.UnregisteredOperation



# Transformation and Pass

In this tutorial we will focus on writing optimizing transformations in ScaIR.

## **Constant Folding and Dead Code Elimination**

### **IR Example**

Say we have the following high-level program:
```python
x = 5
y = 5
z = x + y
print(z)
```


We could write this program in its Intermediate Representation (IR) form like so: 
```mlir
%0 = "arith.constant"() <{value = 5}> : () -> i32
%1 = "arith.constant"() <{value = 5}> : () -> i32
%2 = "arith.addi"(%0, %1) : (i32, i32) -> i32
func.call @print(%2) : (i32) -> ()
```


Immediately, we can see that **`x + y`** is an addition of two constants **`5 + 5`**, which can be folded away to just **`10`**, with the IR like so: 
```mlir
%0 = "arith.constant"() <{value = 5}> : () -> i32
%1 = "arith.constant"() <{value = 5}> : () -> i32
%2 = "arith.constant"() <{value = 10}> : () -> i32
func.call @print(%2) : (i32) -> ()
```


Great! Now our addition has been folded away and replaced by a constant! However, what about the first two constants, are they not completely useless now? 

The two constants are indeed useless, as in they have no **`uses`** left since their SSA values are no longer referenced anywhere in the IR. Therefore, they are dead code and can be eliminated!
```mlir
%0 = "arith.constant"() <{value = 10}> : () -> i32
func.call @print(%0) : (i32) -> ()
```

**Great! Now, how can we implement this in ScaIR?**


### **ScaIR Implementation**

Let's start with our IR constructs, which we will borrow from the [Arith](https://mlir.llvm.org/docs/Dialects/ArithOps/) dialect (already defined in ScaIR) to express our basic arithmetic addition, and constants.

@note You might notice that both Operation extend a trait **[NoMemoryEffect]**. We will come back to its exact function shortly.
```scala
//{
import scair.clair.*
import scair.ir.*
import scair.dialects.builtin.*
//}

case class Constant(
    val value: Attribute,
    val result: Result[Attribute]
) extends DerivedOperation["arith.constant"]
  with NoMemoryEffect
  derives OpDefs

case class AddI(
    val lhs: Operand[IntegerType],
    val rhs: Operand[IntegerType],
    val result: Result[IntegerType]
) extends DerivedOperation["arith.addi"]
	with NoMemoryEffect
  derives OpDefs
```


After parsing our original IR, our internal representation of it in ScaIR would look like so:
```scala sc:nocompile
...
Block(
	Constant(c0: IntegerAttr, r0: Result[IntegerAttr])
	Constant(c1: IntegerAttr, r1: Result[IntegerAttr])
	AddI(r0, r1, r2: Result[IntegerAttr])
	...
)
...
```


Now, we can use ScaIR's **[pattern]** API to define a transformation via simple pattern matching:
	`Note`: You may be wondering what the **[Owner]** is in the matcher of the pattern. Each SSA value keeps track of where it was defined. **[Owner]** is a simple extractor object, allowing us to extract the SSA value's defining Operation and pattern match over it.
	
```scala sc-name:addi-fold
//{
import scair.transformations.*
import scair.dialects.arith.*
import scair.dialects.builtin.		*
import scair.ir.*
//}
val AddIFold = pattern {
	case AddI(
		lhs = Owner(Constant(c0: IntegerAttr, _)),
		rhs = Owner(Constant(c1: IntegerAttr, _)),
	) =>
		Constant(c0 + c1, Result(c0.typ))
}
```

In the pattern above, we first match on an **[AddI]** operation, whose operands' defining operations are both **[Constant]** operations containing an integer attribute. A new constant value is the constructed from the addition of the two integer attributes (via and infix **[+][IntegerAttr+]** operator defined for **[IntegerAttr]** class).


Next, let's define our dead code elimination. And here is where our trait **[NoMemoryEffect]** comes in handy. Similar to MLIR, traits are used to further extend Operations with additional semantics, but also allows us to group Operations with the same semantics. **[NoMemoryEffect]** represents a group of Operations which, as you might have guessed, have no effect on memory. 

If an Operation has no effects on memory, then we can safely erase it after making sure that none of its results are used anywhere in the IR: 
```scala sc-name:dce
//{
import scair.transformations.*
import scair.ir.*
//}
val DeadCodeElimination = pattern {
	case op: NoMemoryEffect if op.results.forall(_.uses.isEmpty) =>
		PatternAction.Erase
}
```


Finally, let's package our transformations into a **[WalkerPass]**.
```scala sc-compile-with:addi-fold,dce
//{
import scair.transformations.*
import scair.MLContext
//}
final case class SampleConstantFoldingAndDCE() extends WalkerPass
    derives PassCompanion:

  override def walker(using MLContext) = PatternRewriteWalker(
    GreedyRewritePatternApplier(
	    Seq(
		    AddIFold,
		    DeadCodeElimination
	    )
	)
  )
```

[WalkerPass] is a kind of IR **[ModulePass]** that walks over all operations in the IR via **[PatternRewriteWalker]**, and applies a given pattern. 
In this case a **[GreedyRewritePatternApplier]**, which itself is a pattern that takes an Operation and applies all given patterns greedily over it, until a change is seen, or all patterns are tried.  

Deriving **[PassCompanion]** uses the Scala class name as the pass name: here `SampleConstantFoldingAndDCE`. Mix in `PassName["..."]` to override it, e.g. `CommonSubexpressionElimination` uses `PassName["CSE"]` and `LoopInvariantCodeMotion` uses `PassName["LICM"]`.

### **Pass options**

A pass's options are the fields of its case class. Give a field a default value to make the option optional:

```scala sc-compile-with:addi-fold,dce
//{
import scair.transformations.*
import scair.MLContext
//}
enum Order:
  case TopDown, BottomUp

final case class SampleOptionsPass(
    iterations: Int,                 // required
    order: Order = Order.TopDown,
    verbose: Boolean = false,
    skip: Seq[String] = Seq(),
) extends WalkerPass derives PassCompanion:

  override def walker(using MLContext) = PatternRewriteWalker(
    GreedyRewritePatternApplier(Seq(AddIFold, DeadCodeElimination))
  )
```

On the command line, the options follow the pass name in braces, using the exact Scala field names and separated by spaces:

```bash
scair-opt -p 'SampleOptionsPass{iterations=3 order=BottomUp verbose skip=a,b}'
```

- Each option may be specified at most once per pass.
- A list option takes comma-separated values.
- A `Boolean` option without a value is read as `true`: `verbose` means `verbose=true`. Other types require a value.
- Values that contain spaces, commas or braces go in double quotes.
- Enum options take their exact Scala case names.

Supported option types are `Int`, `Long`, `Double`, `String`, `Boolean`, simple enums, and `Option`, `Seq` and `List` of those. To support another type, define a `given PassOption[T]`.

Register pass types with `summonPasses[(FirstPass, SecondPass)]`, which returns their derived companions for a tool's `passes` list.

## **Footnotes**


We can also pattern match and transform over Operations not defined within ScaIR. **[UnregisteredOperation]** class is used to instantiate all such Operations.

Consider the following IR:
```mlir
%0 = "some.op-that-does-not-exist"() <{value = 10}> : () -> i32
func.call @print(%0) : (i32) -> ()
```
As you might guess, given the context of this example, `"some.op-that-does-not-exist"` is an operation that does not exist within ScaIR :o 

However, we can still write a pattern over it ( :D ), like so: 

```scala
//{
import scair.dialects.builtin.*
import scair.dialects.arith.*
import scair.transformations.*
import scair.ir.*
//}
val ten = IntegerAttr(IntData(10), IntegerType(IntData(32), Signless)) // 10 : i32

val UnregPat = pattern {
  case op: UnregisteredOperation if op.name == "some.op-that-does-not-exist" =>
    val const = op.properties.getOrElse("value", ten).asInstanceOf[IntegerAttr]
    Constant(const, Result(const))
}
```

to produce the following IR after transformation, effectively translating an unknown operation into a known constant operation:
```mlir
%0 = "arith.constant"() <{value = 10}> : () -> i32
func.call @print(%0) : (i32) -> ()
```