In this tutorial we will focus on writing optimizing transformations in ScaIR. 
- [Constant Folding and Dead Code Elimination](#constant-folding-and-dead-code-elimination)
	- [Example](#example)
	- [ScaIR Implementation](#scair-implementation)
	- 
# Constant Folding and Dead Code Elimination

### Example

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


### ScaIR Implementation

Let's start with our IR constructs, which we will borrow from the [Arith](https://mlir.llvm.org/docs/Dialects/ArithOps/) dialect (already defined in ScaIR) to express our basic arithmetic addition, and constants.
	`Note`: You might notice that both Operation extend a trait **`NoMemoryEffect`**. We will come back to its exact function shortly.
```scala
case class Constant(
    val value: Attribute,
    val result: Result[Attribute]
) extends DerivedOperation["arith.constant", Constant]
	with NoMemoryEffect

case class AddI(
    val lhs: Operand[AnyIntegerType],
    val rhs: Operand[AnyIntegerType],
    val result: Result[AnyIntegerType]
) extends DerivedOperation["arith.addi", AddI]
	with NoMemoryEffect
```


After parsing our original IR, our internal representation of it in ScaIR would look like so:
```scala
...
Block(
	Constant(c0: IntegerAttr, r0: Result[IntegerAttr])
	Constant(c1: IntegerAttr, r1: Result[IntegerAttr])
	AddI(r0, r1, r2: Result[IntegerAttr])
	...
)
...
```


Now, we can use ScaIR's **`pattern`** API to define a transformation via simple pattern matching:
	`Note`: You may be wondering what the **`Owner`** is in the matcher of the pattern. Each SSA value keeps track of where it was defined. **`Owner`** is a simple extractor object, allowing us to extract the SSA value's defining Operation and pattern match over it.
	
```scala
val AddIfold = pattern {
	case AddI(
		Owner(Constant(c0: IntegerAttr, _)),
		Owner(Constant(c1: IntegerAttr, _)),
		_
	) =>
		Constant(c0 + c1, Result(c0.typ))
}
```

In the pattern above, we first match on an **`AddI`** operation, whose operands' defining operations are both **`Constant`** operations containing an integer attribute. A new constant value is the constructed from the addition of the two integer attributes (via and infix **`+`** operator defined for **`IntegerAttr`** class).


Next, let's define our dead code elimination. And here is where our trait **`NoMemoryEffect`** comes in handy. Similar to MLIR, traits are used to further extend Operations with additional semantics, but also allows us to group Operations with the same semantics. **`NoMemoryEffect`** represents a group of Operations which, as you might have guessed, have no effect on memory. 

If an Operation has no effects on memory, then we can safely erase it after making sure that none of its results are used anywhere in the IR: 
```scala
val DeadCodeElimination = pattern {
	case op: NoMemoryEffect if op.results.forall(_.uses.length == 0) =>
		PatternAction.Erase
}
```


Finally, let's package our transformations into a **`WalkerPass`**.
```scala
final class SampleConstantFoldingAndDCE(ctx: MLContext) extends WalkerPass(ctx):
  override val name = "sample-constant-folding-and-dce"

  override final val walker = PatternRewriteWalker(
    GreedyRewritePatternApplier(
	    Seq(
		    AddIFold,
		    DeadCodeElimination
	    )
	)
  )
```

WalkerPass is a kind of IR **`Pass`** that walks over all operations in the IR via **`PatternRewriteWalker`**, and applies a given pattern. 
In this case a **`GreedyRewritePatternApplier`**, which itself is a pattern that takes an Operation and applies all given patterns greedily over it, until a change is seen, or all patterns are tried.  