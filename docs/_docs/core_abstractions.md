---
title: "Core Abstractions"
---

# Core Abstractions

In this tutorial we will focus on the core compilation abstraction used in MLIR. 

**`Note:`** The objective of this tutorial is to gain basic intuitions about the SSA form, Regions as well as the core compilation concepts introduced by MLIR. The tutorial will include brief informal definitions of each concept, complemented by some basic examples in the IR. As such, this tutorial does not serve as a comprehensive guide through each concept. 

For more formal introductions please refer to:
- TODO

## **Static-Single Assignment (SSA)**
MLIR's Intermediate Representation (IR) maintains the SSA form. 

At its core, SSA form represents computations through the assignment and passing of **values** around the program.

In the SSA form, *each **`Value`** produced by some operation is assigned exactly once* within the same scope, and **can not** be mutated during the course of the program represented in the IR. 

However, one **value** can be used many times in the IR, which makes up the **def-use** and **use-def** chains, where each value tracks where it is used, and it's defining operation. 

Let's say we have a simple integer addition operation in Scala, where values `a` and `b` are added to create a new value `c`:
```scala sc:nocompile
...
val c = a + b
...
```
which we can express in the IR like so:
```mlir
...
%c = "arith.addi"(%a, %b) : (i32, i32) -> i32
...
```
We can see a new name `arith.addi` which uniquely identifies an addition operation in MLIR, as well as the `i32` types, which define the types for **`operands`** and **`results`** for the operation. We will explain these shortly, however for let's focus on the values first.

We can see that our values `a`, `b` and `c` in the top program became **`values`** `%a`, `%b` and `%c` in the SSA IR on the bottom. Intuitively, the value `%c` in the IR represents the value `c` in the program, and as such will be used in any subsequent IR operation that referred to `c` in the original program. 

Now, you might be thinking "well, yeah... ?". And you are right, this is a completely trivial example.

However, in many high-level languages you are free to mutate previously defined variables. In Scala, if `a` was defined as a variable, we could re-assign to it:
```python
...
a = a + b
print(a)
```

In the SSA form, however, this direct mutation is illegal. What would happen instead, is we would create a new value, which would then be used to represent `a` in the remainder of the IR, like so:
```mlir
...
%a1 = "arith.addi"(%a0, %b) : (i32, i32) -> i32
func.call @print(%a1) : (i32) -> ()
```
Notice here, that the old reference to `a` is written as `%a0`, and the newly created value `%a1` is used in the `print` function.



## **Regions**
Another core part of the MLIR IR are **`Regions`**, which are used to *express localized scoping in the IR*. 

Let's look at a simple program here:
```scala sc:nocompile
...
val c5 = 5
val c10 = 10

val x = if condition then {
			val a = c5 + c5
			return a
		} else {
			val a = c10 + c10
			return a
		} 
```
The two branches of the `if` expression contained within the curly brackets `{ ... }` define localized scope.
- `val a` is defined by the addition of values defined outside of the immediate scope of the branch, rather its parent scope. 
- Additionally, we can maintain two different `val a` definitions as the `then` branch has no access to the `else` branch, and vice-versa. 
- Lastly, the `return` statement at the end of each branch returns `a` to be assigned to `val x`, as each `a` is only within the scope of its respective branch. 


Similarly in the IR, **`regions`** are used to define localized scope within the different branches of the **`if`** operation: 
```mlir
...
%const5 = "arith.constant"() <{value = 5}> : () -> i64
%const10 = "arith.constant"() <{value = 10}> : () -> i64

%0 = "scf.if"(%condition) ({
  %1 = "arith.addi"(%const5, %const5) : (i64, i64) -> i64
  "scf.yield"(%1) : (i64) -> ()
}, {
  %1 = "arith.addi"(%const10, %const10) : (i64, i64) -> i64
  "scf.yield"(%1) : (i64) -> ()
}) : (i1) -> i64
```
The two regions within the `if` operation are delimited by the curly brackets `{ ... }`, and represent the two branches of the `if` operation. 

The `yield` operation at the end of each region is the returning operation, and its operand `%1` the value of the resulting computation inside the region.

Like the branches in the Scala program, Regions in MLIR IR maintain 3 properties:
- Regions have access to its parent scope.
- Regions do not share any definitions, their scope is localized.
- The outside scope cannot access any values defined within the Region.

### **Blocks**
Each Region in the IR contains a list of **Blocks**.

Blocks represent basic units of control flow within the IR.

Each **Block** contains a list of 0+ **block arguments**, that are explicitly passed into the block, and can be used within the block.

Here is an example of block usage in the IR, **^entry**, **^loop**, **^body**, **^exit** are blocks:
```mlir
...
^entry:
  %n = arith.constant 10 : i32
  %c1 = arith.constant 1 : i32
  %c0 = arith.constant 0 : i32
  cf.br ^loop(%c0 : i32)

^loop(%i : i32):
  %cond = arith.cmpi slt, %i, %n : i32
  cf.cond_br %cond, ^body, ^exit

^body:
  %i_next = arith.addi %i, %c1 : i32
  cf.br ^loop(%i_next : i32)

^exit:
  func.return
...
```


## **Dialect**
Finally, we get to the core abstractions! Let's start simple: **`Dialect`**.

Dialects are simply namespaces for **`attributes`** and **`operations`**. Conceptually, dialects represent a group of attributes and operations of a certain abstraction.
Generally, attribute and operation names are prefaced with the name of the dialect they come from. 

We have seen some examples in the IR already: 
```mlir
%const5 = "arith.constant"() <{value = 5}> : () -> i64
%a1 = "arith.addi"(%a0, %b) : (i32, i32) -> i32
```
Where `arith` is the dialect containing the `addi` and `constant` operation, and represents abstractions for basic arithmetic. 

Or, the `scf` dialect, containing abstractions for expressions structured control flow in the IR:
```mlir
...
%0 = "scf.if"(%condition) ({
  ...
  "scf.yield"(%1) : (i64) -> ()
```

## **Attribute**
We have already seen a number of different attributes in previous examples!
Generally, **`Attributes`** define any and all compile time information in the IR. Crucially, each SSA value defined by an operation must have an associated type attribute; in the example below, `%const5` has the type `i64`.
```mlir
...
%const5 = "arith.constant"() <{value = 5}> : () -> i64
...
```
This operation contains two attributes: **`5`** - an **`Integer`** constant,  and **`i64`** - an **`Integer`** type for a signless 64-bit integer.  

You might notice, that in the previous section I also mentioned that attributes are generally prefaced by its dialect name. And this is indeed *generally* true, however, there is an exception made for the **`builtin`** dialect, which contains both the integer attribute **`5`** and integer type **`i64`**.

Here is a more general example of a complex number attribute from the `complex` dialect: 
```mlir
...
%complx = "arith.constant"() <{value = #complex.number<:f64 1.0, 0.0>}> : () -> complex<f64>
...
```
Here `#complex.number<:f64 1.0, 0.0>` is an attribute representing a complex number of `f64` type.

## **Operation**
**`Operations`** represent abstract units of computation.

Each Operation consists of:
- **`name`** -> a uniquely identifiable name within the IR, and consists of a dialect name followed by the name of the operation within the dialect, like: `"arith.constant"`
- **`operands`** -> a list of used SSA values by the given operation.
- **`results`** -> a list of SSA values produced by the given operation.
- **`regions`** -> a list of Regions contained within the given operation.
- **`successors`** -> a list of blocks to which the operation can pass control-flow.
- **`properties`** -> attribute meta-data semantically attached to an operation; if an operation's definition contains a property it must be shown in the IR, unless a default value is given. Properties are encoded as a dictionary mapping the property name to an attribute, like so:  `<{value = 5}>`.
- **`attributes`** -> a dictionary of additional attribute meta-data; these mappings can be changed and are not required at any point. 

There are two main benefits that derive from this generic representation:
- It allows for expressing a wide range of different abstractions as an Operation.
- Since every operation is a collection of all of these fields, one can reason generically about each operation. This is particularly powerful during generic optimizing transformations (eg. CSE and DCE).

The generic IR form of an operation is the following:
**`results`** `=` `"` **`name`** `"` `(` **`operands`** `)` `[`**`successors`**`]` `<{`**`properties`**`}>` `(`**`regions`**`)` `{`**`attributes`**`}` `:` `(`**`operand_types`**`)` `->` `(`**`result_types`**`)` 

and an example of an operation that contains all of these would be:
```mlir
^bb0():
^bb1():
...
%2, %3 = "sample.op"(%0, %1) [^bb0, ^bb1] <{p1 = 1, p2 = i32}> ({
	...
}, {
	...
}) {hello = "hello", world = "world"} : (i32, f16) -> (i1, i1)
```
