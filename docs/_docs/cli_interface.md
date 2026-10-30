---
title: "CLI Interface"
---

# **CLI Interface**

We implement the `scair-opt` CLI interface for ScaIR. 

`scair-opt` allows the end-user run the framework end-to-end, applying a specified list of passes to the IR contained within the given file, and more! (but mainly this :D)

**Let's install it:**
```bash
~/scair$ ./mill install
```
Great! Now you should be able to simply run `scair-opt`.  

**Let's run it:**

When no arguments are specified, `scair-opt` prompts you to write the IR manually. If you click **`Ctrl + D`**, it will parse the input, and print it back out. 
```bash
~/scair$ scair-opt
```
If no input is given, it will return an empty program:
```bash
builtin.module {
^bb0:
}
```

However, if you provide it with an input file, as well as a list of passes, the framework will pass the IR, apply the passes, then print the transformed IR back out!

Let's run our example pass from [Transformations](transformations.md) tutorial.

To refresh ourselves, let's inspect the IR:
```bash
~/scair$ cat sample.mlir
%0 = "arith.constant"() <{value = 5}> : () -> i32
%1 = "arith.constant"() <{value = 5}> : () -> i32
%2 = "arith.addi"(%0, %1) : (i32, i32) -> i32
func.call @print(%2) : (i32) -> ()
```

Now, let's apply our `SampleConstantFoldingAndDCE` pass:

```bash
~/scair$ scair-opt sample.mlir --passes SampleConstantFoldingAndDCE
builtin.module {
  %0 = "arith.constant"() <{value = 10}> : () -> i32
  func.call @print(%0) : (i32) -> ()
}
```

Additionally, there are a number of flags currently available for `scair-opt`:
- **`<file>`** -> the '.mlir' input file.

- **`--allow-unregistered-dialect`** -> allows operations not registered in ScaIR to be parsed. 

- **`--skip-verify `** -> skips the verification stage of the IR.

- **`--split-input-file`** -> `Note`: for testing purposes only - Split input file on `// -----` string.

- **`--parsing-diagnostics`** -> parsing diagnose mode, i.e parse errors are not fatal for the whole run

- **`--print-generic`** -> prints the IR strictly in the generic format.

- **`-p`, `--passes`** -> the pass pipeline to apply to the IR: a comma-separated list of passes, each optionally followed by its options in braces, e.g. `Canonicalize,MyPass{maxIterations=3 verbose}`. See [pass options](transformations.md#pass-options). MLIR passes run through the `mlir-opt` pass, e.g. `-p 'mlir-opt[cse]'`; see [running MLIR passes](transformations.md#running-mlir-passes).

- **`--verify-diagnostics`** -> verification diagnose mode, i.e verification errors are not fatal for the whole run


## **Footnotes**

By default, the `scair-opt` executable is compiled with the Scala native compiler. 

However, we also provide the option to compile the executable with JVM (GraalVM by default) or GraalVM Native Image compiler.

**JVM:**
```bash
~/scair$ ./mill installJVM
```

**Graal Native Image:**
```bash
~/scair$ ./mill installGraal
```