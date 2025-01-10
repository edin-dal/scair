ScaIR: MLIR inspired Scala Compiler Framework 
===

### Works with versions:

Scala 3.3.4

#### [Guide to installing Scala and SBT](https://docs.scala-lang.org/getting-started/install-scala.html#:~:text=Using%20the%20Scala%20Installer%20(recommended%20way)&text=Install%20it%20on%20your%20system%20with%20the%20following%20instructions.&text=%26%26%20.%2Fcs%20setup-,Run%20the%20following%20command%20in%20your,following%20the%20on%2Dscreen%20instructions.&text=Download%20and%20execute%20the%20Scala,follow%20the%20on%2Dscreen%20instructions.)

### Running

```
sbt run
```

### Testing 

**Local unit tests**
```
sbt test
```

**MLIR compatibility tests**
```
sbt filechecks
```

**Run the entire testing pipeline**
```
sbt testAll
```