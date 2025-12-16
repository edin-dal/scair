ScaIR: MLIR inspired Scala Compiler Framework 
===
[![example branch parameter](https://github.com/edin-dal/scair/actions/workflows/tests.yml/badge.svg)](https://github.com/edin-dal/scair/actions/workflows/tests.yml/badge.svg?branch=main)
[![codecov](https://codecov.io/github/edin-dal/scair/graph/badge.svg?token=H3TBWG1YNT)](https://codecov.io/github/edin-dal/scair)
[![example branch parameter](https://img.shields.io/badge/license-Apache_2.0-blue)](https://github.com/edin-dal/scair/blob/main/LICENSE)

Please see [Scala Workshop talk](https://2025.workshop.scala-lang.org/details/scala-2025/7/ScaIR-Type-safe-Compiler-Framework-Compatible-with-MLIR) for paper and our talks

## Navigation
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Contributing to the Project](#contributing-to-the-project)
    - [Running](#running)
    - [Testing](#testing)  
    - [Code Formatting](#code-formatting) 

---

## Installation as a dependency

ScaIR is regularly published on [**Maven Central**](https://central.sonatype.com/artifact/io.github.edin-dal/scair-tools_3/overview)

Include in your project via: 

- SBT:
```scala
// To consume -SNAPSHOT versions (recommended)
resolvers += "Maven Central Snapshots" at "https://central.sonatype.com/repository/maven-snapshots"
libraryDependencies += "io.github.edin-dal" % "scair-tools_3" % "<version>"
```
- Mill:
```scala
    // To consume -SNAPSHOT versions (recommended)
    override def repositories = Seq("https://central.sonatype.com/repository/maven-snapshots")
    override def mvnDeps = Seq(
      ...,
      mvn"io.github.edin-dal::scair-tools_3:<version>",
      ...
    )
```

---

## Getting Started
Here are some tutorials explaining the core abstractions used in ScaIR, as well as the API's necesssary to define and transform IR within ScaIR:
- [Core Abstractions in ScaIR](https://edin-dal.github.io/scair/docs/core_abstractions.html)
- [Transformations](https://edin-dal.github.io/scair/docs/transformations.html)
- [CLI Interface](https://edin-dal.github.io/scair/docs/cli_interface.html)

We are actively working on more tutorials, this is what we have so far however more will be coming very soon!  


## Contributing to the project

You can contribute to the project by submitting a PR. We are currently setting up a Zulip channel to allow for a more direct communcation with us.

### Running

The **'./mill run'** command allows you to run your main classes in the project. This can be helpful in ad-hoc exploration of the compiler framework, or some localised testing as you are implementing new features (although we do recommend a proper testing suite once the the feature is ready for use).

To use the command you would need to have defined a new main class somewhere in the project. By default, we have no main classes, and as such the command will result in an error.

```
# No main class specified or found!
./mill run

# To run the main CLI:
./mill tools.opt.run
 
# To run the titlegen:
./mill clair.run
```

### Testing

Once your changes are ready to be merged ensure that both unit tests, as well as MLIR compatibility tests are passing. PRs will be blocked from merging otherwise. Our testing suite makes use of two Python3 packages: [**lit**](https://pypi.org/project/lit/) and [**filecheck**](https://pypi.org/project/filecheck/).

#### **Unit tests**
```
./mill test
```

#### **MLIR compatibility tests**
```
./mill filechecks
```

#### **Running the entire testing pipeline**
```
./mill testAll
```

### Code formatting
ScaIR project makes use of an auto-formatter, and some CI/CD checks on GitHub will fail if the code is not formatted properly. Additionally, we enforce that all unused imports be removed, automatic CI/CD checks will fail otherwise. Once you are ready to submit a PR for merging, run the following ./mill command to automatically format the entire code base:

```bash
./mill formatAll
```

To verify that the code has indeed been formatted, run:

```bash
./mill checkFormatAll
```
---
