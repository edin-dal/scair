ScaIR: MLIR inspired Scala Compiler Framework 
===
[![example branch parameter](https://github.com/edin-dal/scair/actions/workflows/tests.yml/badge.svg)](https://github.com/edin-dal/scair/actions/workflows/tests.yml/badge.svg?branch=main)
[![codecov](https://codecov.io/github/edin-dal/scair/graph/badge.svg?token=H3TBWG1YNT)](https://codecov.io/github/edin-dal/scair)
[![example branch parameter](https://img.shields.io/badge/license-Apache_2.0-blue)](https://github.com/edin-dal/scair/blob/main/LICENSE)

## Navigation
- [Installation](#installation)
- [Contributing to the Project](#contributing-to-the-project)
    - [Running](#running-sbt)
    - [Testing](#testing)  
    - [Code Formatting](#code-formatting) 

---

## Installation
The project is implemented under the Scala version 3.3.4, however newer versions should work just as fine! Check out the official **[Getting started](https://docs.scala-lang.org/getting-started/install-scala.html#:~:text=Using%20the%20Scala%20Installer%20(recommended%20way)&text=Install%20it%20on%20your%20system%20with%20the%20following%20instructions.&text=%26%26%20.%2Fcs%20setup-,Run%20the%20following%20command%20in%20your,following%20the%20on%2Dscreen%20instructions.&text=Download%20and%20execute%20the%20Scala,follow%20the%20on%2Dscreen%20instructions.)** guide on how to install Scala and sbt.

### As a dependency

We are still figuring out the Maven Central repository publication process.

ScaIR is usable now through local publication though:

1. Clone the repository and publish its packages locally by running:

```bash
sbt publishLocal
```

Which should end with a line like:

```
[info]  published ivy to <USER-PATH>/.ivy2/local/io.github.edin-dal/scair_3/<version>/ivys/ivy.xml
```

To use this from your project, just add:

```scala
libraryDependencies += "io.github.edin-dal" %% "scair" % "<version>"
```

to your build defintion (Typically `build.sbt`)

---

## Contributing to the project

You can contribute to the project by submitting a PR. We are currently setting up a Zulip channel to allow for a more direct communcation with us.

### Running sbt

The **'sbt run'** command allows you to run your main classes in the project. This can be helpful in ad-hoc exploration of the compiler framework, or some localised testing as you are implementing new features (although we do recommend a proper testing suite once the the feature is ready for use).

To use the command you would need to have defined a new main class somewhere in the project. By default, we have no main classes, and as such the command will result in an error.

```
sbt run
```

### Testing

Once your changes are ready to be merged ensure that both unit tests, as well as MLIR compatibility tests are passing. PRs will be blocked from merging otherwise. Our testing suite makes use of two Python3 packages: [**lit**](https://pypi.org/project/lit/) and [**filecheck**](https://pypi.org/project/filecheck/).

#### **Unit tests**
```
sbt test
```

#### **MLIR compatibility tests**
```
sbt filechecks
```

#### **Running the entire testing pipeline**
```
sbt testAll
```

### Code formatting
ScaIR project makes use of an auto-formatter, and some CI/CD checks on GitHub will fail if the code is not formatted properly. Once you are ready to submit a PR for merging, run the following sbt command to automatically format the entire code base:
```
sbt scalafmtAll
```

Additionally, we enforce that all unused imports be removed, automatic CI/CD checks will fail otherwise. To check for unused imports run the following sbt command:
```
sbt "scalafixAll --check"
```


---