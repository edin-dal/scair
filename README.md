ScaIR: MLIR inspired Scala Compiler Framework 
===
[![example branch parameter](https://github.com/edin-dal/scair/actions/workflows/tests.yml/badge.svg)](https://github.com/edin-dal/scair/actions/workflows/tests.yml/badge.svg?branch=main)
[![codecov](https://codecov.io/github/edin-dal/scair/graph/badge.svg?token=H3TBWG1YNT)](https://codecov.io/github/edin-dal/scair)
[![example branch parameter](https://img.shields.io/badge/license-Apache_2.0-blue)](https://github.com/edin-dal/scair/blob/main/LICENSE)

## Navigation
- [Installation](#installation)
- [Contributing to the Project](#contributing-to-the-project)
    - [Running](#running)
    - [Testing](#testing)  
    - [Code Formatting](#code-formatting) 

---

## Installation as a dependency

ScaIR is **now published** on [**Maven Central**](https://central.sonatype.com/artifact/io.github.edin-dal/scair-tools_3/0.0.0-1-670825/overview)! 🙌✊🥳🎉👏

Include in your project via: 

- SBT:
```scala
libraryDependencies += "io.github.edin-dal" % "scair-tools_3" % "<version>"
```
- Mill:
```scala
    def ivyDeps = Agg(
      ...,
      ivy"io.github.edin-dal::scair-tools_3:<version>",
      ...
    )
```

---

## Contributing to the project

You can contribute to the project by submitting a PR. We are currently setting up a Zulip channel to allow for a more direct communcation with us.

### Running

The **'./mill run'** command allows you to run your main classes in the project. This can be helpful in ad-hoc exploration of the compiler framework, or some localised testing as you are implementing new features (although we do recommend a proper testing suite once the the feature is ready for use).

To use the command you would need to have defined a new main class somewhere in the project. By default, we have no main classes, and as such the command will result in an error.

```
# No main class specified or found!
./mill run
# To run the main CLI:
./mill tools.run
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