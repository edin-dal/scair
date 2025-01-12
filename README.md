ScaIR: MLIR inspired Scala Compiler Framework 
===
[![example branch parameter](https://github.com/edin-dal/scair/actions/workflows/tests.yml/badge.svg)](https://github.com/edin-dal/scair/actions/workflows/tests.yml/badge.svg?branch=main)
[![example branch parameter](https://img.shields.io/badge/license-Apache_2.0-blue)](https://github.com/edin-dal/scair/blob/main/LICENSE)

<hr style="height: 10px; background-color: green; border: none;">

## Navigation
- [Installation](#installation)
- [Contributing to the Project](#contributing-to-the-project)
    - [Running](#running-sbt)
    - [Testing](#testing)  
    - [Code Formatting](#code-formatting) 

<hr style="height: 10px; background-color: green; border: none;">

# Installation
The project is implemented under the Scala version 3.3.4, however newer versions should work just as fine! Check out the official **[Getting started](https://docs.scala-lang.org/getting-started/install-scala.html#:~:text=Using%20the%20Scala%20Installer%20(recommended%20way)&text=Install%20it%20on%20your%20system%20with%20the%20following%20instructions.&text=%26%26%20.%2Fcs%20setup-,Run%20the%20following%20command%20in%20your,following%20the%20on%2Dscreen%20instructions.&text=Download%20and%20execute%20the%20Scala,follow%20the%20on%2Dscreen%20instructions.)** guide on how to install Scala and sbt.

Additionally, our testing suite makes use of two Python3 packages: [**lit**](https://pypi.org/project/lit/) and [**filecheck**](https://pypi.org/project/filecheck/0.0.13/).

<hr style="height: 10px; background-color: green; border: none;">

# Contributing to the project

You can contribute to the project by submitting a PR. We are currently setting up a Zulip channel to allow for a more direct communcation with us.

<hr style="height:2px; background-color: green; border: none;">

## Running sbt

The **'sbt run'** command allows you to run your main classes in the project. This can be helpful in ad-hoc exploration of the compiler framework, or some localised testing as you are implementing new features (although we do recommend a proper testing suite once the the feature is ready for use).

To use the command you would need to have defined a new main class somewhere in the project. By default, we have no main classes, and as such the command will result in an error.

```
sbt run
```

<hr style="height:2px; background-color: green; border: none;">

## Testing

### **Unit tests**
```
sbt test
```

### **MLIR compatibility tests**
```
sbt filechecks
```

### **Running the entire testing pipeline**
```
sbt testAll
```

<hr style="height:2px; background-color: green; border: none;">

## Code formatting
ScaIR project makes use of an auto-formatter, and some CI/CD checks on GitHub might fail if the code is not formatted properly.

**Command for formatting the code**
```
sbt scalafmtAll
```

<hr style="height: 10px; background-color: green; border: none;">