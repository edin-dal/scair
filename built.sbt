import sbt.internal.shaded.com.google.protobuf.compiler.PluginProtos.CodeGeneratorRequest
val scala_version = "3.3.1"

import scala.sys.process._
import java.io.File

ThisBuild / scalaVersion := scala_version

core / libraryDependencies += "com.lihaoyi" %% "fastparse" % "3.1.0"
Test / libraryDependencies += "org.scalatest" % "scalatest_3" % "3.2.19" % "test"
tools / libraryDependencies += "com.github.scopt" %% "scopt" % "4.1.0"

lazy val scair = (project in file(".")).aggregate(
  core,
  ScaIRDL,
  clair,
  native_dialects,
  gen_dialects,
  transformations,
  tools
)

lazy val core = (project in file("core"))

lazy val ScaIRDL = (project in file("ScaIRDL")).dependsOn(core)
lazy val clair = (project in file("clair")).dependsOn(ScaIRDL)

lazy val native_dialects: Project =
  (project in file("dialects"))
    .dependsOn(clair)

lazy val dialect_source =
  settingKey[Seq[String]]("A list of classes that generate dialects")

lazy val gen_dialects =
  (project in file("gen_dialects"))
    .dependsOn(native_dialects)
    .settings(
      dialect_source := Seq(
        "scair.dialects.example.ExampleDialect",
        "scair.dialects.cmathgen.CMathGen"
      ),
      // Add the generated sources to the source directories
      Compile / sourceGenerators += generate_all_dialects().taskValue
    )

lazy val transformations =
  (project in file("transformations")).dependsOn(core, gen_dialects)
lazy val tools =
  (project in file("tools"))
    .dependsOn(gen_dialects, transformations)
    .enablePlugins(JavaAppPackaging)

///////////////////////////
// Testing configuration //
///////////////////////////

// Add .mlir files to watchSources, i.e., SBT can watch them to retrigger
// dependent tasks
watchSources += new WatchSource(
  baseDirectory.value,
  FileFilter.globFilter("*.{mlir,scala}"),
  NothingFilter
)

//Define a filecheck SBT task
lazy val filechecks = taskKey[Unit]("File checks")
filechecks := {
  // It depends on scair-opt, built by the "stage" task currently
  (tools / stage).value
  // And then it's about running lit
  val r = ("lit tests/filecheck -v" !)
  if (r != 0) {
    sys.error("Filechecks failed")
  }
}
// Add a file dependency for automatic filecheck retriggering when any filecheck case changes.
filechecks / fileInputs += (baseDirectory.value / "tests" / "filecheck").toGlob / ** / "*.{mlir,scala}"

// Define a testAll to run all tests types of Scair
// Nicety to have sbt ~testAll running when develloping features or content!
lazy val testAll = taskKey[Unit]("Run all tests")
testAll := {
  // Incremental format check
  scalafmtCheckAll.value
  // Incremental unit tests
  (Test / testQuick).toTask("").value
  // Filechecks
  // Incrementality to add later!
  filechecks.value
}

///////////////////////
// Utility functions //
///////////////////////
// consider factoring out

// Create a task generating a single dialect
def generate_one_dialect_task(
    // The dialect generation source file
    _dialect_source: String,
    // The generated dialect destination file
    dialect_gen_file: String
) = Def.task {
  // To log messages
  val log = streams.value.log

  // Run the dialect generation
  log.info(f"Generating dialect from $dialect_gen_file")
  // This currently is done by running the main function of the dialect generation
  // source file. TODO better!
  (native_dialects / Compile / runMain)
    .toTask(
      f" ${_dialect_source} $dialect_gen_file"
    )
    .value
  // Return the generated file for SBT handling
  new File(dialect_gen_file)
}

// Create a dynamic task aggregating all generated dialects
def generate_all_dialects() = Def.taskDyn {
  // To log messages
  val log = streams.value.log

  // Retrieve the managed sources directory, to put generated code in
  val managed_sources =
    (Compile / sourceManaged).value.getAbsolutePath()
  log.info(s"managed sources: $managed_sources")

  // Retrieve dialect generation sources and prepare paths
  val _dialect_sources = dialect_source.value
  val dialect_gen_file = _dialect_sources.map(f =>
    f"$managed_sources/scala/${f.replace(".", "/")}_gen.scala"
  )

  // Create generation tasks
  log.info("Creating dialects generation tasks...")
  (_dialect_sources, dialect_gen_file).zipped
    .map((s, g) => generate_one_dialect_task(s, g))
    .joinWith(_.join)
}
