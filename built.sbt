import sbt.internal.shaded.com.google.protobuf.compiler.PluginProtos.CodeGeneratorRequest
val scala_version = "3.3.1"

import scala.sys.process._
import java.io.File

ThisBuild / scalaVersion := scala_version

ThisBuild / libraryDependencies ++= Seq(
  "com.lihaoyi" %% "fastparse" % "3.1.0",
  "org.scalatest" % "scalatest_3" % "3.2.19" % "test",
  "com.github.scopt" %% "scopt" % "4.1.0"
)

lazy val root = (project in file(".")).aggregate(
  core,
  ScaIRDL,
  clair,
  dialects,
  transformations,
  tools
)

lazy val core = project in file("core")
lazy val ScaIRDL = project.dependsOn(core) in file("ScaIRDL")
lazy val clair = project.dependsOn(ScaIRDL) in file("clair")

lazy val native_dialects = project.dependsOn(clair) in file("dialects")
lazy val gen_dialects =
  project.dependsOn(native_dialects) in file("gen_dialects")

gen_dialects / Compile / sourceGenerators += Def.taskDyn {
  val managed_sources = (Compile / sourceManaged).value.getAbsolutePath()
  println(managed_sources)
  // Insert your generation logic here
  println("Running dialects generation...")
  // For example, running your generator logic:
  val dialect_source = "scair.dialects.example.ExampleDialect"
  val dialect_gen_file =
    f"$managed_sources/scala/${dialect_source.replace(".", "/")}.gen.scala"
  Def.task {
    (Compile / runMain)
      .toTask(
        f" $dialect_source $dialect_gen_file"
      )
      .value
    Seq[File](new File(dialect_gen_file))
  }
}.taskValue

// Give the poor generated sources what they need to compile
// (The project's class directory, i.e., whatever's already compiled (`scair. ...`))
gen_dialects / Compile / unmanagedClasspath += (Compile / classDirectory).value
// And the external dependencies (Things like The Scala stdlib and fastparse!)
gen_dialects / Compile / unmanagedClasspath ++= (Compile / externalDependencyClasspath).value

lazy val transformations =
  project.dependsOn(core, gen_dialects) in file("transformations")
lazy val tools =
  project
    .dependsOn(gen_dialects, transformations)
    .enablePlugins(JavaAppPackaging) in file("tools")

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
filechecks / fileInputs += (baseDirectory.value / "tests" / "filecheck").toGlob / ** / "*.{mlir,scala}"

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
