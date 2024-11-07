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

lazy val root = (project in file(".")).aggregate(core, ScaIRDL, clair, dialects, transformations, tools)

lazy val core = project in file("core")
lazy val ScaIRDL = project.dependsOn(core) in file("ScaIRDL")
lazy val clair = project.dependsOn(ScaIRDL) in file("clair")

lazy val dialects = project.dependsOn(clair) in file("dialects")
val mySourceGenerator = taskKey[Seq[File]]("...")

def generate_dialect(managed: File, def_file: File): Seq[File] = {
  println(s"managed: $managed")
  println(s"def_file: $def_file")
  val imp_path = s"${managed.getPath}/${def_file.base}.wow.scala"
  println(s"imp_path: $imp_path")
  val imp_file = new File(imp_path)

  IO.write(
    imp_file,
    s"""package scair.dialects.wooow
object Wow
  """
  )
  Seq(imp_file)
}

dialects / Compile / mySourceGenerator := generate_dialect(
  (dialects / Compile / sourceManaged).value,
  file("Affine/Affine_ops.scala")
)

dialects / Compile / sourceGenerators += (dialects / Compile / mySourceGenerator).taskValue

lazy val transformations =
  project.dependsOn(core, dialects) in file("transformations")
lazy val tools = (project in file("tools")).dependsOn(dialects, transformations).enablePlugins(JavaAppPackaging)

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
