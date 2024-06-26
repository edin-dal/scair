val scala_version = "2.12.13"

import scala.sys.process._

scalaVersion := scala_version

libraryDependencies ++= Seq(
  "org.scala-lang" % "scala-reflect" % scala_version,
  "com.lihaoyi" %% "fastparse" % "3.1.0",
  "org.scalatest" % "scalatest_2.12" % "3.1.1" % "test",
  "com.github.scopt" %% "scopt" % "4.1.0"
)

enablePlugins(JavaAppPackaging)

// Add .mlir files to watchSources, i.e., SBT can watch them to retrigger
// dependent tasks
watchSources += new WatchSource(
  baseDirectory.value,
  FileFilter.globFilter("*.mlir"),
  NothingFilter
)

//Define a filecheck SBT task
lazy val filechecks = taskKey[Unit]("File checks")
filechecks := {
  // It depends on scair-opt, built by the "stage" task currently
  stage.value
  // And then it's about running lit
  val r = ("lit tests/filecheck -v" !)
  if (r != 0) {
    sys.error("Filechecks failed")
  }
}
filechecks / fileInputs += (baseDirectory.value / "tests" / "filecheck" / "*.mlir").toGlob

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
