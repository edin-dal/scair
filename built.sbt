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
  native_dialects,
  gen_dialects,
  transformations,
  tools
)

lazy val core = (project in file("core"))

lazy val ScaIRDL = (project in file("ScaIRDL")).dependsOn(core)
lazy val clair = (project in file("clair")).dependsOn(ScaIRDL)

lazy val native_dialects =
  (project in file("dialects")).dependsOn(clair)
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

lazy val dialect_source = settingKey[Seq[String]]("A list of things")
ThisBuild / dialect_source := Seq(
  "scair.dialects.example.ExampleDialect",
  "scair.dialects.cmath.CMathGen"
)

def generate_one_dialect_task(
    _dialect_source: String,
    dialect_gen_file: String
) = Def.task {
  val log = streams.value.log
  log.info(f"Generating dialect $dialect_gen_file")
  (native_dialects / Compile / runMain)
    .toTask(
      f" ${_dialect_source} $dialect_gen_file"
    )
    .value
  new File(dialect_gen_file)
}

def generate_all_dialects() = Def.taskDyn {
  val log = streams.value.log
  val managed_sources =
    (Compile / sourceManaged).value.getAbsolutePath()
  log.info(s"managed sources: $managed_sources")
  // Insert your generation logic here
  log.info("Running dialects generation...")
  val _dialect_sources = dialect_source.value
  // For example, running your generator logic:
  val dialect_gen_file = _dialect_sources.map(f =>
    f"$managed_sources/scala/${f.replace(".", "/")}_gen.scala"
  )

  (_dialect_sources, dialect_gen_file).zipped
    .map((s, g) => generate_one_dialect_task(s, g))
    .joinWith(_.join)
}

gen_dialects / Compile / sourceGenerators += generate_all_dialects().taskValue

// Give the poor generated sources what they need to compile
// (The project's class directory, i.e., whatever's already compiled (`scair. ...`))
gen_dialects / Compile / unmanagedClasspath += (Compile / classDirectory).value
// And the external dependencies (Things like The Scala stdlib and fastparse!)
gen_dialects / Compile / unmanagedClasspath ++= (Compile / externalDependencyClasspath).value

lazy val transformations =
  (project in file("transformations")).dependsOn(core, gen_dialects)
lazy val tools =
  (project in file("tools"))
    .dependsOn(gen_dialects, transformations)
    .enablePlugins(JavaAppPackaging)

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
