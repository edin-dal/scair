import java.io.PrintWriter
import sbt.util.FileInfo.full
import scala.sys.process._
import java.io.{File}
import java.nio.file.{Files, Path, StandardCopyOption}
import scala.collection.JavaConversions._


ThisBuild / scalaVersion := "3.3.4"
ThisBuild / semanticdbEnabled := true
ThisBuild / semanticdbVersion := scalafixSemanticdb.revision
ThisBuild / scalacOptions += "-Wunused:imports"

core / libraryDependencies += "com.lihaoyi" %% "fastparse" % "3.1.0"
ThisBuild / libraryDependencies += "org.scalatest" % "scalatest_3" % "3.2.19" % Test
tools / libraryDependencies += "com.github.scopt" %% "scopt" % "4.1.0"

lazy val scair = (project in file("."))
  .aggregate(
    core,
    ScaIRDL,
    clair,
    native_dialects,
    gen_dialects,
    transformations,
    tools
  )
  .enablePlugins(ScalaUnidocPlugin)

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
        "scair.dialects.affinegen.AffineGen",
        "scair.dialects.arithgen.ArithGen",
        "scair.dialects.cmathgen.CMathGen",
        "scair.dialects.funcgen.FuncGen",
        "scair.dialects.llvmgen.LLVMGen",
        "scair.dialects.memrefgen.MemrefGen"
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
    .settings(
      name := "scair"
    )

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

// Populate a file with the full classpath to compile some scala sources
lazy val filechecks_classpath = taskKey[Unit]("File checks classpath")
filechecks_classpath := {
  val full_cp = (tools / Compile / fullClasspath).value
  val file = new PrintWriter("full-classpath")
  file.print(
    full_cp.map(_.data).mkString(":")
  )
  file.flush()
  file.close()
}

//Define a filecheck SBT task
lazy val filechecks = taskKey[Unit]("File checks")
filechecks := {
  // It expects this task to populate a helper file
  (filechecks_classpath).value
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
  // Incremental unit tests
  (Test / testQuick).toTask("").value
  // Filechecks
  // Incrementality to add later!
  filechecks.value
}

addCommandAlias("formatCheckAll", "scalafixAll --check;scalafmtCheckAll;")

addCommandAlias("formatAll", "scalafixAll;scalafmtAll;")

// Just copy a source directory to a target repository...
def copyDir(source: Path, target: Path) : Unit = {
  if (!Files.exists(target)) {
      Files.createDirectories(target);
  }
  val stream = Files.newDirectoryStream(source)
  for (entry <- stream) {
      val newTarget = target.resolve(source.relativize(entry));
      if (Files.isDirectory(entry)) {
          copyDir(entry, newTarget);
      } else {
          Files.copy(entry, newTarget, StandardCopyOption.REPLACE_EXISTING);
      }
  }
}

// A task to install Scair in the user's standard home directories
// Linux-only a priori, as in $HOME/.local
lazy val install = taskKey[Unit]("Install Scair")
install := {
  (tools / stage).value
  val stage_dir = (tools / Universal  / stage).value.toString()
  val home_dir = sys.env.get("HOME").getOrElse(sys.error("nope"))
  val install_dir = s"$home_dir/.local"
  // To log messages
  val log = streams.value.log
  log.info(f"installing from $stage_dir to $install_dir")
  val target = new File(install_dir).toPath()
  val source = new File(stage_dir).toPath()
  copyDir(source, target)
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
