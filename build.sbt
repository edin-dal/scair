/*  previous old build file for reference
def scoverageVersion = Task("2.4.1")

override def scalaVersion = "3.7.4"

override def scalacOptions =
  super.scalacOptions() ++ Seq("-Wunused:imports", "-explain")

override def artifactName = s"scair-${super.artifactName()}"

override def publishVersion =
  VcsVersion.vcsState().format(untaggedSuffix = "-SNAPSHOT")

override def pomSettings = PomSettings(
  description = artifactName(),
  organization = "io.github.edin-dal",
  url = "https://github.com/edin-dal/scair",
  licenses = Seq(License.`Apache-2.0`),
  versionControl = VersionControl.github(owner = "edin-dal", repo = "scair"),
  // TODO ?
  developers = Seq(
    Developer(
      id = "baymaks",
      name = "Maks Kret",
      url = "https://github.com/baymaks/",
    ),
    Developer(
      id = "papychacal",
      name = "Emilien Bauer",
      url = "https://github.com/PapyChacal/",
    ),
  ),
)
 */

val commonSettings = Seq(
  scalaVersion := "3.7.4",
  scalacOptions ++= Seq("-Wunused:imports", "-explain"),
  organization := "io.github.edin-dal",
  publishMavenStyle := true,
  licenses += ("Apache-2.0", url("http://www.apache.org/licenses/LICENSE-2.0")),
)

lazy val utils = project.in(file("utils")).settings(
  commonSettings,
  name := "scair-utils",
)

lazy val core = project.in(file("core")).dependsOn(utils).settings(
  commonSettings,
  name := "scair-core",
  libraryDependencies ++= Seq(
    // fastparse for parsing
    "com.lihaoyi" %% "fastparse" % "3.1.1",
    "org.scalatest" %% "scalatest" % "3.2.19" % Test,
  ),
)
